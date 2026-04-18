import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _to_tchw(video: torch.Tensor) -> torch.Tensor:
    """Converts video to [T, C, H, W] float tensor in [0, 1]."""
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor, got shape {tuple(video.shape)}")

    # Prefer [T, C, H, W] when second dim is a valid channel count.
    # Otherwise, treat as [C, T, H, W].
    if video.shape[1] in (1, 3, 4):
        pass
    elif video.shape[0] in (1, 3, 4):
        video = video.permute(1, 0, 2, 3)
    else:
        raise ValueError(f"Could not infer video layout from shape {tuple(video.shape)}")

    if video.shape[1] not in (1, 3, 4):
        raise ValueError(f"Could not interpret channel dimension in shape {tuple(video.shape)}")

    if video.shape[1] > 3:
        video = video[:, :3]

    video = video.float()
    if video.min() < 0.0:
        video = (video + 1.0) / 2.0
    return video.clamp(0.0, 1.0)


def _resize_video(video_tchw: torch.Tensor, eval_size: Optional[int]) -> torch.Tensor:
    if eval_size is None or eval_size <= 0:
        return video_tchw

    _, _, h, w = video_tchw.shape
    short_side = min(h, w)
    if short_side <= eval_size:
        return video_tchw

    scale = eval_size / float(short_side)
    target_h = max(8, int(round(h * scale / 8.0) * 8))
    target_w = max(8, int(round(w * scale / 8.0) * 8))

    return F.interpolate(video_tchw, size=(target_h, target_w), mode="bilinear", align_corners=False)


def _sample_temporal(video_tchw: torch.Tensor, max_frames: Optional[int]) -> torch.Tensor:
    if max_frames is None or max_frames <= 0 or video_tchw.shape[0] <= max_frames:
        return video_tchw

    indices = torch.linspace(0, video_tchw.shape[0] - 1, max_frames).long()
    return video_tchw[indices]


def _warp_tensor(source: torch.Tensor, flow_xy: torch.Tensor) -> torch.Tensor:
    """Warps source [B, C, H, W] using pixel-space flow [B, 2, H, W]."""
    b, _, h, w = source.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=source.device, dtype=source.dtype),
        torch.arange(w, device=source.device, dtype=source.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    sample_grid = base_grid + flow_xy.permute(0, 2, 3, 1)

    sample_grid[..., 0] = 2.0 * sample_grid[..., 0] / max(w - 1, 1) - 1.0
    sample_grid[..., 1] = 2.0 * sample_grid[..., 1] / max(h - 1, 1) - 1.0

    return F.grid_sample(source, sample_grid, mode="bilinear", padding_mode="border", align_corners=True)


def _compute_lpips_temporal(
    video_tchw: torch.Tensor,
    device: torch.device,
    lpips_net: str,
) -> Dict[str, object]:
    try:
        import lpips  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "error": f"lpips package unavailable: {exc}",
        }

    model = lpips.LPIPS(net=lpips_net).to(device).eval()
    values = []

    with torch.no_grad():
        for t in range(video_tchw.shape[0] - 1):
            x = (video_tchw[t : t + 1].to(device) * 2.0) - 1.0
            y = (video_tchw[t + 1 : t + 2].to(device) * 2.0) - 1.0
            d = model(x, y)
            values.append(float(d.mean().item()))

    if not values:
        return {
            "available": False,
            "error": "Not enough frames for LPIPS temporal metric.",
        }

    return {
        "available": True,
        "mean": float(sum(values) / len(values)),
        "per_pair": values,
        "pairs": len(values),
    }


def _compute_flow_consistency(
    video_tchw: torch.Tensor,
    device: torch.device,
    flow_model: str,
    fb_alpha: float,
    fb_beta: float,
) -> Dict[str, object]:
    try:
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )
    except Exception as exc:  # pragma: no cover
        return {
            "available": False,
            "error": f"Torchvision optical-flow models unavailable: {exc}",
        }

    flow_model = flow_model.lower()
    if flow_model == "raft_large":
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False)
    else:
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False)

    model = model.to(device).eval()
    transforms = weights.transforms()

    fb_values = []
    photo_values = []
    occ_values = []

    with torch.no_grad():
        for t in range(video_tchw.shape[0] - 1):
            img_t = video_tchw[t : t + 1].to(device)
            img_t1 = video_tchw[t + 1 : t + 2].to(device)

            # Forward and backward flow
            t_fwd, t1_fwd = transforms(img_t, img_t1)
            flow_fwd = model(t_fwd, t1_fwd)[-1]

            t1_bwd, t_bwd = transforms(img_t1, img_t)
            flow_bwd = model(t1_bwd, t_bwd)[-1]

            # Forward-backward consistency in frame t coordinates
            flow_bwd_warped = _warp_tensor(flow_bwd, flow_fwd)
            fb_error = torch.linalg.norm(flow_fwd + flow_bwd_warped, dim=1)

            mag = torch.linalg.norm(flow_fwd, dim=1) + torch.linalg.norm(flow_bwd_warped, dim=1)
            occluded = fb_error > (fb_alpha * mag + fb_beta)
            valid = ~occluded
            occ_ratio = float(occluded.float().mean().item())

            if valid.any():
                fb_values.append(float(fb_error[valid].mean().item()))

                # Photometric warp error (lower is better)
                img_t1_to_t = _warp_tensor(img_t1, flow_fwd)
                photo = torch.abs(img_t1_to_t - img_t).mean(dim=1)
                photo_values.append(float(photo[valid].mean().item()))
            else:
                fb_values.append(float(fb_error.mean().item()))
                img_t1_to_t = _warp_tensor(img_t1, flow_fwd)
                photo = torch.abs(img_t1_to_t - img_t).mean(dim=1)
                photo_values.append(float(photo.mean().item()))

            occ_values.append(occ_ratio)

    if not fb_values:
        return {
            "available": False,
            "error": "Not enough frames for flow consistency metric.",
        }

    return {
        "available": True,
        "flow_model": flow_model,
        "fb_consistency_mean": float(sum(fb_values) / len(fb_values)),
        "warp_l1_mean": float(sum(photo_values) / len(photo_values)),
        "occlusion_ratio_mean": float(sum(occ_values) / len(occ_values)),
        "pairs": len(fb_values),
    }


def compute_temporal_metrics(
    video_4d: torch.Tensor,
    *,
    enable_lpips: bool = True,
    enable_flow: bool = True,
    lpips_net: str = "alex",
    flow_model: str = "raft_small",
    max_frames: int = 65,
    eval_size: int = 256,
    fb_alpha: float = 0.01,
    fb_beta: float = 0.5,
) -> Dict[str, object]:
    """
    Computes temporal quality metrics for generated videos.

    Returns a dictionary with metric values and any recoverable errors.
    """
    video_tchw = _to_tchw(video_4d.detach().cpu())
    video_tchw = _sample_temporal(video_tchw, max_frames)
    video_tchw = _resize_video(video_tchw, eval_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result: Dict[str, object] = {
        "frames_evaluated": int(video_tchw.shape[0]),
        "resolution_evaluated": [int(video_tchw.shape[2]), int(video_tchw.shape[3])],
        "lpips_temporal": {"available": False, "error": "disabled"},
        "optical_flow_consistency": {"available": False, "error": "disabled"},
    }

    if video_tchw.shape[0] < 2:
        result["error"] = "Need at least 2 frames to evaluate temporal metrics."
        return result

    if enable_lpips:
        result["lpips_temporal"] = _compute_lpips_temporal(
            video_tchw=video_tchw,
            device=device,
            lpips_net=lpips_net,
        )

    if enable_flow:
        result["optical_flow_consistency"] = _compute_flow_consistency(
            video_tchw=video_tchw,
            device=device,
            flow_model=flow_model,
            fb_alpha=fb_alpha,
            fb_beta=fb_beta,
        )

    return result


def save_temporal_metrics_json(metrics: Dict[str, object], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
