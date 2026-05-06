import csv
import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from skimage.metrics import structural_similarity

from videox_fun.utils.optical_flow_utils import RAFTFlowExtractor


def normalize_video_for_metrics(video: torch.Tensor, valid_frames: Optional[int] = None) -> torch.Tensor:
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected [B,C,T,H,W] or [C,T,H,W], got shape {tuple(video.shape)}")

    if video.shape[0] != 3 and video.shape[1] == 3:
        video = video.permute(1, 0, 2, 3)
    if video.shape[0] != 3:
        raise ValueError(f"Expected RGB channels, got shape {tuple(video.shape)}")

    frames = video.detach().float().permute(1, 0, 2, 3).contiguous()
    if valid_frames is not None:
        frames = frames[:valid_frames]

    if frames.numel() == 0:
        raise ValueError("Video has no frames after normalization")

    if frames.max() > 1.5:
        frames = frames / 255.0
    elif frames.min() < 0.0:
        frames = (frames + 1.0) / 2.0

    return frames.clamp(0.0, 1.0)


def _compute_psnr(frame1: np.ndarray, frame2: np.ndarray) -> float:
    mse = float(np.mean((frame1 - frame2) ** 2))
    mse = max(mse, 1e-10)
    return float(10.0 * math.log10(1.0 / mse))


def _normalized_base_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    y_coords = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)


def backward_warp(frame: torch.Tensor, backward_flow: torch.Tensor) -> torch.Tensor:
    _, _, height, width = frame.shape
    base_grid = _normalized_base_grid(height, width, frame.device, frame.dtype)

    flow_x = backward_flow[:, 0] * (2.0 / max(width - 1, 1))
    flow_y = backward_flow[:, 1] * (2.0 / max(height - 1, 1))
    flow_grid = torch.stack((flow_x, flow_y), dim=-1)
    sample_grid = base_grid + flow_grid

    return F.grid_sample(
        frame,
        sample_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def summarize_pair_metrics(pair_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    metric_names = [
        "lpips_temporal",
        "optical_flow_consistency_l1",
        "psnr_consecutive",
        "ssim_consecutive",
    ]
    summary: Dict[str, float] = {}

    for metric_name in metric_names:
        values = [float(pair[metric_name]) for pair in pair_metrics]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(array.mean())
        summary[f"{metric_name}_std"] = float(array.std())
        summary[f"{metric_name}_min"] = float(array.min())
        summary[f"{metric_name}_max"] = float(array.max())

    summary["num_pairs"] = len(pair_metrics)
    return summary


class TemporalMetricsEvaluator:
    def __init__(
        self,
        device: Optional[str] = None,
        lpips_model: Optional[torch.nn.Module] = None,
        flow_extractor: Optional[RAFTFlowExtractor] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._lpips_model = lpips_model
        self._flow_extractor = flow_extractor

    def _get_lpips_model(self) -> torch.nn.Module:
        if self._lpips_model is None:
            import lpips

            self._lpips_model = lpips.LPIPS(net="alex").to(self.device)
            self._lpips_model.eval()
        return self._lpips_model

    def _get_flow_extractor(self) -> RAFTFlowExtractor:
        if self._flow_extractor is None:
            self._flow_extractor = RAFTFlowExtractor(device=self.device)
        return self._flow_extractor

    def _compute_pair_metrics(self, frame1: torch.Tensor, frame2: torch.Tensor) -> Dict[str, float]:
        frame1_cpu = frame1.detach().cpu()
        frame2_cpu = frame2.detach().cpu()
        frame1_np = frame1_cpu.permute(1, 2, 0).numpy()
        frame2_np = frame2_cpu.permute(1, 2, 0).numpy()

        psnr = _compute_psnr(frame1_np, frame2_np)

        min_hw = min(frame1_np.shape[0], frame1_np.shape[1])
        win_size = min(7, min_hw)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        ssim = float(
            structural_similarity(
                frame1_np,
                frame2_np,
                channel_axis=2,
                data_range=1.0,
                win_size=win_size,
            )
        )

        lpips_model = self._get_lpips_model()
        lpips_input_1 = frame1.unsqueeze(0).to(self.device) * 2.0 - 1.0
        lpips_input_2 = frame2.unsqueeze(0).to(self.device) * 2.0 - 1.0
        with torch.no_grad():
            lpips_value = float(lpips_model(lpips_input_1, lpips_input_2).mean().item())

        flow_extractor = self._get_flow_extractor()
        flow_frame1 = frame1.unsqueeze(0).to(self.device)
        flow_frame2 = frame2.unsqueeze(0).to(self.device)
        with torch.no_grad():
            backward_flow = flow_extractor.extract_flow(flow_frame2, flow_frame1)
            warped_frame1 = backward_warp(flow_frame1, backward_flow)
            valid_mask = backward_warp(
                torch.ones((1, 1, frame1.shape[1], frame1.shape[2]), device=self.device, dtype=flow_frame1.dtype),
                backward_flow,
            )

        diff = torch.abs(warped_frame1 - flow_frame2)
        valid_pixels = valid_mask > 0.5
        if valid_pixels.any():
            flow_consistency = float(diff.masked_select(valid_pixels.expand_as(diff)).mean().item())
        else:
            flow_consistency = float(diff.mean().item())

        return {
            "lpips_temporal": lpips_value,
            "optical_flow_consistency_l1": flow_consistency,
            "psnr_consecutive": psnr,
            "ssim_consecutive": ssim,
        }

    def evaluate(
        self,
        video: torch.Tensor,
        *,
        video_name: Optional[str] = None,
        stage: Optional[str] = None,
        fps: Optional[int] = None,
        valid_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        frames = normalize_video_for_metrics(video, valid_frames=valid_frames)
        if frames.shape[0] < 2:
            raise ValueError("Temporal metrics require at least two frames")

        pair_metrics: List[Dict[str, float]] = []
        for index in range(frames.shape[0] - 1):
            metrics = self._compute_pair_metrics(frames[index], frames[index + 1])
            pair_metrics.append({
                "t": index,
                "t_next": index + 1,
                **metrics,
            })

        return {
            "video_name": video_name,
            "stage": stage,
            "num_frames": int(frames.shape[0]),
            "num_pairs": int(frames.shape[0] - 1),
            "fps": fps,
            "summary": summarize_pair_metrics(pair_metrics),
            "pairs": pair_metrics,
        }

    @staticmethod
    def format_summary(result: Dict[str, Any]) -> str:
        summary = result["summary"]
        video_name = result.get("video_name") or "video"
        stage = result.get("stage") or "temporal_eval"
        return (
            f"[temporal_eval] {video_name} {stage} | pairs={summary['num_pairs']} | "
            f"lpips={summary['lpips_temporal_mean']:.4f} | "
            f"ofc_l1={summary['optical_flow_consistency_l1_mean']:.4f} | "
            f"psnr={summary['psnr_consecutive_mean']:.2f} | "
            f"ssim={summary['ssim_consecutive_mean']:.4f}"
        )

    @staticmethod
    def write(result: Dict[str, Any], output_prefix: str) -> Dict[str, str]:
        json_path = f"{output_prefix}_temporal_metrics.json"
        csv_path = f"{output_prefix}_temporal_metrics_pairs.csv"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(result, json_file, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "video_name",
                "stage",
                "t",
                "t_next",
                "lpips_temporal",
                "optical_flow_consistency_l1",
                "psnr_consecutive",
                "ssim_consecutive",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for pair in result["pairs"]:
                writer.writerow({
                    "video_name": result.get("video_name"),
                    "stage": result.get("stage"),
                    **pair,
                })

        logger.info(TemporalMetricsEvaluator.format_summary(result))
        return {"json": json_path, "csv": csv_path}


def compute_temporal_metrics(
    video: torch.Tensor,
    *,
    enable_lpips: bool = True,
    enable_flow: bool = True,
    lpips_net: str = "alex",
    flow_model: str = "raft_large",
    max_frames: int = 65,
    eval_size: int = 256,
    fb_alpha: float = 0.01,
    fb_beta: float = 0.5,
) -> Dict[str, Any]:
    del enable_lpips, enable_flow, lpips_net, flow_model, fb_alpha, fb_beta

    if video.ndim == 4:
        video = video.unsqueeze(0)

    if eval_size and eval_size > 0:
        if video.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video.shape)}")
        resized_frames = []
        for t in range(video.shape[2]):
            frame = video[:, :, t]
            frame = F.interpolate(frame, size=(eval_size, eval_size), mode="area")
            resized_frames.append(frame)
        video = torch.stack(resized_frames, dim=2)

    evaluator = TemporalMetricsEvaluator()
    valid_frames = min(max_frames, video.shape[2]) if max_frames and max_frames > 0 else None
    result = evaluator.evaluate(
        video,
        video_name="sequence",
        stage="temporal_eval",
        valid_frames=valid_frames,
    )
    return result["summary"]


def save_temporal_metrics_json(payload: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)
