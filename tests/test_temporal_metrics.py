import csv
import json
import tempfile
import unittest

import torch

from videox_fun.utils.temporal_metrics import TemporalMetricsEvaluator, normalize_video_for_metrics


class DummyLPIPS(torch.nn.Module):
    def forward(self, frame1, frame2):
        return torch.abs(frame1 - frame2).mean(dim=(1, 2, 3), keepdim=True)


class DummyFlowExtractor:
    def extract_flow(self, frame1, frame2):
        batch_size, _, height, width = frame1.shape
        return torch.zeros((batch_size, 2, height, width), dtype=frame1.dtype, device=frame1.device)


class TemporalMetricsTests(unittest.TestCase):
    def test_normalize_video_from_minus_one_to_one(self):
        video = torch.tensor(
            [[[[[-1.0, 1.0], [0.0, 0.5]], [[-1.0, 1.0], [0.0, 0.5]]]]],
            dtype=torch.float32,
        ).repeat(1, 3, 2, 1, 1)
        frames = normalize_video_for_metrics(video, valid_frames=1)
        self.assertEqual(frames.shape, (1, 3, 2, 2))
        self.assertGreaterEqual(float(frames.min()), 0.0)
        self.assertLessEqual(float(frames.max()), 1.0)

    def test_evaluate_and_write_sidecars(self):
        evaluator = TemporalMetricsEvaluator(
            device="cpu",
            lpips_model=DummyLPIPS(),
            flow_extractor=DummyFlowExtractor(),
        )

        video = torch.zeros((1, 3, 3, 8, 8), dtype=torch.float32)
        video[:, :, 1] = 0.25
        video[:, :, 2] = 0.5

        result = evaluator.evaluate(video, video_name="demo", stage="pass1", fps=12)
        self.assertEqual(result["num_frames"], 3)
        self.assertEqual(result["num_pairs"], 2)
        self.assertEqual(len(result["pairs"]), 2)
        self.assertIn("lpips_temporal_mean", result["summary"])

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = evaluator.write(result, f"{temp_dir}/demo")
            with open(paths["json"], "r", encoding="utf-8") as json_file:
                written_json = json.load(json_file)
            self.assertEqual(written_json["video_name"], "demo")

            with open(paths["csv"], "r", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["stage"], "pass1")


if __name__ == "__main__":
    unittest.main()
