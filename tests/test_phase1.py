from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from iris_phase1.labels import parse_choice
from iris_phase1.video import VideoDecodeError, sample_video


class LabelParsingTests(unittest.TestCase):
    def test_accepts_single_choice(self) -> None:
        self.assertEqual(parse_choice(" C\n"), ("C", "Sliding Cone"))

    def test_rejects_explanation(self) -> None:
        self.assertEqual(parse_choice("A. Dropping Ball"), (None, None))


class SamplingTests(unittest.TestCase):
    def test_order_and_short_video_repetition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.avi"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 16)
            )
            self.assertTrue(writer.isOpened())
            for value in (0, 80, 160):
                writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
            writer.release()

            sampled = sample_video(path, 8)
            self.assertEqual(sampled.source_frame_indices, [0, 0, 1, 1, 1, 1, 2, 2])
            self.assertEqual(sampled.timestamps_seconds, [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2])
            means = [float(np.asarray(image).mean()) for image in sampled.frames]
            self.assertEqual(means, sorted(means))

            center = sample_video(path, 1)
            self.assertEqual(center.source_frame_indices, [1])
            self.assertEqual(center.timestamps_seconds, [0.1])

    def test_missing_video_is_a_decode_error(self) -> None:
        with self.assertRaises(VideoDecodeError):
            sample_video("/definitely/not/a/video.mp4", 1)


if __name__ == "__main__":
    unittest.main()
