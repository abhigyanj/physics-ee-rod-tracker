#!/usr/bin/env python3

import math
import unittest

import cv2
import numpy as np

from tracker_engine import (
    Calibration,
    build_sampling_grid,
    calibrated_shaft_scores,
    normalize_observation_scores,
    refine_path,
    shortest_axis_delta,
    viterbi_path,
)


class TrackerEngineTests(unittest.TestCase):
    def test_normalization_preserves_a_unique_strong_peak(self):
        raw = np.zeros(360, dtype=np.float32)
        raw[250:257] = [1.0, 1.4, 1.8, 2.4, 1.9, 1.5, 1.1]
        normalized = normalize_observation_scores(raw)
        self.assertEqual(int(np.argmax(normalized)), 253)
        self.assertEqual(int(np.sum(normalized == normalized.max())), 1)

    def test_calibrated_shaft_recovers_a_synthetic_line(self):
        size = 520
        pivot = size / 2
        radius = 210.0
        expected = 37.3
        calibration = Calibration(
            pivot_x=pivot,
            pivot_y=pivot,
            end1_x=pivot + radius,
            end1_y=pivot,
            end2_x=pivot - radius,
            end2_y=pivot,
            radius=radius,
            lower_bound=(12, 100, 150),
            upper_bound=(30, 255, 255),
        )
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        hsv_color = np.uint8([[[20, 220, 240]]])
        color = tuple(int(value) for value in cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0])
        theta = math.radians(expected)
        point_a = (
            int(round(pivot + radius * math.cos(theta))),
            int(round(pivot - radius * math.sin(theta))),
        )
        point_b = (
            int(round(pivot - radius * math.cos(theta))),
            int(round(pivot + radius * math.sin(theta))),
        )
        cv2.line(frame, point_a, point_b, color, 9, cv2.LINE_AA)
        grid = build_sampling_grid(calibration, (0, size, 0, size))
        scores, _metrics = calibrated_shaft_scores(frame, grid)
        path = np.asarray([int(np.argmax(scores))], dtype=np.int32)
        recovered = float(refine_path(scores[None, :], path)[0])
        error = abs(float(shortest_axis_delta(recovered, expected)))
        self.assertLess(error, 0.5)

    def test_hard_step_viterbi_does_not_bias_valid_motion(self):
        states = 360
        expected = np.arange(20, 90, 3, dtype=np.int32)
        scores = np.full((len(expected), states), -2.0, dtype=np.float32)
        for row, state in enumerate(expected):
            scores[row, state] = 5.0
        path = viterbi_path(scores, initial_angle=float(expected[0]) * 0.5)
        np.testing.assert_array_equal(path, expected)


if __name__ == "__main__":
    unittest.main()
