import math
import unittest

import cv2
import numpy as np

from rod_tracker.uncertainty import (
    displacement_standard_uncertainty, fit_mask_angle, fit_section_centers,
)


class MaskUncertaintyTests(unittest.TestCase):
    def test_standard_uncertainty_has_expected_coverage_for_noisy_lines(self):
        rng = np.random.default_rng(7391)
        along = np.r_[np.linspace(-300, -60, 12), np.linspace(60, 300, 12)]
        covered, estimates = 0, []
        for _ in range(600):
            fit = fit_section_centers(along, rng.normal(0, 1.5, len(along)), 37.0)
            difference = (fit["angleDegrees"] - 37.0 + 90) % 180 - 90
            covered += abs(difference) <= fit["standardUncertaintyDegrees"]
            estimates.append(fit["standardUncertaintyDegrees"])
        self.assertGreater(covered / 600, 0.60)
        self.assertLess(covered / 600, 0.75)
        self.assertGreater(np.mean(estimates), 0)

    def test_noisier_masks_have_larger_fit_uncertainty(self):
        along = np.r_[np.linspace(-300, -60, 12), np.linspace(60, 300, 12)]
        deviations = np.random.default_rng(91).normal(size=len(along))
        clean = fit_section_centers(along, deviations * 0.5, 25.0)
        noisy = fit_section_centers(along, deviations * 3.0, 25.0)
        self.assertGreater(noisy["standardUncertaintyDegrees"], 5 * clean["standardUncertaintyDegrees"])

    def test_mask_fit_recovers_angles_through_horizontal_and_vertical(self):
        size, pivot, radius = 520, 260, 210
        for angle in (0.0, 37.0, 89.0, 143.0):
            with self.subTest(angle=angle):
                mask = np.zeros((size, size), dtype=np.uint8)
                theta = math.radians(angle)
                points = [
                    (round(pivot + sign * radius * math.cos(theta)),
                     round(pivot - sign * radius * math.sin(theta)))
                    for sign in (-1, 1)
                ]
                cv2.line(mask, *points, 255, 7)
                fit = fit_mask_angle(mask, (pivot, pivot), radius, angle)
                self.assertIsNotNone(fit)
                difference = (fit["angleDegrees"] - angle + 90) % 180 - 90
                self.assertLess(abs(difference), 0.3)
                self.assertTrue(math.isfinite(fit["standardUncertaintyDegrees"]))

    def test_absent_broad_and_one_sided_masks_are_unavailable(self):
        mask = np.zeros((520, 520), dtype=np.uint8)
        self.assertIsNone(fit_mask_angle(mask, (260, 260), 210, 0))
        cv2.line(mask, (260, 260), (470, 260), 255, 7)
        self.assertIsNone(fit_mask_angle(mask, (260, 260), 210, 0))
        self.assertIsNone(fit_mask_angle(np.full_like(mask, 255), (260, 260), 210, 0))

    def test_displacement_uses_reference_and_preserves_missing_fits(self):
        propagated = displacement_standard_uncertainty([0.1, 0.2, np.nan])
        self.assertEqual(propagated[0], 0)
        self.assertAlmostEqual(propagated[1], math.hypot(0.1, 0.2))
        self.assertTrue(math.isnan(propagated[2]))
        self.assertTrue(np.isnan(displacement_standard_uncertainty([np.nan, 0.2])).all())


if __name__ == "__main__":
    unittest.main()
