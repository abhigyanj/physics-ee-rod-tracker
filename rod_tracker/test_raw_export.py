import tempfile
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from rod_tracker.io import EXPORT_COLUMNS, export_csv, export_results, raw_measurements, uncertainty_path
from rod_tracker.uncertainty import DISPLACEMENT_STD, FIT_ANGLE, FIT_DIFFERENCE, FIT_STD


class RawExportTests(unittest.TestCase):
    def test_export_preserves_turns_and_irregular_motion(self):
        data = pd.DataFrame({"Time (s)": [0, 0.033, 0.067, 0.1],
                             "Angular Displacement (degrees)": [0, 179.9, 360.1, 721.2],
                             "Tracking Confidence": [0.9, 0.7, 0.8, 0.9]})
        with tempfile.TemporaryDirectory() as folder:
            path = export_csv(data, Path(folder) / "data.csv")
            saved = pd.read_csv(path)
            self.assertEqual(list(saved.columns), EXPORT_COLUMNS)
            np.testing.assert_array_equal(saved.to_numpy(), data[EXPORT_COLUMNS].to_numpy())
            self.assertEqual([p.name for p in Path(folder).iterdir()], ["data.csv"])
            with self.assertRaises(FileExistsError):
                export_csv(data, path)

    def test_rejects_duplicate_frame_times(self):
        data = pd.DataFrame({"Time (s)": [0, 0], "Angular Displacement (degrees)": [0, 1]})
        with self.assertRaises(ValueError):
            raw_measurements(data)

    def test_fit_report_preserves_raw_csv_and_encodes_missing_fits_as_null(self):
        data = pd.DataFrame({
            "Time (s)": [0, 0.033], "Angular Displacement (degrees)": [0, 7.4],
            FIT_ANGLE: [30.0, np.nan], FIT_STD: [0.034, np.nan],
            DISPLACEMENT_STD: [0, np.nan], FIT_DIFFERENCE: [0.1, np.nan],
            "Mask Fit Residual Std (pixels)": [0.2, np.nan],
            "Mask Fit Sections": [24, 0], "Mask Fit Pixels": [2100, 0],
            "Tracking Valid": [True, False],
        })
        quality = {"maskFitUncertainty": {"validFrames": 1, "unavailableFrames": 1}}
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "data.csv"
            csv, report = export_results(data, quality, output)
            np.testing.assert_array_equal(pd.read_csv(csv).to_numpy(), data[EXPORT_COLUMNS].to_numpy())
            content = json.loads(report.read_text())
            self.assertAlmostEqual(content["frames"][0]["orientationStandardUncertaintyDegrees"], 0.034)
            self.assertIsNone(content["frames"][1]["orientationStandardUncertaintyDegrees"])
            self.assertNotIn("NaN", report.read_text())
            with self.assertRaises(FileExistsError):
                export_results(data, quality, output)

    def test_existing_report_prevents_creating_a_partial_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "data.csv"
            report = uncertainty_path(output)
            report.write_text("original")
            with self.assertRaises(FileExistsError):
                export_results(pd.DataFrame(), {}, output)
            self.assertFalse(output.exists())
            self.assertEqual(report.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
