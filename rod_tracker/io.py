"""Read saved calibration points and export the raw measurements."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .tracker_engine import Calibration
from .uncertainty import DISPLACEMENT_STD, FIT_ANGLE, FIT_DIFFERENCE, FIT_STD


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATIONS = ROOT / "calibrations.json"
EXPORT_COLUMNS = ["Time (s)", "Angular Displacement (degrees)"]


def load_items(path=DEFAULT_CALIBRATIONS):
    return json.loads(Path(path).read_text())["items"]


def load_calibration(item):
    return Calibration.from_mapping(item["calibration"])


def raw_measurements(data):
    raw = data.loc[:, EXPORT_COLUMNS].copy()
    values = raw.to_numpy(dtype=float)
    if len(raw) < 2 or not np.isfinite(values).all():
        raise ValueError("Tracking did not produce a finite time/angle series.")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Frame times must be strictly increasing.")
    return raw


def export_csv(data, output):
    output = Path(output)
    raw = raw_measurements(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental overwrite of an earlier run.
    with output.open("x", newline="") as stream:
        raw.to_csv(stream, index=False)
    return output


def uncertainty_path(output):
    output = Path(output)
    return output.with_name(output.stem + "_uncertainty.json")


def export_results(data, quality, output):
    """Keep the raw CSV unchanged; write calculated fit diagnostics separately."""
    output = Path(output)
    report = uncertainty_path(output)
    if output.exists() or report.exists():
        raise FileExistsError(f"Output already exists: {output} or {report}")

    def finite(value):
        return float(value) if np.isfinite(value) else None

    rows = []
    for _, row in data.iterrows():
        rows.append({
            "timeSeconds": float(row["Time (s)"]),
            "orientationDegrees": finite(row[FIT_ANGLE]),
            "orientationStandardUncertaintyDegrees": finite(row[FIT_STD]),
            "displacementStandardUncertaintyDegrees": finite(row[DISPLACEMENT_STD]),
            "fitToTrackedDifferenceDegrees": finite(row[FIT_DIFFERENCE]),
            "residualStdPixels": finite(row["Mask Fit Residual Std (pixels)"]),
            "sections": int(row["Mask Fit Sections"]),
            "pixels": int(row["Mask Fit Pixels"]),
            "trackingValid": bool(row["Tracking Valid"]),
        })
    payload = json.dumps({
        **quality["maskFitUncertainty"],
        "scope": "Fit precision only; systematic tracking errors are not included.",
        "assumptions": "Straight shaft, independent equal-variance section errors; independent frame errors for relative displacement.",
        "frames": rows,
    }, indent=2, allow_nan=False) + "\n"
    export_csv(data, output)
    created_report = False
    try:
        with report.open("x") as stream:
            created_report = True
            stream.write(payload)
    except Exception:
        output.unlink()
        if created_report:
            report.unlink()
        raise
    return output, report
