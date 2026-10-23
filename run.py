#!/usr/bin/env python3
"""Track one calibrated video and write only time and angular displacement."""

import argparse
from pathlib import Path

from rod_tracker.io import DEFAULT_CALIBRATIONS, export_results, load_calibration, load_items, uncertainty_path
from rod_tracker.tracker_engine import track_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--key", required=True, help="Key in the saved calibration file")
    parser.add_argument("--calibrations", type=Path, default=DEFAULT_CALIBRATIONS)
    parser.add_argument("--out", type=Path, required=True, help="New CSV file to create")
    args = parser.parse_args()
    if args.out.exists() or uncertainty_path(args.out).exists():
        parser.error(f"Output already exists: {args.out} or {uncertainty_path(args.out)}")
    item = load_items(args.calibrations)[args.key]
    data, quality, _background = track_video(
        args.video.resolve(strict=True),
        load_calibration(item),
        shaft_residual_issue_deg=float(item.get("shaftResidualIssueThresholdDegrees", 1.0)),
        progress=lambda fraction, message: print(f"{fraction:5.0%} {message}", flush=True),
    )
    _csv, report = export_results(data, quality, args.out)
    print(f"Saved {len(data)} frames to {args.out}")
    print(f"Saved mask-fit standard uncertainties (1 sigma) to {report}")


if __name__ == "__main__":
    main()
