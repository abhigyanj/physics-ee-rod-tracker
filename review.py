#!/usr/bin/env python3
"""Local frame-by-frame review of raw rod tracking; no velocity or graphs."""

import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

import cv2

from rod_tracker.io import DEFAULT_CALIBRATIONS, export_results, load_calibration, load_items, uncertainty_path
from rod_tracker.tracker_engine import draw_overlay, open_portrait_video, track_video
from rod_tracker.uncertainty import DISPLACEMENT_STD, FIT_DIFFERENCE


class Review:
    def __init__(self, videos, calibrations, output):
        self.videos = videos.resolve(strict=True)
        self.output = output.resolve()
        self.items = load_items(calibrations)
        self.lock = threading.RLock()
        self.decode_lock = threading.Lock()
        self.key = None
        self.data = None
        self.quality = None
        self.video = None
        self.csv_path = None
        self.status = {"running": False, "progress": 0, "message": "Choose a clip."}
        self.frames = None

    def source(self, key):
        path = (self.videos / self.items[key]["video"]).resolve(strict=True)
        if not path.is_relative_to(self.videos) or not path.is_file():
            raise ValueError("The video must be inside the selected video folder.")
        return path

    def state(self):
        with self.lock:
            return {**self.status, "key": self.key, "rows": len(self.data) if self.data is not None else 0,
                    "csv": self.csv_path.name if self.csv_path else None,
                    "uncertainty": uncertainty_path(self.csv_path).name if self.csv_path else None}

    def start(self, key):
        with self.lock:
            if self.status["running"]:
                raise ValueError("Wait for the current clip to finish.")
            source = self.source(key)
            output = self.output / (key.removesuffix("_preview.jpg") + ".csv")
            if output.exists() or uncertainty_path(output).exists():
                raise ValueError(f"Output already exists: {output}. Choose a fresh --out folder to rerun.")
            self.key = key
            self.data = self.quality = self.csv_path = None
            self.status = {"running": True, "progress": 0, "message": "Starting tracking."}
        threading.Thread(target=self.work, args=(key, source, output), daemon=True).start()

    def work(self, key, source, output):
        def progress(fraction, message):
            with self.lock:
                self.status.update(progress=fraction, message=message)
        try:
            item = self.items[key]
            data, quality, _background = track_video(
                source, load_calibration(item),
                shaft_residual_issue_deg=float(item.get("shaftResidualIssueThresholdDegrees", 1.0)),
                progress=progress,
            )
            export_results(data, quality, output)
            with self.decode_lock:
                if self.frames is not None:
                    self.frames.release()
                self.frames = open_portrait_video(source)
                self.next_frame = 0
            with self.lock:
                self.video, self.data, self.quality, self.csv_path = source, data, quality, output
                self.status = {"running": False, "progress": 1, "message": "CSV and mask-fit report saved. Review the overlay."}
        except Exception as error:
            with self.lock:
                self.status = {"running": False, "progress": 0, "message": str(error)}

    def row(self, index):
        with self.lock:
            if self.data is None:
                raise ValueError("Track a clip first.")
            index = max(0, min(int(index), len(self.data) - 1))
            return self.data.iloc[index].copy(), dict(self.quality), self.items[self.key]

    def frame(self, index, overlay):
        row, quality, item = self.row(index)
        target = int(row["Frame"])
        with self.decode_lock:
            # Decode sequentially from a fresh capture on backward seeks.
            # Timestamp seeking in these iPhone files can skip frames.
            if target < self.next_frame:
                self.frames.release()
                self.frames = open_portrait_video(self.video)
                self.next_frame = 0
            while self.next_frame <= target:
                ok, image = self.frames.read()
                if not ok:
                    raise ValueError("Could not decode the requested source frame.")
                self.current_frame = image
                self.next_frame += 1
            image = self.current_frame.copy()
        if overlay:
            calibration = replace(load_calibration(item), pivot_x=quality["pivot"][0], pivot_y=quality["pivot"][1])
            image = draw_overlay(image, calibration, float(row["Tracked Orientation (degrees)"]),
                                 float(row["Tracking Confidence"]), not bool(row["Tracking Valid"]))
        height, width = image.shape[:2]
        image = cv2.resize(image, (round(width * min(1, 900 / height)), round(height * min(1, 900 / height))))
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise ValueError("Could not encode the review frame.")
        return encoded.tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def send(self, data, content_type, status=200):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self.send((Path(__file__).parent / "rod_tracker/web/index.html").read_bytes(), "text/html; charset=utf-8")
            elif parsed.path == "/api/clips":
                self.send({"clips": [{"key": key, "label": f"Trial {item['trialNumber']} — {item['frequency']} Hz"}
                                     for key, item in self.review.items.items()
                                     if (self.review.videos / item["video"]).is_file()]}, "application/json")
            elif parsed.path == "/api/state":
                self.send(self.review.state(), "application/json")
            elif parsed.path == "/api/row":
                row, _quality, _item = self.review.row(int(query.get("row", [0])[0]))
                fit_std = float(row[DISPLACEMENT_STD])
                fit_difference = float(row[FIT_DIFFERENCE])
                self.send({"time": float(row["Time (s)"]), "angle": float(row["Angular Displacement (degrees)"]),
                           "fitStdDeg": fit_std if math.isfinite(fit_std) else None,
                           "fitDifferenceDeg": fit_difference if math.isfinite(fit_difference) else None,
                           "flagged": not bool(row["Tracking Valid"])}, "application/json")
            elif parsed.path == "/api/frame":
                self.send(self.review.frame(int(query.get("row", [0])[0]), query.get("overlay", ["1"])[0] == "1"), "image/jpeg")
            elif parsed.path == "/data.csv" and self.review.csv_path:
                self.send(self.review.csv_path.read_bytes(), "text/csv; charset=utf-8")
            elif parsed.path == "/uncertainty.json" and self.review.csv_path:
                self.send(uncertainty_path(self.review.csv_path).read_bytes(), "application/json")
            else:
                self.send({"error": "Not found"}, "application/json", 404)
        except Exception as error:
            self.send({"error": str(error)}, "application/json", 400)

    def do_POST(self):
        try:
            if urlparse(self.path).path != "/api/track":
                self.send({"error": "Not found"}, "application/json", 404)
                return
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 8192)))
            self.review.start(body["key"])
            self.send({"ok": True}, "application/json", 202)
        except Exception as error:
            self.send({"error": str(error)}, "application/json", 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Folder containing the trial segment folders")
    parser.add_argument("--calibrations", type=Path, default=DEFAULT_CALIBRATIONS)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--port", type=int, default=8878)
    args = parser.parse_args()
    Handler.review = Review(args.videos, args.calibrations, args.out)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Open http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
