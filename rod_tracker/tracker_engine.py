#!/usr/bin/env python3
"""Rigid-assembly tracker for the EE rotating rod videos.

This tracker deliberately models three parts of the same object:
the thin shaft through the pivot and the two broad bottle/end regions.
It never writes into the existing tracking result tree unless a caller
explicitly exports a result elsewhere.
"""

from __future__ import annotations

import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from rod_tracker.uncertainty import (
    DISPLACEMENT_STD, FIT_ANGLE, FIT_DIFFERENCE, FIT_STD,
    displacement_standard_uncertainty, fit_mask_angle,
)

ANGLE_STEP_DEG = 0.5
BACKGROUND_SAMPLES = 51
MAX_STEP_DEG = 24.0
# The calibrated shaft observation is already sharp. A soft cost on angular
# speed biased accelerating clips away from the visible shaft (by more than
# 10 degrees on Trial Three 240 Hz). Keep only the hard maximum-step guard.
TRANSITION_PENALTY = 0.0
ANCHOR_PENALTY = 0.10
CORRECTION_PENALTY = 0.32
ISSUE_CONFIDENCE = 0.52
SHAFT_RESIDUAL_ISSUE_DEG = 1.0
PIVOT_REFINE_SAMPLES = 31
PIVOT_REFINE_MAX_PX = 6.0


@dataclass(frozen=True)
class Calibration:
    pivot_x: float
    pivot_y: float
    end1_x: float
    end1_y: float
    end2_x: float
    end2_y: float
    radius: float
    start_frame: int = 0
    lower_bound: tuple[int, int, int] | None = None
    upper_bound: tuple[int, int, int] | None = None

    @property
    def initial_angle(self) -> float:
        return math.degrees(
            math.atan2(self.pivot_y - self.end1_y, self.end1_x - self.pivot_x)
        ) % 180.0

    @classmethod
    def from_mapping(cls, payload: dict) -> "Calibration":
        def hsv_triplet(name: str) -> tuple[int, int, int] | None:
            values = payload.get(name)
            if not isinstance(values, (list, tuple)) or len(values) != 3:
                return None
            return tuple(int(np.clip(value, 0, 255)) for value in values)

        return cls(
            pivot_x=float(payload["pivot_x"]),
            pivot_y=float(payload["pivot_y"]),
            end1_x=float(payload["end1_x"]),
            end1_y=float(payload["end1_y"]),
            end2_x=float(payload["end2_x"]),
            end2_y=float(payload["end2_y"]),
            radius=float(payload["radius"]),
            start_frame=int(payload.get("start_frame", 0)),
            lower_bound=hsv_triplet("lower_bound"),
            upper_bound=hsv_triplet("upper_bound"),
        )


def shortest_axis_delta(angle: float | np.ndarray, reference: float | np.ndarray):
    return (np.asarray(angle) - np.asarray(reference) + 90.0) % 180.0 - 90.0


def open_portrait_video(video_path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    # OpenCV builds disagree on the default. Force one coordinate system.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def read_frame_at_source_time(
    cap: cv2.VideoCapture,
    source_time: float,
    fps: float,
) -> tuple[bool, np.ndarray | None]:
    """Read the decoded frame whose timestamp is closest to ``source_time``.

    Exact seeks to the last timestamp in these MOV files can land just beyond
    EOF. A one-frame-earlier retry asks the decoder for that same final frame
    without sequential reads after a seek (which can corrupt this HEVC stream).
    """
    target = max(0.0, float(source_time))
    if target <= 1e-9:
        return cap.read()

    frame_interval = 1.0 / max(float(fps), 1.0)
    for frames_early in (0, 1, 2, 3):
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            max(0.0, target - frames_early * frame_interval) * 1000.0,
        )
        ok, frame = cap.read()
        if ok:
            return True, frame
    return False, None


def video_metadata(video_path: str | Path) -> dict:
    cap = open_portrait_video(video_path)
    metadata = {
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "frames": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        "orientationMeta": float(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
        "orientationAuto": True,
    }
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame: {video_path}")
    metadata["decodedWidth"] = int(frame.shape[1])
    metadata["decodedHeight"] = int(frame.shape[0])
    return metadata


def roi_bounds(calibration: Calibration, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape[:2]
    pad = calibration.radius + max(40.0, calibration.radius * 0.12)
    return (
        max(0, int(math.floor(calibration.pivot_x - pad))),
        min(frame_w, int(math.ceil(calibration.pivot_x + pad)) + 1),
        max(0, int(math.floor(calibration.pivot_y - pad))),
        min(frame_h, int(math.ceil(calibration.pivot_y + pad)) + 1),
    )


def build_low_percentile_background(
    video_path: str | Path,
    calibration: Calibration,
    sample_count: int = BACKGROUND_SAMPLES,
) -> tuple[np.ndarray, tuple[int, int, int, int], dict]:
    cap = open_portrait_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read first frame: {video_path}")
    bounds = roi_bounds(calibration, first.shape)
    x_min, x_max, y_min, y_max = bounds
    indexes = np.linspace(0, max(0, total - 1), sample_count).astype(int)
    targets = set(int(index) for index in indexes)
    crops = []
    frame_index = 0
    frame = first
    last_target = max(targets) if targets else -1
    while frame_index <= last_target:
        if frame_index in targets:
            crops.append(frame[y_min:y_max, x_min:x_max])
        if frame_index >= last_target:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    if not crops:
        raise RuntimeError(f"Could not sample video: {video_path}")
    # The rod and bottles are brighter than the carpet. A low percentile is
    # less likely than a median to bake a long settled pose into the background.
    stack = np.stack(crops).astype(np.uint8)
    background = np.percentile(stack, 20, axis=0).astype(np.uint8)
    return background, bounds, {"fps": fps, "frames": total, "samples": len(crops)}


def build_sampling_grid(calibration: Calibration, bounds: tuple[int, int, int, int]) -> dict:
    x_min, x_max, y_min, y_max = bounds
    roi_w = x_max - x_min
    roi_h = y_max - y_min
    pivot_x = calibration.pivot_x - x_min
    pivot_y = calibration.pivot_y - y_min
    radius = calibration.radius
    angles = np.arange(0.0, 180.0, ANGLE_STEP_DEG, dtype=np.float32)

    shaft_r = np.arange(max(35.0, radius * 0.12), radius * 0.76, 3.0, dtype=np.float32)
    shaft_r = np.concatenate([-shaft_r[::-1], shaft_r])
    shaft_weights = np.clip((np.abs(shaft_r) / radius) ** 0.8, 0.22, 1.0).astype(np.float32)
    positive = shaft_r > 0
    negative = shaft_r < 0

    endpoint_r = np.linspace(radius * 0.72, radius * 1.03, 22, dtype=np.float32)
    endpoint_r = np.concatenate([-endpoint_r[::-1], endpoint_r])
    endpoint_positive = endpoint_r > 0
    endpoint_negative = endpoint_r < 0

    def coordinates(radii: np.ndarray, perpendicular_offset: float = 0.0):
        theta = np.radians(angles)[:, None]
        x = pivot_x + radii[None, :] * np.cos(theta) + perpendicular_offset * np.sin(theta)
        y = pivot_y - radii[None, :] * np.sin(theta) + perpendicular_offset * np.cos(theta)
        xi = np.rint(x).astype(np.int32)
        yi = np.rint(y).astype(np.int32)
        valid = (xi >= 0) & (xi < roi_w) & (yi >= 0) & (yi < roi_h)
        return np.clip(xi, 0, roi_w - 1), np.clip(yi, 0, roi_h - 1), valid

    endpoint_offsets = (-38.0, -24.0, -12.0, 0.0, 12.0, 24.0, 38.0)
    return {
        "angles": angles,
        "pivot": (pivot_x, pivot_y),
        "radius": radius,
        "shaft": coordinates(shaft_r),
        "shaft_left": coordinates(shaft_r, -9.0),
        "shaft_right": coordinates(shaft_r, 9.0),
        "shaft_weights": shaft_weights,
        "shaft_positive": positive,
        "shaft_negative": negative,
        "endpoints": [coordinates(endpoint_r, offset) for offset in endpoint_offsets],
        "endpoint_positive": endpoint_positive,
        "endpoint_negative": endpoint_negative,
        "hsv_bounds": (
            np.asarray(calibration.lower_bound, dtype=np.uint8),
            np.asarray(calibration.upper_bound, dtype=np.uint8),
        ) if calibration.lower_bound is not None and calibration.upper_bound is not None else None,
    }


def normalized_frame_maps(frame_roi: np.ndarray, background_roi: np.ndarray) -> dict:
    frame_f = frame_roi.astype(np.float32)
    background_f = background_roi.astype(np.float32)
    diff = np.max(np.abs(frame_f - background_f), axis=2)
    motion = np.clip((diff - 8.0) / 58.0, 0.0, 1.0)

    b, g, r = cv2.split(frame_f)
    brightness = (b + g + r) / 3.0
    warm = np.clip((r - b + 16.0) / 62.0, 0.0, 1.0)
    pale = np.clip((brightness - 104.0) / 112.0, 0.0, 1.0)
    appearance = np.maximum(0.72 * warm + 0.28 * pale, pale * 0.74)

    foreground = np.clip(0.72 * motion + 0.28 * appearance, 0.0, 1.0)
    binary = ((motion > 0.20) & ((appearance > 0.22) | (pale > 0.36))).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    ridge = np.clip(distance / 7.0, 0.0, 1.0)
    endpoint = cv2.GaussianBlur(np.clip(0.82 * motion + 0.18 * pale, 0.0, 1.0), (0, 0), 5.0)
    return {
        "motion": motion.astype(np.float32),
        "appearance": appearance.astype(np.float32),
        "foreground": foreground.astype(np.float32),
        "ridge": ridge.astype(np.float32),
        "endpoint": endpoint.astype(np.float32),
    }


def weighted_rows(values: np.ndarray, weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    effective = valid * weights[None, :]
    return (values * effective).sum(axis=1) / (effective.sum(axis=1) + 1e-6)


def sample_map(image: np.ndarray, coords) -> np.ndarray:
    x, y, valid = coords
    values = image[y, x].astype(np.float32)
    values[~valid] = 0.0
    return values


def calibrated_shaft_mask(frame_roi: np.ndarray, grid: dict) -> np.ndarray | None:
    bounds = grid.get("hsv_bounds")
    if bounds is None:
        return None
    lower, upper = bounds
    hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )


def calibrated_shaft_scores(frame_roi: np.ndarray, grid: dict) -> tuple[np.ndarray | None, dict]:
    """Score the thin colored shaft using the saved per-clip HSV calibration.

    The broad end pieces are useful for confirming that an assembly exists,
    but rolling shutter and their offset geometry make their apparent axis
    diverge from the shaft during fast rotation. Sampling only the central
    12%-76% of each arm avoids that systematic bias.
    """
    mask = calibrated_shaft_mask(frame_roi, grid)
    if mask is None:
        return None, {"calibratedPixels": 0}

    binary = (mask > 0).astype(np.uint8)
    distance = np.clip(cv2.distanceTransform(binary, cv2.DIST_L2, 5) / 5.0, 0.0, 1.0)
    weights = grid["shaft_weights"]
    valid = grid["shaft"][2]
    center = sample_map(distance, grid["shaft"])
    positive = grid["shaft_positive"]
    negative = grid["shaft_negative"]

    line = weighted_rows(center, weights, valid)
    positive_score = weighted_rows(center[:, positive], weights[positive], valid[:, positive])
    negative_score = weighted_rows(center[:, negative], weights[negative], valid[:, negative])
    arm_balance = np.minimum(positive_score, negative_score) - 0.40 * np.abs(positive_score - negative_score)

    binary_f = binary.astype(np.float32)
    side_mean = 0.5 * (
        weighted_rows(sample_map(binary_f, grid["shaft_left"]), weights, grid["shaft_left"][2])
        + weighted_rows(sample_map(binary_f, grid["shaft_right"]), weights, grid["shaft_right"][2])
    )
    score = 1.30 * line + arm_balance - 0.30 * side_mean
    # A separate mask fit diagnoses precision without changing tracking scores.
    mask_fit = None
    if float(np.max(score)) >= 0.25:
        mask_fit = fit_mask_angle(
            mask, grid["pivot"], grid["radius"],
            float(grid["angles"][int(np.argmax(score))]),
        )
    return score.astype(np.float32), {
        "maskFit": mask_fit,
        "calibratedPixels": int(binary.sum()),
        "calibratedLine": line.astype(np.float32),
        "calibratedArmBalance": arm_balance.astype(np.float32),
    }


def refine_calibration_pivot(
    video_path: str | Path,
    calibration: Calibration,
    sample_count: int = PIVOT_REFINE_SAMPLES,
) -> tuple[Calibration, dict]:
    """Refine the clicked pivot from robust shaft-line offsets across a clip."""
    if calibration.lower_bound is None or calibration.upper_bound is None:
        return calibration, {"applied": False, "reason": "no_hsv_bounds"}

    cap = open_portrait_video(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        cap.release()
        return calibration, {"applied": False, "reason": "no_frame"}
    bounds = roi_bounds(calibration, first.shape)
    grid = build_sampling_grid(calibration, bounds)
    x_min, x_max, y_min, y_max = bounds
    local_pivot_x = calibration.pivot_x - x_min
    local_pivot_y = calibration.pivot_y - y_min
    indexes = np.linspace(0, max(0, total - 1), max(3, int(sample_count))).round().astype(int)
    targets = set(int(index) for index in np.unique(indexes))
    line_rows = []

    frame_index = 0
    frame = first
    last_target = max(targets) if targets else -1
    while frame_index <= last_target:
        if frame_index in targets:
            roi = frame[y_min:y_max, x_min:x_max]
            scores, _metrics = calibrated_shaft_scores(roi, grid)
            mask = calibrated_shaft_mask(roi, grid)
            if scores is not None and mask is not None:
                state = int(np.argmax(scores))
                angle = float(refine_path(scores[None, :], np.asarray([state], dtype=np.int32))[0])
                y_values, x_values = np.nonzero(mask > 0)
                dx = x_values.astype(np.float64) - local_pivot_x
                dy = local_pivot_y - y_values.astype(np.float64)
                radii = np.hypot(dx, dy)
                theta = math.radians(angle)
                along = dx * math.cos(theta) + dy * math.sin(theta)
                perpendicular = -dx * math.sin(theta) + dy * math.cos(theta)
                keep = (
                    (radii >= 0.12 * calibration.radius)
                    & (radii <= 0.70 * calibration.radius)
                    & (np.abs(perpendicular) < 14.0)
                )
                positive_count = int(np.sum(keep & (along > 0)))
                negative_count = int(np.sum(keep & (along < 0)))
                if positive_count >= 60 and negative_count >= 60:
                    offset = float(np.median(perpendicular[keep]))
                    line_rows.append(((-math.sin(theta), math.cos(theta)), offset))
        if frame_index >= last_target:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
    cap.release()

    if len(line_rows) < 8:
        return calibration, {"applied": False, "reason": "too_few_lines", "lines": len(line_rows)}
    normals = np.asarray([row[0] for row in line_rows], dtype=np.float64)
    offsets = np.asarray([row[1] for row in line_rows], dtype=np.float64)
    if np.linalg.matrix_rank(normals) < 2 or float(np.linalg.cond(normals)) > 12.0:
        return calibration, {"applied": False, "reason": "insufficient_angle_span", "lines": len(line_rows)}

    correction = np.linalg.lstsq(normals, offsets, rcond=None)[0]
    residuals = offsets - normals @ correction
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    inliers = np.abs(residuals - center) <= max(1.0, 3.5 * mad)
    if int(np.sum(inliers)) >= 8:
        correction = np.linalg.lstsq(normals[inliers], offsets[inliers], rcond=None)[0]
        residuals = offsets[inliers] - normals[inliers] @ correction

    correction_norm = float(np.linalg.norm(correction))
    if correction_norm > PIVOT_REFINE_MAX_PX:
        correction *= PIVOT_REFINE_MAX_PX / correction_norm
        correction_norm = PIVOT_REFINE_MAX_PX
    refined = replace(
        calibration,
        pivot_x=calibration.pivot_x + float(correction[0]),
        pivot_y=calibration.pivot_y - float(correction[1]),
    )
    return refined, {
        "applied": True,
        "lines": int(len(residuals)),
        "dx": round(float(correction[0]), 4),
        "dy": round(float(-correction[1]), 4),
        "magnitude": round(correction_norm, 4),
        "p95ResidualPx": round(float(np.percentile(np.abs(residuals), 95)), 4),
    }


def normalize_observation_scores(raw: np.ndarray) -> np.ndarray:
    """Normalize a frame without flattening its strongest angular peak.

    The former upper clip at 2.5 turned every sufficiently strong neighboring
    angle into an exact tie. On fast clips that erased the distinction between
    the visible shaft and an angle more than 10 degrees away. A conservative
    scale floor handles genuinely flat frames without discarding peak shape.
    """
    low = float(np.percentile(raw, 12))
    high = float(np.percentile(raw, 97))
    scale = max(high - low, 0.05)
    normalized = (raw - low) / scale
    return np.maximum(normalized, -0.5).astype(np.float32)


def observation_scores(frame_roi: np.ndarray, background_roi: np.ndarray, grid: dict) -> tuple[np.ndarray, dict]:
    maps = normalized_frame_maps(frame_roi, background_roi)
    weights = grid["shaft_weights"]
    shaft_valid = grid["shaft"][2]
    center_foreground = sample_map(maps["foreground"], grid["shaft"])
    center_ridge = sample_map(maps["ridge"], grid["shaft"])
    center_appearance = sample_map(maps["appearance"], grid["shaft"])
    side_left = sample_map(maps["foreground"], grid["shaft_left"])
    side_right = sample_map(maps["foreground"], grid["shaft_right"])

    shaft = weighted_rows(0.48 * center_foreground + 0.38 * center_ridge + 0.14 * center_appearance, weights, shaft_valid)
    pos = grid["shaft_positive"]
    neg = grid["shaft_negative"]
    pos_score = weighted_rows(center_foreground[:, pos], weights[pos], shaft_valid[:, pos])
    neg_score = weighted_rows(center_foreground[:, neg], weights[neg], shaft_valid[:, neg])
    arm_balance = np.minimum(pos_score, neg_score) - 0.35 * np.abs(pos_score - neg_score)

    side_mean = 0.5 * (
        weighted_rows(side_left, weights, grid["shaft_left"][2])
        + weighted_rows(side_right, weights, grid["shaft_right"][2])
    )
    thinness = np.clip(shaft - 0.20 * side_mean, -0.4, 1.0)

    endpoint_samples = np.stack([
        sample_map(maps["endpoint"], coords) for coords in grid["endpoints"]
    ])
    endpoint_mean = endpoint_samples.mean(axis=0)
    endpoint_valid = grid["endpoints"][0][2]
    epos = grid["endpoint_positive"]
    eneg = grid["endpoint_negative"]
    endpoint_pos = weighted_rows(endpoint_mean[:, epos], np.ones(int(epos.sum()), np.float32), endpoint_valid[:, epos])
    endpoint_neg = weighted_rows(endpoint_mean[:, eneg], np.ones(int(eneg.sum()), np.float32), endpoint_valid[:, eneg])
    endpoint_pair = np.minimum(endpoint_pos, endpoint_neg) - 0.25 * np.abs(endpoint_pos - endpoint_neg)

    rigid_raw = 1.25 * shaft + 0.95 * arm_balance + 0.65 * thinness + 0.85 * endpoint_pair
    calibrated_raw, calibrated_metrics = calibrated_shaft_scores(frame_roi, grid)
    if calibrated_raw is not None and float(np.max(calibrated_raw)) >= 0.25:
        # The saved color calibration directly targets the visible shaft.
        # A small rigid-model term only breaks genuinely ambiguous ties.
        raw = calibrated_raw + 0.08 * rigid_raw
        observation_model = "calibrated_shaft"
    else:
        raw = rigid_raw
        calibrated_raw = np.zeros_like(raw, dtype=np.float32)
        observation_model = "rigid_fallback"
    normalized = normalize_observation_scores(raw)
    return normalized, {
        "shaft": shaft.astype(np.float32),
        "armBalance": arm_balance.astype(np.float32),
        "endpointPair": endpoint_pair.astype(np.float32),
        "rawScore": raw.astype(np.float32),
        "calibratedShaftScore": calibrated_raw.astype(np.float32),
        "observationModel": observation_model,
        "maxObservation": float(np.max(normalized)),
        **calibrated_metrics,
    }


def viterbi_path(scores: np.ndarray, initial_angle: float, end_angle: float | None = None) -> np.ndarray:
    frame_count, state_count = scores.shape
    angles = np.arange(state_count, dtype=np.float32) * ANGLE_STEP_DEG
    max_bins = int(round(MAX_STEP_DEG / ANGLE_STEP_DEG))
    offsets = np.arange(-max_bins, max_bins + 1, dtype=np.int32)
    transition = TRANSITION_PENALTY * (offsets.astype(np.float32) * ANGLE_STEP_DEG) ** 2
    start_delta = np.abs(shortest_axis_delta(angles, initial_angle))
    previous = scores[0].astype(np.float64) - ANCHOR_PENALTY * start_delta**2
    back = np.zeros((frame_count, state_count), dtype=np.uint16)
    state_ids = np.arange(state_count, dtype=np.int32)

    for frame_index in range(1, frame_count):
        current = np.empty(state_count, dtype=np.float64)
        for state in range(state_count):
            prior = (state + offsets) % state_count
            values = previous[prior] - transition
            best = int(np.argmax(values))
            back[frame_index, state] = int(prior[best])
            current[state] = float(scores[frame_index, state]) + float(values[best])
        previous = current

    if end_angle is None:
        end_state = int(np.argmax(previous))
    else:
        end_delta = np.abs(shortest_axis_delta(angles, end_angle))
        end_state = int(np.argmax(previous - ANCHOR_PENALTY * end_delta**2))
    path = np.empty(frame_count, dtype=np.int32)
    path[-1] = end_state
    for frame_index in range(frame_count - 1, 0, -1):
        path[frame_index - 1] = int(back[frame_index, path[frame_index]])
    return path


def refine_path(scores: np.ndarray, path: np.ndarray) -> np.ndarray:
    refined = np.empty(len(path), dtype=np.float64)
    states = scores.shape[1]
    for index, state in enumerate(path):
        left = float(scores[index, (state - 1) % states])
        center = float(scores[index, state])
        right = float(scores[index, (state + 1) % states])
        denominator = left - 2.0 * center + right
        offset = 0.0 if abs(denominator) < 1e-9 else 0.5 * (left - right) / denominator
        refined[index] = ((state + float(np.clip(offset, -1.0, 1.0))) * ANGLE_STEP_DEG) % 180.0
    return refined


def unwrap_axis(mod_angles: np.ndarray, initial_angle: float) -> np.ndarray:
    result = np.empty(len(mod_angles), dtype=np.float64)
    result[0] = initial_angle + float(shortest_axis_delta(mod_angles[0], initial_angle))
    for index in range(1, len(mod_angles)):
        result[index] = result[index - 1] + float(shortest_axis_delta(mod_angles[index], result[index - 1] % 180.0))
    return result


def circular_peak_margin(frame_scores: np.ndarray, state: int, exclusion_degrees: float = 5.0) -> float:
    exclusion = int(round(exclusion_degrees / ANGLE_STEP_DEG))
    mask = np.ones(len(frame_scores), dtype=bool)
    for offset in range(-exclusion, exclusion + 1):
        mask[(state + offset) % len(frame_scores)] = False
    alternate = float(np.max(frame_scores[mask])) if np.any(mask) else float(frame_scores[state])
    return float(frame_scores[state]) - alternate


def track_video(
    video_path: str | Path,
    calibration: Calibration,
    anchors: dict[int, float] | None = None,
    shaft_residual_issue_deg: float = SHAFT_RESIDUAL_ISSUE_DEG,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    progress = progress or (lambda _fraction, _message: None)
    input_pivot = (calibration.pivot_x, calibration.pivot_y)
    progress(0.01, "Refining pivot from calibrated shaft lines")
    calibration, pivot_refinement = refine_calibration_pivot(video_path, calibration)
    background, bounds, meta = build_low_percentile_background(video_path, calibration)
    grid = build_sampling_grid(calibration, bounds)
    x_min, x_max, y_min, y_max = bounds
    cap = open_portrait_video(video_path)
    # Seeking to frame zero is not a no-op for these iPhone MOV files: OpenCV
    # starts two to three decoded frames later. Leave a fresh capture untouched
    # at zero, and record the decoder timestamp for exact later playback.
    if calibration.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, calibration.start_frame)
    scores = []
    features = []
    frame_indices = []
    source_times = []
    total = max(1, int(meta["frames"]) - calibration.start_frame)
    frame_index = calibration.start_frame
    pending_scores = deque()

    def collect_scored_frame() -> None:
        future, decoded_frame, decoded_time = pending_scores.popleft()
        score, feature = future.result()
        scores.append(score)
        features.append(feature)
        frame_indices.append(decoded_frame)
        source_times.append(decoded_time)
        if len(scores) == 1 or len(scores) % 15 == 0:
            progress(min(0.82, 0.82 * len(scores) / total), f"Scoring frame {len(scores)} / {total}")

    # One decoder feeds a bounded queue of independent frame-scoring jobs.
    # This uses the available CPU without opening concurrent HEVC streams.
    with ThreadPoolExecutor(max_workers=8) as score_pool:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded_frame = max(0, int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
            decoded_time = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not np.isfinite(decoded_time) or decoded_time < 0:
                decoded_time = (frame_index - calibration.start_frame) / float(meta["fps"] or 30.0)
            roi = frame[y_min:y_max, x_min:x_max].copy()
            pending_scores.append(
                (
                    score_pool.submit(observation_scores, roi, background, grid),
                    decoded_frame,
                    decoded_time,
                )
            )
            if len(pending_scores) >= 8:
                collect_scored_frame()
            frame_index += 1
        while pending_scores:
            collect_scored_frame()
    cap.release()
    if not scores:
        raise RuntimeError(f"No frames decoded: {video_path}")

    score_matrix = np.vstack(scores).astype(np.float32)
    applied_anchors = {}
    for absolute_frame, corrected_angle in (anchors or {}).items():
        row_index = int(absolute_frame) - calibration.start_frame
        if not 0 <= row_index < len(score_matrix):
            continue
        delta = shortest_axis_delta(grid["angles"], float(corrected_angle))
        score_matrix[row_index] -= (CORRECTION_PENALTY * delta * delta).astype(np.float32)
        applied_anchors[int(absolute_frame)] = float(corrected_angle) % 180.0
    progress(0.84, "Solving global angle path")
    forward_path = viterbi_path(score_matrix, calibration.initial_angle)
    forward_mod = refine_path(score_matrix, forward_path)
    forward_unwrapped = unwrap_axis(forward_mod, calibration.initial_angle)

    # Solve independently from the opposite end. The old implementation seeded
    # this pass with the forward result and anchored it back to the same start,
    # which made near-zero disagreement circular rather than diagnostic.
    reverse_scores = score_matrix[::-1]
    reverse_start = float(grid["angles"][int(np.argmax(score_matrix[-1]))])
    reverse_path = viterbi_path(reverse_scores, reverse_start)
    reverse_mod = refine_path(reverse_scores, reverse_path)[::-1]
    disagreement = np.abs(shortest_axis_delta(forward_mod, reverse_mod)).astype(np.float64)

    local_path = np.argmax(score_matrix, axis=1).astype(np.int32)
    local_mod = refine_path(score_matrix, local_path)
    local_disagreement = np.abs(shortest_axis_delta(forward_mod, local_mod)).astype(np.float64)

    margins = np.array([
        circular_peak_margin(score_matrix[index], int(state))
        for index, state in enumerate(forward_path)
    ])
    selected = score_matrix[np.arange(len(forward_path)), forward_path]
    raw_score_matrix = np.vstack([feature["rawScore"] for feature in features]).astype(np.float32)
    raw_selected = raw_score_matrix[np.arange(len(forward_path)), forward_path]
    raw_margins = np.array([
        circular_peak_margin(raw_score_matrix[index], int(state))
        for index, state in enumerate(forward_path)
    ])
    calibrated_score_matrix = np.vstack([
        feature["calibratedShaftScore"] for feature in features
    ]).astype(np.float32)
    calibrated_strong = np.max(calibrated_score_matrix, axis=1) >= 0.25
    calibrated_selected = calibrated_score_matrix[np.arange(len(forward_path)), forward_path]
    calibrated_local_path = np.argmax(calibrated_score_matrix, axis=1).astype(np.int32)
    calibrated_local_mod = refine_path(calibrated_score_matrix, calibrated_local_path)
    calibrated_residual = np.abs(
        shortest_axis_delta(forward_mod, calibrated_local_mod)
    ).astype(np.float64)
    mask_fits = [feature.get("maskFit") for feature in features]
    fit_angle = np.asarray([
        fit["angleDegrees"] if fit is not None else np.nan for fit in mask_fits
    ])
    fit_std = np.asarray([
        fit["standardUncertaintyDegrees"] if fit is not None else np.nan for fit in mask_fits
    ])
    fit_displacement_std = displacement_standard_uncertainty(fit_std)
    fit_difference = np.abs(shortest_axis_delta(forward_mod, fit_angle))

    # This is signal confidence, not an accuracy percentage. It combines
    # absolute shaft support, peak separation, and two independent consistency
    # checks. Capping at 95% avoids presenting a self-score as certainty.
    # Exposure flicker and a thin, stationary shaft can make absolute support
    # small even when the angular peak is clean. Combine absolute support with
    # scale-free peak separation, while requiring some absolute evidence so a
    # forcibly normalized noise peak cannot claim high confidence.
    strength = np.clip((raw_selected - 0.02) / 0.30, 0.0, 1.0)
    relative_separation = np.clip((margins - 0.70) / 1.30, 0.0, 1.0)
    evidence = np.sqrt(strength * (0.35 + 0.65 * relative_separation))
    confidence = 0.35 + 0.60 * evidence
    confidence *= np.exp(-0.18 * local_disagreement)
    confidence *= np.exp(-0.14 * disagreement)
    confidence *= np.exp(-0.25 * np.where(calibrated_strong, calibrated_residual, 0.0))
    confidence = np.clip(confidence, 0.0, 0.95)
    issue = (
        (confidence < ISSUE_CONFIDENCE)
        | (local_disagreement > 2.0)
        | (disagreement > 2.0)
        # A residual from a weak color mask is not an independent measurement.
        # On motion-blurred frames observation_scores deliberately uses the
        # rigid fallback; enforce shaft agreement only when the calibrated
        # color evidence itself cleared that same 0.25 strength gate.
        | (calibrated_strong & (calibrated_residual > float(shaft_residual_issue_deg)))
    )

    fps = float(meta["fps"] or 30.0)
    source_time_values = np.asarray(source_times, dtype=float)
    if (
        len(source_time_values) >= 2
        and np.all(np.diff(source_time_values) >= 0)
        and source_time_values[-1] > source_time_values[0]
    ):
        time_values = source_time_values - source_time_values[0]
    else:
        time_values = np.arange(len(frame_indices), dtype=float) / fps
        source_time_values = time_values.copy()
    data = pd.DataFrame({
        "Frame": frame_indices,
        "Time (s)": time_values,
        "Source Time (s)": np.round(source_time_values, 6),
        "Angular Displacement (degrees)": np.round(forward_unwrapped, 1),
        "Tracked Orientation (degrees)": np.round(forward_unwrapped, 1),
        "Tracking Valid": ~issue,
        "Tracking Confidence": np.round(confidence, 4),
        FIT_ANGLE: fit_angle,
        FIT_STD: fit_std,
        DISPLACEMENT_STD: fit_displacement_std,
        FIT_DIFFERENCE: fit_difference,
        "Mask Fit Sections": [fit["sections"] if fit else 0 for fit in mask_fits],
        "Mask Fit Pixels": [fit["pixels"] if fit else 0 for fit in mask_fits],
        "Mask Fit Residual Std (pixels)": [fit["residualStdPixels"] if fit else np.nan for fit in mask_fits],
        "Forward Backward Disagreement (degrees)": np.round(disagreement, 3),
        "Local Evidence Disagreement (degrees)": np.round(local_disagreement, 3),
        "Calibrated Shaft Residual (degrees)": np.round(calibrated_residual, 3),
        "Observation Score": np.round(selected, 4),
        "Raw Observation Score": np.round(raw_selected, 4),
        "Peak Margin": np.round(margins, 4),
        "Raw Peak Margin": np.round(raw_margins, 4),
        "Calibrated Shaft Score": np.round(calibrated_selected, 4),
        "Calibrated Shaft Evidence Strong": calibrated_strong,
        "Effective Pivot X": np.full(len(frame_indices), calibration.pivot_x),
        "Effective Pivot Y": np.full(len(frame_indices), calibration.pivot_y),
    })
    # Displacement is relative to the first tracked orientation; no smoothing.
    orientation = data["Tracked Orientation (degrees)"].to_numpy(dtype=float)
    data["Angular Displacement (degrees)"] = np.round(orientation - orientation[0], 1)
    issue_frames = np.flatnonzero(issue).astype(int).tolist()
    valid_strong = calibrated_strong & ~issue
    strong_residual = calibrated_residual[valid_strong]
    summary = {
        "method": "calibrated_shaft_v3",
        "videoPath": str(video_path),
        "frames": int(len(data)),
        "fps": fps,
        "orientation": "portrait_forced",
        "pivot": [round(calibration.pivot_x, 2), round(calibration.pivot_y, 2)],
        "inputPivot": [round(input_pivot[0], 2), round(input_pivot[1], 2)],
        "pivotRefinement": pivot_refinement,
        "radius": round(calibration.radius, 2),
        "initialAngle": round(calibration.initial_angle, 3),
        "meanConfidence": round(float(np.mean(confidence)), 4),
        "minConfidence": round(float(np.min(confidence)), 4),
        "issueFrames": issue_frames,
        "issueCount": len(issue_frames),
        "issuePct": round(100.0 * len(issue_frames) / len(data), 3),
        "maxForwardBackwardDisagreement": round(float(np.max(disagreement)), 3),
        "maxLocalEvidenceDisagreement": round(float(np.max(local_disagreement)), 3),
        "calibratedStrongFrameCount": int(np.sum(calibrated_strong)),
        "calibratedFallbackFrameCount": int(np.sum(~calibrated_strong)),
        "calibratedValidStrongFrameCount": int(np.sum(valid_strong)),
        "medianCalibratedShaftResidual": round(float(np.median(strong_residual)), 3) if len(strong_residual) else 0.0,
        "p95CalibratedShaftResidual": round(float(np.percentile(strong_residual, 95)), 3) if len(strong_residual) else 0.0,
        "maxCalibratedShaftResidual": round(float(np.max(strong_residual)), 3) if len(strong_residual) else 0.0,
        "p95CalibratedShaftResidualAllFrames": round(float(np.percentile(calibrated_residual, 95)), 3),
        "maxCalibratedShaftResidualAllFrames": round(float(np.max(calibrated_residual)), 3),
        "shaftResidualIssueThresholdDegrees": round(float(shaft_residual_issue_deg), 3),
        "netAngle": round(float(forward_unwrapped[-1] - forward_unwrapped[0]), 3),
        "backgroundSamples": int(meta["samples"]),
        "correctionAnchors": {str(key): round(value, 3) for key, value in sorted(applied_anchors.items())},
        "maskFitUncertainty": {
            "method": "shaft_section_centroids_linear_fit_v1",
            "coverage": "one standard deviation, conditional on the shaft mask",
            "validFrames": int(np.isfinite(fit_std).sum()),
            "unavailableFrames": int((~np.isfinite(fit_std)).sum()),
            "referenceFitAvailable": bool(np.isfinite(fit_std[0])),
        },
    }
    progress(1.0, "Tracking complete")
    return data, summary, background


def draw_overlay(frame: np.ndarray, calibration: Calibration, angle: float, confidence: float, issue: bool) -> np.ndarray:
    output = frame.copy()
    theta = math.radians(float(angle))
    radius = calibration.radius
    point_a = (
        int(round(calibration.pivot_x + radius * math.cos(theta))),
        int(round(calibration.pivot_y - radius * math.sin(theta))),
    )
    point_b = (
        int(round(calibration.pivot_x - radius * math.cos(theta))),
        int(round(calibration.pivot_y + radius * math.sin(theta))),
    )
    color = (0, 120, 255) if issue else (0, 0, 255)
    cv2.line(output, point_a, point_b, color, 4, cv2.LINE_AA)
    cv2.circle(
        output,
        (int(round(calibration.pivot_x)), int(round(calibration.pivot_y))),
        8,
        (255, 220, 0),
        -1,
        cv2.LINE_AA,
    )
    label = f"{angle:.1f} deg  confidence {confidence * 100:.0f}%"
    cv2.putText(output, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(output, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return output
