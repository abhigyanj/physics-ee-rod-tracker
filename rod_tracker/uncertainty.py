"""Conditional angular precision estimated from a fitted shaft mask.

Fit a free-intercept line to 24 spatial section centroids, rather than
treating thousands of adjacent mask pixels as independent measurements.
For v = a + b*u, s_b = sqrt(RSS / ((n-2)*Sxx)) and
s_theta = s_b / (1+b*b) radians. This is a one-standard-deviation estimate
under independent, equal-variance section errors and a straight shaft.
Residual-variance reference: https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd431.htm

This quantifies precision conditional on the selected HSV mask. It does
not measure systematic segmentation, calibration or motion-blur errors.
Displacement combines frame and reference fit variances assuming independent
frame errors; shared-reference errors correlate different displacements.
No target uncertainty is imposed, and missing fits remain unavailable.
"""

import math

import numpy as np


FIT_ANGLE = "Mask Fit Orientation (degrees)"
FIT_STD = "Mask Fit Orientation Standard Uncertainty (degrees)"
DISPLACEMENT_STD = "Mask Fit Displacement Standard Uncertainty (degrees)"
FIT_DIFFERENCE = "Mask Fit Difference (degrees)"


def fit_section_centers(along, perpendicular, seed_angle):
    """Return a line angle and its residual-based standard uncertainty."""
    along = np.asarray(along, dtype=float)
    perpendicular = np.asarray(perpendicular, dtype=float)
    if along.ndim != 1 or along.shape != perpendicular.shape or len(along) < 12:
        return None
    if not np.isfinite(along).all() or not np.isfinite(perpendicular).all():
        return None
    centered = along - along.mean()
    sxx = float(centered @ centered)
    if sxx <= 1e-12:
        return None
    slope = float(centered @ (perpendicular - perpendicular.mean()) / sxx)
    intercept = float(perpendicular.mean() - slope * along.mean())
    residual = perpendicular - (intercept + slope * along)
    residual_variance = float(residual @ residual) / (len(along) - 2)
    slope_std = math.sqrt(residual_variance / sxx)
    return {
        "angleDegrees": (float(seed_angle) + math.degrees(math.atan(slope))) % 180.0,
        "standardUncertaintyDegrees": math.degrees(slope_std / (1.0 + slope * slope)),
        "residualStdPixels": math.sqrt(residual_variance),
        "sections": len(along),
        "spanPixels": float(np.ptp(along)),
    }


def fit_mask_angle(mask, pivot, radius, seed_angle):
    """Fit the central 12%-70% of both arms, excluding pivot and end pieces.

The local search corridor is the same 14-pixel half-width used for pivot
refinement. Broad, absent or one-sided masks cannot establish a shaft fit.
"""
    if mask is None or radius <= 0 or not np.isfinite(seed_angle):
        return None
    yy, xx = np.nonzero(mask)
    dx = xx.astype(float) - float(pivot[0])
    dy = float(pivot[1]) - yy.astype(float)
    theta = math.radians(float(seed_angle))
    along = dx * math.cos(theta) + dy * math.sin(theta)
    perpendicular = -dx * math.sin(theta) + dy * math.cos(theta)
    keep = ((np.abs(along) >= 0.12 * radius)
            & (np.abs(along) <= 0.70 * radius)
            & (np.abs(perpendicular) < 14.0))
    along, perpendicular = along[keep], perpendicular[keep]
    centers_u, centers_v, counts = [], [], []
    arm_counts = []
    for sign in (-1, 1):
        arm_sections = 0
        edges = np.linspace(0.12 * radius, 0.70 * radius, 13)
        for lower, upper in zip(edges[:-1], edges[1:]):
            section = (sign * along >= lower) & (sign * along < upper)
            values = perpendicular[section]
            if len(values) < 5:
                continue
            # A mask filling the corridor is not evidence of a thin shaft.
            if float(np.ptp(values)) >= 25.0:
                continue
            centers_u.append(float(along[section].mean()))
            centers_v.append(float(values.mean()))
            counts.append(int(len(values)))
            arm_sections += 1
        arm_counts.append(arm_sections)
    if min(arm_counts) < 4 or len(centers_u) < 12:
        return None
    if float(np.ptp(centers_u)) < 0.8 * radius:
        return None
    result = fit_section_centers(centers_u, centers_v, seed_angle)
    if result is not None:
        result["pixels"] = sum(counts)
    return result


def displacement_standard_uncertainty(orientation_std):
    """Propagate the shared first-frame reference without temporal smoothing."""
    values = np.asarray(orientation_std, dtype=float)
    result = np.full(values.shape, np.nan)
    if len(values) and np.isfinite(values[0]):
        result = np.hypot(values, values[0])
        result[0] = 0.0  # theta[0] - theta[0] is identically zero.
    return result
