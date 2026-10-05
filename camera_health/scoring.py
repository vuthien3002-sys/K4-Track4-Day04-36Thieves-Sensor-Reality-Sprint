"""
Pipeline step: features -> Soiling Severity -> Camera Health Score 0-100 -> Healthy / Degraded / Unreliable -> Camera Weight.
"""

import numpy as np

from .features import compute_features, roi_for_camera, roi_mask
from .quality import image_quality

STATES = ["Healthy", "Degraded", "Unreliable"]


def soiling_severity(features, cfg):
    w = cfg["severity_weights"]
    s = (
        w["global"] * features["effective_occlusion"]
        + w["spatial"] * features["spatial_occlusion"]
        + w["roi"] * features["roi_occlusion"]
    ) / (w["global"] + w["spatial"] + w["roi"])
    return float(np.clip(s, 0.0, 1.0))


def health_score(severity):
    return round(100.0 * (1.0 - severity), 2)


def camera_state(score, features, cfg):
    if features["roi_opaque"] >= cfg["roi_opaque_override"]:
        return "Unreliable"
    t = cfg["thresholds"]
    if score >= t["healthy"]:
        return "Healthy"
    if score < t["unreliable"]:
        return "Unreliable"
    return "Degraded"


def image_check(image, camera, cfg, valid=None):
    """ROI edge density relative to the camera's clean reference; None when the check is off or not calibrated."""
    ic = cfg.get("image_check", {})
    ref = ic.get("reference_edge_density", {}).get(camera)
    if image is None or not ic.get("enabled") or not ref:
        return None
    roi = roi_mask(image.shape[:2], roi_for_camera(camera, cfg))
    if valid is not None:
        roi &= valid
    return image_quality(image, roi)["edge_density"] / ref


def assess(mask, camera, cfg, valid=None, image=None):
    """
    Full pipeline for one camera frame: mask (+ optional image) -> features, severity, score, state, weight.
    The image is only used when cfg["image_check"]["enabled"] is true.
    """
    f = compute_features(mask, camera, cfg, valid)
    sev = soiling_severity(f, cfg)
    score = health_score(sev)
    state = camera_state(score, f, cfg)
    edge_ratio = image_check(image, camera, cfg, valid)
    image_flag = edge_ratio is not None and edge_ratio < cfg["image_check"]["min_ratio"]
    if image_flag:
        state = "Unreliable"
    return {
        **f,
        "severity": sev,
        "health_score": score,
        "state": state,
        "override": f["roi_opaque"] >= cfg["roi_opaque_override"],
        "roi_edge_ratio": edge_ratio,
        "image_flag": image_flag,
        "camera_weight": cfg["camera_weight"][state],
    }
