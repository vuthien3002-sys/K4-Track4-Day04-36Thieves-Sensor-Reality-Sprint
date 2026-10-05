import numpy as np
import pytest

from camera_health import assess, load_config
from camera_health.features import roi_mask

CFG = load_config()
H = W = 512


def blob(cx, cy, radius, value):
    yy, xx = np.mgrid[:H, :W]
    m = np.zeros((H, W), dtype=np.uint8)
    m[np.hypot(xx - cx, yy - cy) <= radius] = value
    return m


def test_clean_camera_is_healthy():
    r = assess(np.zeros((H, W), np.uint8), "FV", CFG)
    assert r["health_score"] == 100.0
    assert (r["state"], r["camera_weight"]) == ("Healthy", 1.0)


def test_fully_opaque_camera_is_unreliable():
    r = assess(np.full((H, W), 3, np.uint8), "FV", CFG)
    assert r["health_score"] == 0.0
    assert (r["state"], r["camera_weight"]) == ("Unreliable", 0.0)


def test_same_coverage_more_opaque_scores_lower():
    scores = [assess(blob(256, 256, 120, c), "FV", CFG)["health_score"] for c in (1, 2, 3)]
    assert scores[0] > scores[1] > scores[2]


def test_ratios_split_the_soiled_area():
    m = blob(150, 256, 60, 1) + blob(360, 256, 60, 3)
    r = assess(m, "FV", CFG)
    assert r["transparent_ratio"] == pytest.approx(0.5, abs=0.01)
    assert r["opaque_ratio"] == pytest.approx(0.5, abs=0.01)
    assert r["semi_ratio"] == 0.0


def test_centre_soiling_costs_more_than_corner_soiling():
    centre = assess(blob(256, 256, 70, 3), "FV", CFG)
    corner = assess(blob(75, 75, 70, 3), "FV", CFG)  # fully inside the frame: same area
    assert centre["coverage"] == pytest.approx(corner["coverage"])
    assert centre["health_score"] < corner["health_score"]
    assert centre["soiling_centroid_r"] < corner["soiling_centroid_r"]


def test_opaque_roi_forces_unreliable():
    roi = roi_mask((H, W), CFG["roi"]["FV"])
    m = np.zeros((H, W), np.uint8)
    ys, xs = np.where(roi)
    m[ys.min() : ys.max() + 1, xs.min() : (xs.min() + xs.max()) // 2 + 20] = 3  # a bit over half the ROI
    r = assess(m, "FV", CFG)
    assert r["roi_opaque"] >= CFG["roi_opaque_override"]
    assert r["override"] and r["state"] == "Unreliable"


def test_pixels_outside_fisheye_are_ignored():
    valid = blob(256, 256, 230, 1).astype(bool)
    m = np.where(valid, 0, 3).astype(np.uint8)  # soiling only outside the visible circle
    r = assess(m, "FV", CFG, valid=valid)
    assert r["coverage"] == 0.0 and r["health_score"] == 100.0


def test_image_check_flags_a_blind_camera_only_when_enabled():
    mask = np.zeros((H, W), np.uint8)
    flat = np.full((H, W, 3), 128, np.uint8)  # no edges at all, e.g. a lens fogged by a transparent film
    assert assess(mask, "FV", CFG, image=flat)["state"] == "Healthy"  # off by default
    cfg = {**CFG, "image_check": {**CFG["image_check"], "enabled": True}}
    r = assess(mask, "FV", cfg, image=flat)
    assert r["image_flag"] and r["state"] == "Unreliable" and r["camera_weight"] == 0.0
    yy, xx = np.mgrid[:H, :W]
    checker = (((yy // 8 + xx // 8) % 2) * 255).astype(np.uint8)[..., None].repeat(3, -1)
    assert assess(mask, "FV", cfg, image=checker)["state"] == "Healthy"


def test_state_thresholds():
    t = CFG["thresholds"]
    for frac, expected in [(0.05, "Healthy"), (0.35, "Degraded"), (0.9, "Unreliable")]:
        m = np.zeros((H, W), np.uint8)
        half = int(W * frac / 2)
        m[:, W // 2 - half : W // 2 + half] = 2  # centred vertical band of semi-transparent soiling
        r = assess(m, "FV", CFG)
        assert r["state"] == expected, (frac, r["health_score"], t)
