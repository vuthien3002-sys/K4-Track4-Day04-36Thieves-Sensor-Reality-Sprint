"""
Soiling features from a lens soiling mask (0 clear, 1 transparent, 2 semi-transparent, 3 opaque).

Pipeline step: Predicted mask -> Coverage, Transparent ratio, Opaque ratio, Spatial position, Critical ROI coverage.
"""

import numpy as np

CLEAR, TRANSPARENT, SEMI, OPAQUE = 0, 1, 2, 3


def radius_map(shape):
    """Normalised distance from the image centre (1.0 = middle of an edge, ~1.41 = corner)."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return np.hypot((xx - (w - 1) / 2) / (w / 2), (yy - (h - 1) / 2) / (h / 2))


def roi_mask(shape, box):
    """Boolean mask of a normalised [x0, y0, x1, y1] box."""
    h, w = shape
    x0, y0, x1, y1 = box
    out = np.zeros(shape, dtype=bool)
    out[int(round(y0 * h)) : int(round(y1 * h)), int(round(x0 * w)) : int(round(x1 * w))] = True
    return out


def roi_for_camera(camera, cfg):
    return cfg["roi"].get(camera, cfg["roi"]["default"])


def compute_features(mask, camera, cfg, valid=None):
    """
    mask:   HxW integer array with classes 0..3
    camera: FV / RV / MVL / MVR (selects the critical ROI)
    valid:  optional HxW bool array, the visible fisheye area; pixels outside are ignored
    """
    mask = np.asarray(mask)
    if valid is None:
        valid = np.ones(mask.shape, dtype=bool)
    opacity = np.asarray(cfg["class_opacity"], dtype=np.float64)[np.clip(mask, 0, 3)]
    soiled = (mask > CLEAR) & valid
    n_valid = max(int(valid.sum()), 1)
    n_soiled = int(soiled.sum())

    counts = {c: int(((mask == c) & valid).sum()) for c in (TRANSPARENT, SEMI, OPAQUE)}

    def share(c):
        return counts[c] / n_soiled if n_soiled else 0.0

    # Spatial position: centre-weighted occlusion and where the soiling mass sits
    r = radius_map(mask.shape)
    centre_w = np.exp(-0.5 * (r / cfg["spatial_sigma"]) ** 2) * valid
    occ_valid = opacity * valid
    if occ_valid.sum() > 0:
        yy, xx = np.mgrid[: mask.shape[0], : mask.shape[1]]
        cy = (occ_valid * yy).sum() / occ_valid.sum()
        cx = (occ_valid * xx).sum() / occ_valid.sum()
        h, w = mask.shape
        centroid_r = float(np.hypot((cx - (w - 1) / 2) / (w / 2), (cy - (h - 1) / 2) / (h / 2)))
    else:
        centroid_r = float("nan")

    roi = roi_mask(mask.shape, roi_for_camera(camera, cfg)) & valid
    n_roi = max(int(roi.sum()), 1)

    return {
        "coverage": n_soiled / n_valid,
        "transparent_ratio": share(TRANSPARENT),
        "semi_ratio": share(SEMI),
        "opaque_ratio": share(OPAQUE),
        "effective_occlusion": float(occ_valid.sum() / n_valid),
        "spatial_occlusion": float((opacity * centre_w).sum() / max(centre_w.sum(), 1e-9)),
        "soiling_centroid_r": centroid_r,
        "roi_coverage": float(((mask > CLEAR) & roi).sum() / n_roi),
        "roi_occlusion": float((opacity * roi).sum() / n_roi),
        "roi_opaque": float(((mask == OPAQUE) & roi).sum() / n_roi),
    }
