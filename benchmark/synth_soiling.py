"""
Controlled synthetic lens soiling: changes one factor at a time (coverage, type, position) on the same frame.

The random blob field depends only on (seed, position), so for one frame the 25% region contains the 10% region,
and transparent / opaque use the same region: only the factor under test changes.
"""

import numpy as np
from scipy import ndimage

from camera_health.features import OPAQUE, TRANSPARENT, radius_map

BLOB_SIGMA = 24  # px at 512x512, size of the soiling blobs
POSITION_BIAS = 1.5  # how strongly blobs are pulled to the centre or to the periphery
EDGE_SOFTNESS = 3  # px, feathered border of the rendered soiling
MUD_RGB = np.array([60.0, 48.0, 36.0])


def soiling_field(shape, seed, position):
    rng = np.random.default_rng(seed)
    noise = ndimage.gaussian_filter(rng.standard_normal(shape), BLOB_SIGMA)
    noise = (noise - noise.mean()) / noise.std()
    r = radius_map(shape)
    bias = -r if position == "centre" else r
    return noise + POSITION_BIAS * (bias - bias.mean()) / bias.std()


def soiling_region(field, coverage, valid):
    """Exactly `coverage` of the visible area, taken where the field is highest."""
    if coverage <= 0:
        return np.zeros(field.shape, dtype=bool)
    thr = np.quantile(field[valid], 1.0 - coverage)
    return (field >= thr) & valid


def apply_soiling(image, mask, region, kind, seed):
    """Render soiling on the image and add it to the label mask (keeps the heavier class where both exist)."""
    img = image.astype(np.float64)
    alpha = ndimage.gaussian_filter(region.astype(np.float64), EDGE_SOFTNESS)[..., None]
    if kind == "opaque":
        rng = np.random.default_rng(seed + 7)
        texture = ndimage.gaussian_filter(rng.standard_normal(region.shape), 6)[..., None] * 40
        layer, alpha = MUD_RGB + texture, 0.97 * alpha
        cls = OPAQUE
    elif kind == "transparent":
        # Water film / haze: the scene stays visible but blurred and washed out
        layer, alpha = 0.7 * ndimage.gaussian_filter(img, (6, 6, 0)) + 0.3 * 230.0, 0.9 * alpha
        cls = TRANSPARENT
    else:
        raise ValueError(kind)
    out = np.clip(img * (1 - alpha) + layer * alpha, 0, 255).astype(np.uint8)
    new_mask = np.where(region, np.maximum(mask, cls), mask).astype(np.uint8)
    return out, new_mask
