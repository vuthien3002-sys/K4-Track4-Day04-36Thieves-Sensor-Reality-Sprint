"""
Image-based proxies, used only to check that a lower health score matches real information loss in the image.
They are not part of the score. All values are computed inside the critical ROI.
"""

import numpy as np
from scipy import ndimage

EDGE_THRESHOLD = 0.1  # Sobel magnitude on a [0, 1] grey image


def to_gray(image):
    img = np.asarray(image, dtype=np.float64) / 255.0
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def image_quality(image, region):
    g = to_gray(image)
    lap = ndimage.laplace(g)
    mag = np.hypot(ndimage.sobel(g, axis=0), ndimage.sobel(g, axis=1)) / 4.0
    return {
        "sharpness": float(lap[region].var()),  # Laplacian variance, higher = sharper
        "edge_density": float((mag[region] > EDGE_THRESHOLD).mean()),  # share of pixels on an edge
        "contrast": float(g[region].std()),  # RMS contrast
    }
