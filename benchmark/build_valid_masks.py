"""
Build the visible fisheye area of each camera (camera_health/assets/valid_<CAM>.png).

A per-image brightness threshold would also remove dark opaque mud inside the circle, so the area is estimated
once per camera from the median brightness of many frames, like a static calibration mask.
"""

import collections
import pathlib

import numpy as np
from PIL import Image
from scipy import ndimage

from camera_health.io_utils import VALID_DIR, camera_from_name

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RGB_DIR = REPO_ROOT / "woodscape_input" / "rgbImages_512"
DARK_THRESHOLD = 20  # max RGB channel, out of 255

if __name__ == "__main__":
    frames = collections.defaultdict(list)
    for p in sorted(RGB_DIR.glob("*.png")):
        frames[camera_from_name(p)].append(np.array(Image.open(p).convert("RGB")).max(-1))
    VALID_DIR.mkdir(parents=True, exist_ok=True)
    for cam, stack in sorted(frames.items()):
        bright = np.median(np.stack(stack), axis=0) > DARK_THRESHOLD
        labels, n = ndimage.label(bright)
        largest = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
        valid = ndimage.binary_fill_holes(largest)
        Image.fromarray((valid * 255).astype(np.uint8)).save(VALID_DIR / f"valid_{cam}.png")
        print(f"{cam}: {len(stack)} frames, visible area {valid.mean():.1%} of the 512x512 frame")
