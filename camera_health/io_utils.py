import pathlib

import numpy as np
import yaml
from PIL import Image

PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_DIR / "config.yaml"
VALID_DIR = PACKAGE_DIR / "assets"


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def camera_from_name(path):
    """WoodScape file names end with the camera: 0001_FV.png, 0002_MVL.png ..."""
    return pathlib.Path(path).stem.split("_")[-1]


def load_mask(path, size):
    # NEAREST keeps class ids; the default bicubic resize would invent in-between labels
    return np.array(Image.open(path).resize(tuple(size), Image.NEAREST), dtype=np.uint8)


def load_image(path, size):
    return np.array(Image.open(path).convert("RGB").resize(tuple(size)), dtype=np.uint8)


def load_valid_mask(camera, size):
    """Visible fisheye area of a camera (built by benchmark/build_valid_masks.py); None if not built."""
    p = VALID_DIR / f"valid_{camera}.png"
    if not p.exists():
        return None
    return np.array(Image.open(p).resize(tuple(size), Image.NEAREST)) > 0
