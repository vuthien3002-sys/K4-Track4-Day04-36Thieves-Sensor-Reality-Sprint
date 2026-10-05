"""
Score every camera frame in a folder: image + lens soiling mask -> Camera Health Score, state, weight.

  python -m camera_health --masks model_outputs/<model>/predictions --out results/health.csv
  python -m camera_health --masks <masks> --images woodscape_input/rgbImages_512 --image-check --out results/health_v2.csv

Output columns: features, severity, health_score (0-100), state (Healthy/Degraded/Unreliable), camera_weight.
"""

import argparse
import pathlib

import pandas as pd
from tqdm import tqdm

from . import assess, camera_from_name, load_config, load_image, load_mask, load_valid_mask
from .io_utils import DEFAULT_CONFIG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks", required=True, help="folder of soiling masks (0..3), WoodScape names")
    parser.add_argument("--images", help="folder of the RGB frames with the same names (needed for --image-check)")
    parser.add_argument("--image-check", action="store_true", help="also flag cameras whose ROI has almost no edges")
    parser.add_argument("--out", required=True, help="output CSV")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.image_check:
        if not args.images:
            parser.error("--image-check needs --images")
        cfg["image_check"]["enabled"] = True
    size = cfg["working_size"]
    rows = []
    for path in tqdm(sorted(pathlib.Path(args.masks).glob("*.png"))):
        cam = camera_from_name(path)
        image = load_image(pathlib.Path(args.images) / path.name, size) if args.images else None
        rows.append({"file": path.name, "camera": cam, **assess(load_mask(path, size), cam, cfg, load_valid_mask(cam, size), image)})
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df["state"].value_counts().to_string())
    print(f"mean health score {df['health_score'].mean():.1f} -> {out}")
