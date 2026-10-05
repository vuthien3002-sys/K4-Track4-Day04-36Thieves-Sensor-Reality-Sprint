"""
TV1 data checks: what the benchmark data really is before anyone scores it.

  python -m benchmark.tv1_data_checks       (after fetch_woodscape_subset.py and build_valid_masks.py)

Checks: frame counts per camera and split, mask format, soiling coverage, duplicated GT masks inside the test set,
exact-mask overlap between test and train/val, frame-id ranges (recording sessions), visible fisheye area,
and the label-resize artefact of the repo. Writes results/tv1_data/{log.txt, *.png, *.csv}.
Standalone on purpose: it only needs numpy, pandas, PIL and matplotlib.
"""

import collections
import hashlib
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
GT_DIR = REPO_ROOT / "woodscape_input" / "gtLabels"
RGB_DIR = REPO_ROOT / "woodscape_input" / "rgbImages_512"
MODEL_DIR = REPO_ROOT / "model_outputs"
PRED_DIR = MODEL_DIR / "fpn_resnet18_torch_cross_entropy_correct_files" / "predictions"
VALID_DIR = REPO_ROOT / "camera_health" / "assets"
OUT = REPO_ROOT / "results" / "tv1_data"
CAMERAS = ["FV", "RV", "MVL", "MVR"]
CLASSES = ["clear", "transparent", "semi", "opaque"]


def camera(name):
    return pathlib.Path(name).stem.split("_")[-1]


def frame_id(name):
    return int(pathlib.Path(name).stem.split("_")[0])


class Log:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    log = Log(OUT / "log.txt")
    gt_files = sorted(GT_DIR.glob("*.png"))
    test = sorted(p.name for p in PRED_DIR.glob("*.png"))
    test_set = set(test)

    log("# TV1 data checks")
    log("\n## 1. Frames and split")
    log(f"GT masks (all WoodScape soiling frames): {len(gt_files)}")
    log(f"Test frames (= repo prediction files): {len(test)}   train/val: {len(gt_files) - len(test)}   (paper P1: 4503 / 497)")
    split = pd.DataFrame({"file": [p.name for p in gt_files]})
    split["camera"] = split.file.map(camera)
    split["split"] = np.where(split.file.isin(test_set), "test", "train/val")
    log(pd.crosstab(split.camera, split.split).reindex(CAMERAS).to_string())

    log("\n## 2. Mask format")
    g = Image.open(gt_files[0])
    p = Image.open(PRED_DIR / test[0])
    log(f"GT  : mode {g.mode}, size {g.size}, classes 0..3 = {CLASSES}")
    log(f"Pred: mode {p.mode}, size {p.size} (repo resizes to 512x512 before predicting)")

    log("\n## 3. Soiling in the test set (GT, full frame)")
    hashes, rows, class_px = {}, [], np.zeros(4, np.int64)
    for f in gt_files:
        a = np.array(Image.open(f))
        hashes[f.name] = hashlib.md5(a.tobytes()).hexdigest()
        if f.name in test_set:
            counts = np.bincount(a.ravel(), minlength=4)[:4]
            class_px += counts
            rows.append({"file": f.name, "camera": camera(f.name), "coverage": 1 - counts[0] / a.size})
    cov = pd.DataFrame(rows)
    cov.to_csv(OUT / "test_gt_coverage.csv", index=False)
    log("Pixel share per class:", {c: f"{v:.1%}" for c, v in zip(CLASSES, class_px / class_px.sum())})
    q = cov.coverage.quantile([0, 0.25, 0.5, 0.75, 1])
    log(f"Coverage min {q[0]:.1%}, Q1 {q[0.25]:.1%}, median {q[0.5]:.1%}, Q3 {q[0.75]:.1%}, max {q[1]:.1%}")
    log(f"Frames with coverage < 5%: {(cov.coverage < 0.05).sum()}   > 50%: {(cov.coverage > 0.5).sum()}   fully clean: {(cov.coverage == 0).sum()}")
    log(cov.groupby("camera").coverage.describe()[["count", "min", "50%", "max"]].reindex(CAMERAS).round(3).to_string())

    log("\n## 4. Duplicated GT masks (frames of one sequence share the annotation)")
    groups = collections.defaultdict(list)
    for n, h in hashes.items():
        groups[h].append(n)
    test_hashes = collections.Counter(hashes[n] for n in test)
    log(f"All frames: {len(hashes)} -> {len(groups)} distinct masks")
    log(f"Test frames: {len(test)} -> {len(test_hashes)} distinct masks, largest group {max(test_hashes.values())} frames")
    example = max((v for v in groups.values() if len(v) > 1 and v[0] in test_set), key=len)
    log("Example group in test:", sorted(example))

    log("\n## 5. Leakage: test frames whose exact mask is also used by a train/val frame")
    leak = sorted(n for n in test if any(m not in test_set for m in groups[hashes[n]]))
    log(f"{len(leak)} of {len(test)} test frames ({len(leak) / len(test):.1%}): {leak}")
    log("Paper P1 regrouped the split by sequence; what is left is exact-mask overlap only (near-duplicates not checked)")

    log("\n## 6. Frame ids (recording sessions)")
    ids = np.array([frame_id(n) for n in test])
    hist, edges = np.histogram(ids, bins=[0, 1000, 2000, 2500, 3000, 4000, 5000])
    log("Test frames per id range:", {f"{edges[i]}-{edges[i + 1] - 1}": int(hist[i]) for i in range(len(hist))})
    log("-> two separate blocks; Exp C calibrates on id < 2500 and evaluates on id >= 2500")

    log("\n## 7. Visible fisheye area (camera_health/assets, from build_valid_masks.py)")
    valid = {c: np.array(Image.open(VALID_DIR / f"valid_{c}.png")) > 0 for c in CAMERAS}
    log({c: f"{v.mean():.1%}" for c, v in valid.items()})

    log("\n## 8. Repo label-resize artefact")
    changed = []
    for n in test[:100]:
        im = Image.open(GT_DIR / n)
        changed.append((np.array(im.resize((512, 512))) != np.array(im.resize((512, 512), Image.NEAREST))).mean())
    log(f"PIL default (bicubic) vs NEAREST resize of GT labels: {np.mean(changed):.2%} of pixels differ (first 100 test frames) -> negligible")

    log("\n## 9. Repo models (evaluations/stats of model_outputs.zip)")
    ev = []
    for d in sorted(MODEL_DIR.iterdir()):
        s = d / "evaluations" / "stats"
        if (s / "relative_stats_per_class.csv").exists():
            iou = pd.read_csv(s / "relative_stats_per_class.csv", index_col=0).loc["IoU"]
            acc = pd.read_csv(s / "general_stats.csv", index_col=0).iloc[0].Accuracy
            ev.append({"model": d.name, "accuracy": acc, "mIoU": iou.mean(), "IoU_transparent": iou["Transparent"], "IoU_opaque": iou["Opaque"]})
    ev = pd.DataFrame(ev).sort_values("mIoU", ascending=False)
    ev.to_csv(OUT / "repo_models_ranking.csv", index=False)
    log(f"{len(ev)} models. Best 3 and worst 1 by mIoU:")
    log(pd.concat([ev.head(3), ev.tail(1)]).round(3).to_string(index=False))
    log(f"IoU transparent over all models: {ev.IoU_transparent.min():.2f}-{ev.IoU_transparent.max():.2f}; opaque: {ev.IoU_opaque.min():.2f}-{ev.IoU_opaque.max():.2f}")

    # Plots
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist([cov.coverage[cov.camera == c] * 100 for c in CAMERAS], bins=np.arange(0, 105, 5), label=CAMERAS, stacked=True)
    ax.set(xlabel="GT soiling coverage of the frame (%)", ylabel="test frames", title=f"Test set soiling coverage ({len(cov)} frames, none clean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "coverage_hist.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, c in zip(axes, CAMERAS):
        sample = cov[cov.camera == c].sort_values("coverage").iloc[0].file
        ax.imshow(Image.open(RGB_DIR / sample))
        edge = valid[c] & ~ndimage.binary_erosion(valid[c], iterations=3)
        ax.imshow(np.ma.masked_where(~edge, edge), cmap="autumn", interpolation="nearest")
        ax.set_title(f"{c}: visible area {valid[c].mean():.1%}\n{sample}")
        ax.axis("off")
    fig.suptitle("Fisheye valid area per camera (red outline); pixels outside are ignored by the health score")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "valid_masks.png", dpi=90)
    plt.close(fig)
    log("\nPlots: results/tv1_data/coverage_hist.png, results/tv1_data/valid_masks.png")
