"""
Experiment B - controlled soiling on the cleanest real frames.

Baseline : the 8 least-soiled test frames of each camera (32 frames, GT coverage 3-8%), unchanged.
Condition: synthetic soiling added to the same frames, one factor at a time:
           type {transparent, opaque} x position {centre, periphery} x added coverage {10, 25, 40, 60}%.
Measured : Camera Health Score / state / weight from the soiling mask, and image proxies in the critical ROI
           (edge density, sharpness, contrast) relative to the same frame's baseline.

  python -m benchmark.exp_b_controlled
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm

from camera_health import STATES, assess, camera_from_name, load_config, load_image, load_mask, load_valid_mask
from camera_health.features import roi_for_camera, roi_mask
from camera_health.quality import image_quality
from benchmark.common import CAMERAS, GT_DIR, MAIN_MODEL, MASK_CMAP, MODEL_DIR, RESULTS, RGB_DIR, Log
from benchmark.synth_soiling import apply_soiling, soiling_field, soiling_region

OUT = RESULTS / "exp_b_controlled"
FRAMES_PER_CAMERA = 8
LEVELS = [0.10, 0.25, 0.40, 0.60]
KINDS = ["transparent", "opaque"]
POSITIONS = ["centre", "periphery"]
SEED = 2026
STYLE = {("transparent", "centre"): ("#1e88e5", "-"), ("transparent", "periphery"): ("#1e88e5", "--"), ("opaque", "centre"): ("#e53935", "-"), ("opaque", "periphery"): ("#e53935", "--")}


def base_frames(size):
    names = sorted(p.name for p in (MODEL_DIR / MAIN_MODEL / "predictions").glob("*.png"))
    cov = pd.DataFrame({"file": names, "camera": [camera_from_name(n) for n in names], "coverage": [(load_mask(GT_DIR / n, size) > 0).mean() for n in names]})
    return cov.sort_values("coverage").groupby("camera").head(FRAMES_PER_CAMERA).sort_values(["camera", "coverage"]).reset_index(drop=True)


def conditions():
    yield "baseline", None, 0.0
    for kind in KINDS:
        for pos in POSITIONS:
            for level in LEVELS:
                yield kind, pos, level


def run(cfg, frames, save_dir=None):
    size = cfg["working_size"]
    valid = {c: load_valid_mask(c, size) for c in CAMERAS}
    rows = []
    for i, fr in tqdm(frames.iterrows(), total=len(frames), desc="frames"):
        cam = fr.camera
        img, gt = load_image(RGB_DIR / fr.file, size), load_mask(GT_DIR / fr.file, size)
        region_roi = roi_mask(gt.shape, roi_for_camera(cam, cfg)) & valid[cam]
        fields = {p: soiling_field(gt.shape, SEED + i, p) for p in POSITIONS}
        for kind, pos, level in conditions():
            if kind == "baseline":
                out_img, out_mask = img, gt
            else:
                out_img, out_mask = apply_soiling(img, gt, soiling_region(fields[pos], level, valid[cam]), kind, SEED + i)
            cond = "baseline" if kind == "baseline" else f"{kind}/{pos}/{int(level * 100)}%"
            if save_dir is not None:
                stem = f"{fr.file[:-4]}__{cond.replace('/', '_').replace('%', '')}"
                Image.fromarray(out_img).save(save_dir / "images" / f"{stem}.png")
                Image.fromarray(out_mask).save(save_dir / "masks" / f"{stem}.png")
            rows.append({"file": fr.file, "camera": cam, "condition": cond, "kind": kind, "position": pos, "added_coverage": level, **assess(out_mask, cam, cfg, valid[cam]), **image_quality(out_img, region_roi)})
    df = pd.DataFrame(rows)
    base = df[df.kind == "baseline"].set_index("file")
    for q in ("edge_density", "sharpness", "contrast"):
        df[f"{q}_retained"] = df[q] / df.file.map(base[q])
    return df


def plot_lines(df, column, ylabel, title, path, cfg=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    base = df[df.kind == "baseline"][column]
    for (kind, pos), (color, ls) in STYLE.items():
        d = df[(df.kind == kind) & (df.position == pos)].groupby("added_coverage")[column]
        x = np.r_[0, d.mean().index * 100]
        y = np.r_[base.mean(), d.mean().values]
        e = np.r_[base.std(), d.std().values]
        ax.errorbar(x, y, yerr=e, color=color, ls=ls, marker="o", capsize=3, label=f"{kind}, {pos}")
    if cfg is not None:
        for v, name in ((cfg["thresholds"]["healthy"], "Healthy"), (cfg["thresholds"]["unreliable"], "Unreliable")):
            ax.axhline(v, color="grey", lw=0.8, ls=":")
            ax.text(61, v + 1, f"{name} threshold", fontsize=8, color="grey")
    ax.set(xlabel="Added soiling coverage of the visible area (%)", ylabel=ylabel, title=title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(frame, cfg, path):
    size = cfg["working_size"]
    cam = frame.camera
    valid = load_valid_mask(cam, size)
    img, gt = load_image(RGB_DIR / frame.file, size), load_mask(GT_DIR / frame.file, size)
    field = soiling_field(gt.shape, SEED + int(frame.name), "centre")
    fig, ax = plt.subplots(4, 5, figsize=(18, 14.5))
    for r, kind in enumerate(KINDS):
        for c, level in enumerate([0.0] + LEVELS):
            out_img, out_mask = (img, gt) if level == 0 else apply_soiling(img, gt, soiling_region(field, level, valid), kind, SEED + int(frame.name))
            res = assess(out_mask, cam, cfg, valid)
            ax[2 * r, c].imshow(out_img)
            ax[2 * r, c].set_title(f"{kind} centre +{int(level * 100)}%\nscore {res['health_score']:.0f} -> {res['state']} (w={res['camera_weight']})", fontsize=10)
            ax[2 * r + 1, c].imshow(out_mask, cmap=MASK_CMAP, vmin=0, vmax=3, interpolation="nearest")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"Exp B: controlled soiling on {frame.file} (mask colours: blue transparent, amber semi, red opaque)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()
    log = Log(OUT / "log.txt")
    frames = base_frames(cfg["working_size"])
    frames.to_csv(OUT / "base_frames.csv", index=False)
    save_dir = RESULTS.parent / "woodscape_input" / "synthetic_soiling"  # images reused by exp_c, not committed
    (save_dir / "images").mkdir(parents=True, exist_ok=True)
    (save_dir / "masks").mkdir(parents=True, exist_ok=True)
    df = run(cfg, frames, save_dir)
    df.to_csv(OUT / "per_frame.csv", index=False)

    log(f"# Exp B - controlled soiling: {len(frames)} base frames ({FRAMES_PER_CAMERA} per camera), {df.condition.nunique()} conditions, seed {SEED}")
    log(f"Base frame GT coverage: {frames.coverage.min():.1%} - {frames.coverage.max():.1%}")
    order = list(dict.fromkeys(df.condition))
    summary = df.groupby("condition", sort=False).agg(
        coverage=("coverage", "mean"),
        score_mean=("health_score", "mean"),
        score_std=("health_score", "std"),
        weight_mean=("camera_weight", "mean"),
        healthy=("state", lambda s: (s == "Healthy").mean()),
        degraded=("state", lambda s: (s == "Degraded").mean()),
        unreliable=("state", lambda s: (s == "Unreliable").mean()),
        edge_retained=("edge_density_retained", "mean"),
        sharpness_retained=("sharpness_retained", "mean"),
        contrast_retained=("contrast_retained", "mean"),
    ).reindex(order)
    summary.to_csv(OUT / "summary_conditions.csv")
    log("\n## Per condition (mean over frames)")
    log(summary.round(3).to_string())

    log("\n## Claim checks")
    checks = []
    for kind in KINDS:
        for pos in POSITIONS:
            d = df[(df.kind == kind) & (df.position == pos)].pivot(index="file", columns="added_coverage", values="health_score")
            d.insert(0, 0.0, df[df.kind == "baseline"].set_index("file").health_score)
            mono = (np.diff(d.values, axis=1) <= 1e-9).all(axis=1).mean()
            checks.append(f"C1 score never rises with coverage ({kind}, {pos}): {mono:.0%} of frames")
    for pos in POSITIONS:
        for level in LEVELS:
            t = df[(df.kind == "transparent") & (df.position == pos) & (df.added_coverage == level)].set_index("file").health_score
            o = df[(df.kind == "opaque") & (df.position == pos) & (df.added_coverage == level)].set_index("file").health_score
            checks.append(f"C2 opaque < transparent at {pos} +{int(level * 100)}%: {(o < t).mean():.0%} of frames, mean gap {(t - o).mean():.1f} pts")
    for kind in KINDS:
        for level in LEVELS:
            c = df[(df.kind == kind) & (df.position == "centre") & (df.added_coverage == level)].set_index("file").health_score
            p = df[(df.kind == kind) & (df.position == "periphery") & (df.added_coverage == level)].set_index("file").health_score
            checks.append(f"C3 centre < periphery for {kind} +{int(level * 100)}%: {(c < p).mean():.0%} of frames, mean gap {(p - c).mean():.1f} pts")
    soiled = df[df.kind != "baseline"]
    for q in ("edge_density_retained", "sharpness_retained", "contrast_retained"):
        rho, pval = spearmanr(soiled.health_score, soiled[q])
        checks.append(f"C4 Spearman(health score, ROI {q}) = {rho:.2f} (p={pval:.1e}, n={len(soiled)})")
    for c in checks:
        log(c)

    plot_lines(df, "health_score", "Camera Health Score (0-100)", "Exp B: health score vs controlled soiling", OUT / "score_vs_coverage.png", cfg)
    plot_lines(df, "edge_density_retained", "ROI edge density / baseline", "Exp B: image information left in the critical ROI", OUT / "edges_vs_coverage.png")
    plot_examples(frames.iloc[0], cfg, OUT / "example_strip.png")
