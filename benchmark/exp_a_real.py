"""
Experiment A - real WoodScape test frames (497 frames, 4 cameras).

Baseline : camera health computed from the ground-truth soiling mask (gtLabels).
Condition: camera health computed from the mask predicted by the repo's segmentation models.
Question : how much do segmentation errors change the camera decision (score, state, weight)?

  python -m benchmark.exp_a_real
"""

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from camera_health import STATES, assess, camera_from_name, load_config, load_mask, load_valid_mask
from camera_health.features import roi_for_camera
from benchmark.common import CAMERAS, GT_DIR, MAIN_MODEL, MASK_CMAP, MODEL_DIR, MODELS, RESULTS, RGB_DIR, STATE_RANK, Log

OUT = RESULTS / "exp_a_real"
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # kappa is undefined when a bin has one state
COVERAGE_BINS = [0, 0.1, 0.3, 0.6, 1.01]
COVERAGE_LABELS = ["<10%", "10-30%", "30-60%", ">60%"]


def score_frames(cfg):
    size = cfg["working_size"]
    valid = {c: load_valid_mask(c, size) for c in CAMERAS}
    names = sorted(p.name for p in (MODEL_DIR / MAIN_MODEL / "predictions").glob("*.png"))
    rows, conf = [], {m: np.zeros((4, 4), np.int64) for m in MODELS}
    for name in tqdm(names, desc="scoring"):
        cam = camera_from_name(name)
        gt = load_mask(GT_DIR / name, size)
        rows.append({"file": name, "camera": cam, "source": "GT", **assess(gt, cam, cfg, valid[cam])})
        for model in MODELS:
            pred = load_mask(MODEL_DIR / model / "predictions" / name, size)
            rows.append({"file": name, "camera": cam, "source": model, **assess(pred, cam, cfg, valid[cam])})
            v = valid[cam]
            conf[model] += np.bincount(gt[v].astype(np.int64) * 4 + pred[v], minlength=16).reshape(4, 4)
    return pd.DataFrame(rows), conf


def iou_from_conf(cm):
    tp = np.diag(cm)
    return tp / (cm.sum(0) + cm.sum(1) - tp)


def decision_metrics(gt, pred):
    """gt and pred are aligned rows of the same frames."""
    g, p = gt["state"].values, pred["state"].values
    gt_unrel, gt_healthy = g == "Unreliable", g == "Healthy"
    err = pred["health_score"].values - gt["health_score"].values
    return {
        "n": len(g),
        "score_MAE": np.abs(err).mean(),
        "score_bias": err.mean(),  # > 0: the model makes the camera look healthier than it is
        "state_acc": (g == p).mean(),
        "kappa": cohen_kappa_score(g, p, labels=STATES, weights="linear"),
        "unsafe_rate": ((p == "Healthy") & gt_unrel).sum() / max(gt_unrel.sum(), 1),
        "missed_unreliable": ((p != "Unreliable") & gt_unrel).sum() / max(gt_unrel.sum(), 1),
        "false_alarm": ((p == "Unreliable") & gt_healthy).sum() / max(gt_healthy.sum(), 1),
        "weight_MAE": np.abs(pred["camera_weight"].values - gt["camera_weight"].values).mean(),
    }


def states_with_thresholds(df, healthy, unreliable, cfg):
    s = np.where(df["health_score"] >= healthy, "Healthy", np.where(df["health_score"] < unreliable, "Unreliable", "Degraded"))
    return np.where(df["roi_opaque"] >= cfg["roi_opaque_override"], "Unreliable", s)


def ablation(gt, pred, cfg):
    """Same thresholds, simpler scores: is the extra information robust to segmentation errors?"""
    t = cfg["thresholds"]
    variants = {
        "coverage only": lambda d: 100 * (1 - d["coverage"]),
        "coverage x opacity": lambda d: 100 * (1 - d["effective_occlusion"]),
        "full pipeline": lambda d: d["health_score"],
    }
    rows = []
    for name, fn in variants.items():
        sg, sp = fn(gt), fn(pred)
        stg = np.where(sg >= t["healthy"], "Healthy", np.where(sg < t["unreliable"], "Unreliable", "Degraded"))
        stp = np.where(sp >= t["healthy"], "Healthy", np.where(sp < t["unreliable"], "Unreliable", "Degraded"))
        if name == "full pipeline":
            stg, stp = gt["state"].values, pred["state"].values
        unrel = stg == "Unreliable"
        rows.append(
            {
                "score": name,
                "score_MAE": np.abs(sp.values - sg.values).mean(),
                "state_acc": (stg == stp).mean(),
                "GT_unreliable_share": unrel.mean(),
                "unsafe_rate": ((stp == "Healthy") & unrel).sum() / max(unrel.sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def plot_scatter(gt, pred, cfg, path):
    t = cfg["thresholds"]
    fig, ax = plt.subplots(figsize=(7, 7))
    for cam, color in zip(CAMERAS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]):
        sel = gt["camera"].values == cam
        ax.scatter(gt["health_score"][sel], pred["health_score"][sel], s=12, alpha=0.6, label=cam, color=color)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    for v in (t["healthy"], t["unreliable"]):
        ax.axvline(v, color="grey", lw=0.8)
        ax.axhline(v, color="grey", lw=0.8)
    ax.fill_between([0, t["unreliable"]], t["healthy"], 100, color="red", alpha=0.08, label="unsafe: GT Unreliable, pred Healthy")
    ax.set(xlabel="Health score from GT mask", ylabel=f"Health score from predicted mask ({MODELS[MAIN_MODEL]})", xlim=(0, 100), ylim=(0, 100))
    ax.set_title("Exp A: camera health, GT vs predicted soiling mask")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(gt, pred, path):
    cm = confusion_matrix(gt["state"], pred["state"], labels=STATES)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), STATES)
    ax.set_yticks(range(3), STATES)
    ax.set(xlabel=f"State from predicted mask ({MODELS[MAIN_MODEL]})", ylabel="State from GT mask")
    ax.set_title("Exp A: camera state confusion")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def plot_models(summary, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(summary))
    ax[0].bar(x - 0.2, summary["seg_mIoU"], 0.4, label="segmentation mIoU")
    ax[0].bar(x + 0.2, summary["state_acc"], 0.4, label="camera state accuracy")
    ax[0].set_xticks(x, summary["label"])
    ax[0].set_ylim(0, 1.2)
    ax[0].legend(loc="upper left", ncol=2)
    ax[0].set_title("Pixel metric vs decision metric")
    ax[1].bar(x - 0.2, summary["missed_unreliable"], 0.4, label="missed Unreliable", color="#e53935")
    ax[1].bar(x + 0.2, summary["false_alarm"], 0.4, label="false alarm (GT Healthy -> Unreliable)", color="#8e24aa")
    ax[1].set_xticks(x, summary["label"])
    ax[1].legend()
    ax[1].set_title("Decision errors per segmentation model")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_failures(cases, cfg, path):
    size = cfg["working_size"]
    fig, axes = plt.subplots(len(cases), 3, figsize=(13, 4.3 * len(cases)))
    axes = np.atleast_2d(axes)
    for ax, (_, c) in zip(axes, cases.iterrows()):
        img = np.array(Image.open(RGB_DIR / c["file"]).convert("RGB").resize(tuple(size)))
        gt = load_mask(GT_DIR / c["file"], size)
        pred = load_mask(MODEL_DIR / MAIN_MODEL / "predictions" / c["file"], size)
        x0, y0, x1, y1 = roi_for_camera(c["camera"], cfg)
        for a, data, title in [
            (ax[0], img, f"{c['file']}  ({c['camera']})"),
            (ax[1], gt, f"GT: score {c['gt_score']:.0f} -> {c['gt_state']}"),
            (ax[2], pred, f"{MODELS[MAIN_MODEL]}: score {c['pred_score']:.0f} -> {c['pred_state']}"),
        ]:
            if data.ndim == 2:
                a.imshow(data, cmap=MASK_CMAP, vmin=0, vmax=3, interpolation="nearest")
            else:
                a.imshow(data)
            a.add_patch(patches.Rectangle((x0 * size[0], y0 * size[1]), (x1 - x0) * size[0], (y1 - y0) * size[1], fill=False, ec="white", lw=1.5, ls="--"))
            a.set_title(title, fontsize=10)
            a.axis("off")
    fig.suptitle("Mask colours: dark = clear, blue = transparent, amber = semi-transparent, red = opaque; dashed box = critical ROI", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pixel_confusion_of(file, cfg):
    size = cfg["working_size"]
    cam = camera_from_name(file)
    v = load_valid_mask(cam, size)
    gt = load_mask(GT_DIR / file, size)[v]
    pred = load_mask(MODEL_DIR / MAIN_MODEL / "predictions" / file, size)[v]
    names = ["clear", "transp", "semi", "opaque"]
    cm = np.bincount(gt.astype(np.int64) * 4 + pred, minlength=16).reshape(4, 4)
    worst = sorted(((cm[i, j], f"GT {names[i]} -> pred {names[j]}") for i in range(4) for j in range(4) if i != j), reverse=True)[:2]
    return "; ".join(f"{label} {n / v.sum():.0%} of frame" for n, label in worst)


def label_resize_artefact(names, size):
    """The repo resizes label PNGs with PIL's default (bicubic for mode L); count invented labels."""
    changed = []
    for n in names[:100]:
        im = Image.open(GT_DIR / n)
        changed.append((np.array(im.resize(tuple(size))) != np.array(im.resize(tuple(size), Image.NEAREST))).mean())
    return float(np.mean(changed))


if __name__ == "__main__":
    cfg = load_config()
    log = Log(OUT / "log.txt")
    df, conf = score_frames(cfg)
    df.to_csv(OUT / "per_frame.csv", index=False)
    gt = df[df.source == "GT"].reset_index(drop=True)

    log("# Exp A - real WoodScape test frames:", len(gt), "frames,", gt.camera.value_counts().to_dict())
    log("\n## GT camera states (reference)")
    log(pd.crosstab(gt.camera, gt.state).reindex(columns=STATES, fill_value=0).to_string())

    rows = []
    for model, label in MODELS.items():
        pred = df[df.source == model].reset_index(drop=True)
        assert (pred.file.values == gt.file.values).all()
        iou = iou_from_conf(conf[model])
        rows.append({"model": model, "label": label, "seg_mIoU": iou.mean(), "IoU_transparent": iou[1], "IoU_opaque": iou[3], **decision_metrics(gt, pred)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary_models.csv", index=False)
    log("\n## Decision quality per segmentation model (baseline = GT mask)")
    log(summary.drop(columns="model").round(3).to_string(index=False))

    main = df[df.source == MAIN_MODEL].reset_index(drop=True)
    log(f"\n## Main model {MODELS[MAIN_MODEL]}: per camera")
    per_cam = pd.DataFrame([{"camera": c, **decision_metrics(gt[gt.camera == c], main[gt.camera == c])} for c in CAMERAS])
    per_cam.to_csv(OUT / "summary_per_camera.csv", index=False)
    log(per_cam.round(3).to_string(index=False))

    log(f"\n## Main model {MODELS[MAIN_MODEL]}: per GT coverage level (trend as soiling grows)")
    bins = pd.cut(gt["coverage"], COVERAGE_BINS, labels=COVERAGE_LABELS, right=False)
    per_cov = []
    for b in COVERAGE_LABELS:
        sel = (bins == b).values
        per_cov.append({"gt_coverage": b, "gt_score_mean": gt.health_score[sel].mean(), "pred_score_mean": main.health_score[sel].mean(), **decision_metrics(gt[sel], main[sel])})
    per_cov = pd.DataFrame(per_cov)
    per_cov.to_csv(OUT / "summary_per_coverage.csv", index=False)
    log(per_cov.round(3).to_string(index=False))

    log("\n## Threshold sensitivity (main model)")
    sens = []
    for h in (70, 75, 80, 85, 90):
        for u in (40, 50, 60):
            g_s, p_s = states_with_thresholds(gt, h, u, cfg), states_with_thresholds(main, h, u, cfg)
            unrel, healthy = g_s == "Unreliable", g_s == "Healthy"
            sens.append({"healthy>=": h, "unreliable<": u, "GT_unreliable_share": unrel.mean(), "state_acc": (g_s == p_s).mean(), "missed_unreliable": ((p_s != "Unreliable") & unrel).sum() / max(unrel.sum(), 1), "false_alarm": ((p_s == "Unreliable") & healthy).sum() / max(healthy.sum(), 1)})
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "threshold_sensitivity.csv", index=False)
    log(sens.round(3).to_string(index=False))

    log("\n## Ablation: simpler scores with the same thresholds (main model)")
    abl = ablation(gt, main, cfg)
    abl.to_csv(OUT / "ablation_scores.csv", index=False)
    log(abl.round(3).to_string(index=False))

    plot_scatter(gt, main, cfg, OUT / "scatter_gt_vs_pred.png")
    cm = plot_confusion(gt, main, OUT / "state_confusion.png")
    plot_models(summary, OUT / "models_pixel_vs_decision.png")
    log("\n## State confusion (rows GT, cols pred):", STATES)
    log(cm)

    # Failure cases: prediction more optimistic than GT (dangerous), then the worst false alarm
    pair = pd.DataFrame({"file": gt.file, "camera": gt.camera, "gt_score": gt.health_score, "pred_score": main.health_score, "gt_state": gt.state, "pred_state": main.state, "gt_coverage": gt.coverage, "pred_coverage": main.coverage, "gt_roi_opaque": gt.roi_opaque, "pred_roi_opaque": main.roi_opaque})
    pair["rank_gap"] = gt.state.map(STATE_RANK) - main.state.map(STATE_RANK)
    pair["score_gap"] = pair.pred_score - pair.gt_score
    optimistic = pair[pair.rank_gap > 0].sort_values(["rank_gap", "score_gap"], ascending=False)
    pessimistic = pair[pair.rank_gap < 0].sort_values(["rank_gap", "score_gap"])
    cases = pd.concat([optimistic.head(3), pessimistic.head(1)])
    cases["pixel_errors"] = [pixel_confusion_of(f, cfg) for f in cases.file]
    cases.to_csv(OUT / "failure_cases.csv", index=False)
    plot_failures(cases, cfg, OUT / "failure_cases.png")
    log(f"\n## Failure cases: {len(optimistic)} frames look healthier than GT, {len(pessimistic)} look worse")
    log(cases[["file", "gt_score", "pred_score", "gt_state", "pred_state", "pixel_errors"]].round(1).to_string(index=False))

    log(f"\n## Repo note: bicubic label resize changes {label_resize_artefact(list(gt.file), cfg['working_size']):.2%} of GT pixels vs NEAREST")
