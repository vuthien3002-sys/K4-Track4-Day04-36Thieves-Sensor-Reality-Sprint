"""
Experiment C - improvement for the Exp B failure case: add an image check to the mask-only pipeline.

v1 (baseline): state from the soiling mask only (Exp A / Exp B).
v2 (improved): v1, plus Unreliable when ROI edge density < min_ratio x the camera's clean reference.

Calibration: reference = median ROI edge density of real test frames whose GT state is Healthy, frame id < 2500.
Evaluation : real test frames with id >= 2500 (a different recording session: ids 0-2000 vs 4000-5000), and the
             Exp B conditions (does v2 catch heavy transparent soiling?).

  python -m benchmark.exp_c_image_check      (run exp_a_real and exp_b_controlled first)
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from camera_health import load_config, load_image, load_valid_mask
from camera_health.features import roi_for_camera, roi_mask
from camera_health.quality import image_quality
from benchmark.common import MAIN_MODEL, RESULTS, RGB_DIR, Log

OUT = RESULTS / "exp_c_image_check"
RATIOS = [0.1, 0.15, 0.2, 0.25, 0.3]
CHOSEN = 0.2
CALIBRATION_MAX_ID = 2500


def roi_edges(files, cameras, cfg):
    size = cfg["working_size"]
    out = []
    for f, cam in zip(files, cameras):
        roi = roi_mask(tuple(size), roi_for_camera(cam, cfg)) & load_valid_mask(cam, size)
        out.append(image_quality(load_image(RGB_DIR / f, size), roi)["edge_density"])
    return np.array(out)


def v2_states(states, edge_ratio, k):
    return np.where(edge_ratio < k, "Unreliable", states)


if __name__ == "__main__":
    cfg = load_config()
    log = Log(OUT / "log.txt")
    b = pd.read_csv(RESULTS / "exp_b_controlled" / "per_frame.csv")
    a = pd.read_csv(RESULTS / "exp_a_real" / "per_frame.csv")

    gt_all = a[a.source == "GT"].reset_index(drop=True)
    gt_all["edge_density"] = roi_edges(gt_all.file, gt_all.camera, cfg)
    frame_id = gt_all.file.str[:4].astype(int)
    calib = gt_all[(frame_id < CALIBRATION_MAX_ID) & (gt_all.state == "Healthy")]
    ref = calib.groupby("camera").edge_density.median()
    (OUT / "reference_edge_density.json").write_text(json.dumps(ref.round(4).to_dict(), indent=2))
    log("# Exp C - image check on top of the mask-only pipeline")
    log(f"Calibration: {len(calib)} GT-Healthy frames with id < {CALIBRATION_MAX_ID}", calib.camera.value_counts().to_dict())
    log("Reference ROI edge density per camera:", ref.round(4).to_dict())

    # Exp B: does v2 catch the failure case?
    b["edge_ratio"] = b.edge_density / b.camera.map(ref)
    rows = []
    for cond, d in b.groupby("condition", sort=False):
        row = {"condition": cond, "v1_unreliable": (d.state == "Unreliable").mean(), "v1_weight": d.camera_weight.mean()}
        for k in RATIOS:
            row[f"v2_unreliable@{k}"] = (v2_states(d.state.values, d.edge_ratio.values, k) == "Unreliable").mean()
        rows.append(row)
    syn = pd.DataFrame(rows)
    syn.to_csv(OUT / "exp_b_conditions_v1_vs_v2.csv", index=False)
    log("\n## Exp B conditions: share of frames Unreliable, v1 vs v2 (min_ratio)")
    log(syn.round(2).to_string(index=False))

    # Real frames of the other recording session
    eval_files = set(gt_all.file[frame_id >= CALIBRATION_MAX_ID])
    gt = gt_all[gt_all.file.isin(eval_files)].reset_index(drop=True)
    pred = a[(a.source == MAIN_MODEL) & a.file.isin(eval_files)].reset_index(drop=True)
    assert (gt.file.values == pred.file.values).all()
    edge_ratio = gt.edge_density.values / gt.camera.map(ref).values
    real = []
    for k in RATIOS:
        g2, p2 = v2_states(gt.state.values, edge_ratio, k), v2_states(pred.state.values, edge_ratio, k)
        unrel_v1 = gt.state.values == "Unreliable"
        real.append(
            {
                "min_ratio": k,
                "GT_Healthy_flagged": (edge_ratio < k)[gt.state == "Healthy"].mean(),
                "GT_Degraded_flagged": (edge_ratio < k)[gt.state == "Degraded"].mean(),
                "GT_Unreliable_flagged": (edge_ratio < k)[gt.state == "Unreliable"].mean(),
                "unreliable_share_v1": unrel_v1.mean(),
                "unreliable_share_v2": (g2 == "Unreliable").mean(),
                "pred_missed_unreliable_v1": ((pred.state.values != "Unreliable") & unrel_v1).sum() / unrel_v1.sum(),
                "pred_missed_unreliable_v2": ((p2 != "Unreliable") & unrel_v1).sum() / unrel_v1.sum(),
            }
        )
    real = pd.DataFrame(real)
    real.to_csv(OUT / "real_frames_tradeoff.csv", index=False)
    log(f"\n## Real test frames id >= {CALIBRATION_MAX_ID} (n={len(gt)}): cost and benefit of v2")
    log(real.round(3).to_string(index=False))

    flagged = gt.assign(edge_ratio=edge_ratio)[(edge_ratio < CHOSEN) & (gt.state != "Unreliable")]
    log(f"\n## Frames v2 (min_ratio={CHOSEN}) moves to Unreliable although the GT mask is not Unreliable: {len(flagged)}")
    log(flagged[["file", "camera", "state", "health_score", "coverage", "transparent_ratio", "edge_ratio"]].round(3).to_string(index=False))

    # Plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(syn))
    ax[0].bar(x - 0.2, syn.v1_unreliable, 0.4, label="v1 mask only")
    ax[0].bar(x + 0.2, syn[f"v2_unreliable@{CHOSEN}"], 0.4, label=f"v2 + image check (min_ratio {CHOSEN})")
    ax[0].set_xticks(x, syn.condition, rotation=70, fontsize=8)
    ax[0].set(ylabel="share of frames Unreliable", title="Exp C on Exp B conditions")
    ax[0].legend()
    t60 = syn.set_index("condition").loc["transparent/centre/60%"]
    ax[1].plot(RATIOS, [t60[f"v2_unreliable@{k}"] for k in RATIOS], "o-", label="caught: transparent/centre/60% (Exp B)")
    ax[1].plot(RATIOS, real.GT_Healthy_flagged, "s-", label="false alarm: real frames, GT Healthy")
    ax[1].plot(RATIOS, real.GT_Degraded_flagged, "^-", label="real frames, GT Degraded -> Unreliable")
    ax[1].axvline(CHOSEN, color="grey", ls=":")
    ax[1].set(xlabel="min_ratio (ROI edge density / clean reference)", ylabel="share of frames", title="Trade-off of the image check threshold")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "image_check_tradeoff.png", dpi=120)
    plt.close(fig)
