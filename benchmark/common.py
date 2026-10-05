import pathlib

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
GT_DIR = REPO_ROOT / "woodscape_input" / "gtLabels"
RGB_DIR = REPO_ROOT / "woodscape_input" / "rgbImages_512"
MODEL_DIR = REPO_ROOT / "model_outputs"

# Segmentation models of the repo used in the benchmark: best, a second strong one, a middle one, the worst
MODELS = {
    "fpn_resnet18_torch_cross_entropy_correct_files": "FPN-R18",
    "fpn_resnet50_torch_cross_entropy_all_files": "FPN-R50",
    "unet_resnet18_torch_cross_entropy_all_files": "UNet-R18",
    "pan_resnet18_torch_cross_entropy_correct_clear_strict_files": "PAN-R18-strict",
}
MAIN_MODEL = "fpn_resnet18_torch_cross_entropy_correct_files"
STATE_RANK = {"Healthy": 0, "Degraded": 1, "Unreliable": 2}
CAMERAS = ["FV", "RV", "MVL", "MVR"]
MASK_CMAP = matplotlib.colors.ListedColormap(["#1b1b3a", "#4fc3f7", "#ffb300", "#e53935"])  # clear/transp/semi/opaque


class Log:
    """Print and keep a copy of everything in results/<exp>/log.txt."""

    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def __call__(self, *parts):
        text = " ".join(str(p) for p in parts)
        print(text)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
