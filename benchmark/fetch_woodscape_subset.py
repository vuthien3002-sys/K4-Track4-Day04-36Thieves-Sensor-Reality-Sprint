"""
Fetch only the parts of the two Google Drive zips (README, step 1 and 3) that the camera-health
benchmark needs, using HTTP range requests instead of downloading 5.5 GB + 6.6 GB.

What is fetched (same folder layout as the README):
  model_outputs/<model>/evaluations/stats/*.csv   for every model    (--evals)
  model_outputs/<model>/predictions/*.png          for --models       (498 test masks, 512x512)
  woodscape_input/gtLabels/*.png                   all 5000 GT masks  (--gt)
  woodscape_input/rgbImages_512/*.png              test images only, resized to 512x512 (--rgb)

Usage:
  python benchmark/fetch_woodscape_subset.py --evals
  python benchmark/fetch_woodscape_subset.py --models fpn_resnet18_torch_cross_entropy_correct_clear_strict_files --gt --rgb
"""

import argparse
import io
import pathlib
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from tqdm import tqdm

DATASET_ID = "1WNlDBADwlaheMaVpIjEeAMklw7jle9Ja"  # woodscape_input.zip (5.5 GB)
MODELS_ID = "13k17SjgQHZCO-1Ctr3DY_bW6DGvQZZie"  # model_outputs.zip (6.6 GB)
URL = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MISS_CHUNK = 4 << 20  # bytes fetched on a cache miss (central directory parsing)


def http_range(url, start, end):
    """Return bytes [start, end) of url."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start:
                return data
        except OSError:
            if attempt == 4:
                raise
    raise IOError(f"Short read {start}-{end}")


class RemoteZipFile(io.RawIOBase):
    """Seekable file object over HTTP with an explicit span cache, so zipfile can read from it."""

    def __init__(self, url):
        self.url, self.pos, self.spans = url, 0, {}
        self.lock = threading.Lock()
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def prefetch(self, start, end):
        data = http_range(self.url, start, min(end, self.size))
        with self.lock:
            self.spans[start] = data

    def drop_cache(self):
        with self.lock:
            self.spans = {}

    def _from_cache(self, n):
        with self.lock:
            for start, data in self.spans.items():
                if start <= self.pos < start + len(data):
                    off = self.pos - start
                    return data[off : off + n]
        return None

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        out = b""
        while n > 0 and self.pos < self.size:
            chunk = self._from_cache(n)
            if chunk is None:
                self.prefetch(self.pos, self.pos + max(n, MISS_CHUNK))
                continue
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return out

    def readinto(self, b):
        d = self.read(len(b))
        b[: len(d)] = d
        return len(d)


def open_remote_zip(file_id):
    raw = RemoteZipFile(URL.format(file_id))
    zf = zipfile.ZipFile(raw)
    infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
    # Byte end of each entry = start of the next local header (or the central directory)
    ends = {}
    for cur, nxt in zip(infos, infos[1:] + [None]):
        ends[cur.filename] = nxt.header_offset if nxt else zf.start_dir
    return raw, zf, ends


def group_runs(infos, ends, max_gap=1 << 20, max_run=64 << 20):
    """Merge entries that are close in the archive into a few big range requests."""
    runs = []
    for info in sorted(infos, key=lambda i: i.header_offset):
        start, end = info.header_offset, ends[info.filename]
        if runs and start - runs[-1][1] <= max_gap and end - runs[-1][0] <= max_run:
            runs[-1][1] = end
            runs[-1][2].append(info)
        else:
            runs.append([start, end, [info]])
    return runs


def fetch(raw, zf, ends, infos, save_fn, desc, batch_runs=24, workers=8):
    runs = group_runs(infos, ends)
    with tqdm(total=len(infos), desc=desc) as bar:
        for b in range(0, len(runs), batch_runs):
            batch = runs[b : b + batch_runs]
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda r: raw.prefetch(r[0], r[1]), batch))
            for _, _, members in batch:
                for info in members:
                    save_fn(info, zf.read(info))
                    bar.update(1)
            raw.drop_cache()


def save_plain(root):
    def _save(info, data):
        out = root / info.filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)

    return _save


def save_rgb_resized(out_dir, size=(512, 512)):
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(info, data):
        # Same resize as networks_run/pytorch_networks/predict/pytorch_l_base_pred.py
        img = Image.open(io.BytesIO(data)).convert("RGB").resize(size)
        img.save(out_dir / pathlib.PurePosixPath(info.filename).name)

    return _save


def is_file(info):
    return not info.is_dir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evals", action="store_true", help="evaluation stats CSVs of all models")
    parser.add_argument("--models", nargs="*", default=[], help="model folder names whose predictions to fetch")
    parser.add_argument("--gt", action="store_true", help="all gtLabels")
    parser.add_argument("--rgb", action="store_true", help="RGB of the test images (names from fetched predictions)")
    parser.add_argument("--root", default=str(REPO_ROOT))
    args = parser.parse_args()
    root = pathlib.Path(args.root)

    if args.evals or args.models:
        raw, zf, ends = open_remote_zip(MODELS_ID)
        if args.evals:
            sel = [i for i in zf.infolist() if is_file(i) and "/evaluations/stats/" in i.filename and i.filename.endswith(".csv")]
            fetch(raw, zf, ends, sel, save_plain(root), "evaluations")
        for model in args.models:
            prefix = f"model_outputs/{model}/predictions/"
            sel = [i for i in zf.infolist() if is_file(i) and i.filename.startswith(prefix)]
            if not sel:
                raise SystemExit(f"No predictions for model {model}")
            fetch(raw, zf, ends, sel, save_plain(root), f"pred {model[:30]}")

    if args.gt or args.rgb:
        raw, zf, ends = open_remote_zip(DATASET_ID)
        if args.gt:
            sel = [i for i in zf.infolist() if is_file(i) and i.filename.startswith("woodscape_input/gtLabels/")]
            fetch(raw, zf, ends, sel, save_plain(root), "gtLabels")
        if args.rgb:
            pred_dirs = list((root / "model_outputs").glob("*/predictions"))
            if not pred_dirs:
                raise SystemExit("Fetch predictions first (--models) so the test image names are known")
            test_names = {p.name for d in pred_dirs for p in d.glob("*.png")}
            sel = [
                i
                for i in zf.infolist()
                if is_file(i)
                and i.filename.startswith("woodscape_input/rgbImages/")
                and pathlib.PurePosixPath(i.filename).name in test_names
            ]
            fetch(raw, zf, ends, sel, save_rgb_resized(root / "woodscape_input" / "rgbImages_512"), "rgb test", batch_runs=16)
