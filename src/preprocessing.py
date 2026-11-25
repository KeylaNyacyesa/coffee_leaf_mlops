"""Image preprocessing utilities.

Used by:
  * the notebook (EDA + training data preparation)
  * the API (/predict and /upload)
  * the retraining pipeline (preprocessing of newly uploaded images)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import CLASS_NAMES, DATA_DIR, IMG_SIZE, INSIGHTS_PATH, MANIFEST_PATH

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---- Loading single images ----
def open_image(src) -> Image.Image:
    """Open an image from a path, bytes or file-like object and return RGB."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)      # respect phone-camera orientation
    return img.convert("RGB")


def standardize(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Resize to the model input size (the datasets are already 128x128;
    a handful of Cercospora images are 83x128, and user uploads can be anything)."""
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return img


def image_to_array(src, size: int = IMG_SIZE) -> np.ndarray:
    """Path / bytes -> uint8 array of shape (size, size, 3).

    Pixel scaling to [-1, 1] happens INSIDE the model (Rescaling layer), so
    training, the API and retraining all feed raw 0-255 pixels and cannot drift apart.
    """
    return np.asarray(standardize(open_image(src), size), dtype=np.uint8)


def is_valid_image(src) -> bool:
    try:
        img = open_image(src)
        img.load()
        return min(img.size) >= 16
    except Exception:
        return False


def file_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def dihedral_hash(img: Image.Image) -> int:
    """Perceptual difference-hash that is identical for all 8 rotations/flips of an
    image. The source datasets were built by rotating/flipping each photo, so this
    groups those copies together. We split train/test by group to avoid leakage."""
    base = np.asarray(img.convert("L").resize((32, 32)), dtype=np.uint8)
    best = None
    for k in range(4):
        for flip in (False, True):
            b = np.rot90(base, k)
            if flip:
                b = b[:, ::-1]
            s = np.asarray(Image.fromarray(np.ascontiguousarray(b)).resize((9, 8)), dtype=np.int16)
            bits = (s[:, 1:] > s[:, :-1]).flatten()
            v = int("".join("1" if x else "0" for x in bits), 2)
            best = v if best is None else min(best, v)
    return best


# ---- Dataset loading (manifest-driven) ----
def read_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: Path = MANIFEST_PATH) -> None:
    fields = ["path", "label", "group", "split", "md5", "source"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_split(split: str, manifest: list[dict] | None = None, root: Path | None = None):
    """Return (X uint8 [N,128,128,3], y int [N], paths) for 'train', 'val' or 'test'."""
    manifest = manifest if manifest is not None else read_manifest()
    root = root or MANIFEST_PATH.parent.parent
    rows = [r for r in manifest if r["split"] == split]
    X = np.stack([image_to_array(root / r["path"]) for r in rows]) if rows else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    y = np.array([CLASS_NAMES.index(r["label"]) for r in rows], dtype=np.int64)
    return X, y, [r["path"] for r in rows]


def load_folder(folder: Path):
    """Load a <folder>/<class>/*.jpg tree (fallback when no manifest exists)."""
    X, y, paths = [], [], []
    for ci, cname in enumerate(CLASS_NAMES):
        for p in sorted((folder / cname).glob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                X.append(image_to_array(p)); y.append(ci); paths.append(str(p))
    return np.stack(X), np.array(y), paths


def class_weights(y: np.ndarray, n_classes: int = len(CLASS_NAMES)) -> dict:
    """Balanced class weights (n_samples / (n_classes * count)) to counter imbalance."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1
    w = len(y) / (n_classes * counts)
    return {i: float(w[i]) for i in range(n_classes)}


# ---- Image features for the EDA and dashboard ----
def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).convert("HSV"), dtype=np.float32) / 255.0


def image_features(arr: np.ndarray) -> dict:
    """Interpretable features of a leaf image.

    * green_index      - mean Excess-Green (2G-R-B)/255: how 'green' the leaf is
    * lesion_fraction  - share of leaf pixels whose hue is NOT green (yellow/orange/brown
                         spots = rust pustules, Cercospora/Phoma necrosis, miner trails)
    * brightness       - mean V channel
    * texture          - mean gradient magnitude: spots and trails add edges
    * mean_r/g/b       - channel means
    """
    a = arr.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    hsv = _rgb_to_hsv(arr)
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    leaf = (v > 0.12) & (s > 0.15)                    # ignore dark/grey background
    green = (h >= 65) & (h <= 170)
    leaf_n = max(int(leaf.sum()), 1)
    gray = a.mean(axis=2)
    gy, gx = np.gradient(gray)
    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        "green_index": float(((2 * g - r - b) / 255.0).mean()),
        "lesion_fraction": float((leaf & ~green).sum() / leaf_n),
        "brightness": float(v.mean()),
        "texture": float(np.hypot(gx, gy).mean()),
    }


FEATURES = ["green_index", "lesion_fraction", "brightness", "texture", "mean_r", "mean_g", "mean_b"]


def compute_insights(X: np.ndarray, y: np.ndarray, out_path: Path | None = INSIGHTS_PATH,
                     raw_counts: dict | None = None) -> dict:
    """Aggregate per-class feature statistics + histograms for the UI dashboard."""
    feats = [image_features(x) for x in X]
    out = {"classes": CLASS_NAMES, "counts": {}, "features": {}, "raw_counts": raw_counts or {}}
    for ci, c in enumerate(CLASS_NAMES):
        out["counts"][c] = int((y == ci).sum())
    for f in FEATURES:
        vals = np.array([d[f] for d in feats])
        lo, hi = float(vals.min()), float(vals.max())
        bins = np.linspace(lo, hi, 21)
        out["features"][f] = {"bins": bins.round(4).tolist(), "per_class": {}}
        for ci, c in enumerate(CLASS_NAMES):
            cv = vals[y == ci]
            if len(cv) == 0:
                continue
            hist, _ = np.histogram(np.clip(cv, lo, hi), bins=bins)
            out["features"][f]["per_class"][c] = {
                "mean": float(cv.mean()), "std": float(cv.std()), "median": float(np.median(cv)),
                "hist": (hist / max(hist.sum(), 1)).round(4).tolist(),
            }
    out["mean_color"] = {c: [float(X[y == ci][..., k].mean()) for k in range(3)]
                         for ci, c in enumerate(CLASS_NAMES) if (y == ci).any()}
    if out_path:
        Path(out_path).write_text(json.dumps(out, indent=1))
    return out


def load_insights() -> dict:
    """Dashboard statistics for the whole dataset (train + val + test).

    Cached in data/insights.json; the retraining pipeline deletes the cache so it is
    rebuilt (with the new images) the next time the dashboard asks for it."""
    if not INSIGHTS_PATH.exists():
        manifest = read_manifest()
        parts = [load_split(split, manifest) for split in ("train", "val", "test")]
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        summary_path = DATA_DIR / "dataset_summary.json"
        raw_counts = ({k: v["raw"] for k, v in json.loads(summary_path.read_text()).items()}
                      if summary_path.exists() else {})
        compute_insights(X, y, INSIGHTS_PATH, raw_counts=raw_counts)
    return json.loads(INSIGHTS_PATH.read_text())
