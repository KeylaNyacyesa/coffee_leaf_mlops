"""Data acquisition: build data/train and data/test from the Mendeley downloads.

Sources (CC BY 4.0, Jepkoech et al.):
  * JMuBEN  - https://data.mendeley.com/datasets/t2r6rszp5c/1  (Cercospora, Leaf rust, Phoma)
  * JMuBEN2 - https://data.mendeley.com/datasets/tgv3zb82nd/1  (Healthy, Miner)

Usage:
    python -m src.data_acquisition --sources path/to/t2r6rszp5c-1.zip path/to/jmuben2.zip

Steps
  1. Extract the archives (they contain nested zips).
  2. Map folder names to the 5 class labels.
  3. Drop unreadable files and byte-identical duplicates (md5).
  4. Group rotated / flipped copies of the same leaf photo (dihedral perceptual hash).
  5. Split by GROUP into train / val / test so no copy of a test leaf is ever trained on.
  6. Copy files to data/train/<class>/ and data/test/<class>/ and write split_manifest.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from .config import CLASS_NAMES, DATA_DIR, MANIFEST_PATH, SEED
from .preprocessing import IMAGE_EXTS, dihedral_hash, file_md5, write_manifest

FOLDER_TO_CLASS = {
    "healthy": "Healthy", "miner": "Miner", "phoma": "Phoma",
    "cerscospora": "Cercospora", "cercospora": "Cercospora",
    "leaf rust": "Leaf_rust", "leaf_rust": "Leaf_rust", "rust": "Leaf_rust",
}


def _label_for(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        key = part.lower().split("-2021")[0].strip()
        if key in FOLDER_TO_CLASS:
            return FOLDER_TO_CLASS[key]
    return None


def extract_all(sources: list[Path], workdir: Path) -> Path:
    """Extract every zip (recursively) into workdir; plain folders are copied as-is."""
    for i, src in enumerate(sources):
        dst = workdir / f"src{i}"
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            with zipfile.ZipFile(src) as z:
                z.extractall(dst)
    while True:   # nested archives
        inner = [p for p in workdir.rglob("*.zip")]
        if not inner:
            break
        for z in inner:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.with_suffix(""))
            z.unlink()
    return workdir


def collect(workdir: Path) -> tuple[dict, Counter]:
    """Return {class: [(path, md5, group_hash)]} with exact duplicates removed."""
    per_class: dict[str, list] = defaultdict(list)
    raw_counts: Counter = Counter()
    seen: set[str] = set()
    bad = 0
    for p in sorted(workdir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        label = _label_for(p.relative_to(workdir))
        if label is None:
            continue
        raw_counts[label] += 1
        data = p.read_bytes()
        md5 = file_md5(data)
        if md5 in seen:
            continue
        try:
            img = Image.open(p); img.load(); img = img.convert("RGB")
        except Exception:
            bad += 1
            continue
        seen.add(md5)
        per_class[label].append((p, md5, dihedral_hash(img)))
    print(f"Unreadable files skipped: {bad}")
    return per_class, raw_counts


def group_split(items, test_frac=0.2, val_frac=0.15, seed=SEED):
    """Assign whole groups (all rotations/flips of one leaf) to a single split."""
    groups: dict[int, list] = defaultdict(list)
    for it in items:
        groups[it[2]].append(it)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    out = {}
    for i, k in enumerate(keys):
        split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for it in groups[k]:
            out[it[1]] = (split, f"{k:016x}")
    return out, n


def build(sources: list[Path], out_root: Path = DATA_DIR) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        work = extract_all(sources, Path(tmp))
        per_class, raw_counts = collect(work)
        for d in (out_root / "train", out_root / "test"):
            if d.exists():
                shutil.rmtree(d)
        rows, summary = [], {}
        for label in CLASS_NAMES:
            items = per_class.get(label, [])
            assign, n_groups = group_split(items)
            summary[label] = {"raw": raw_counts[label], "unique": len(items), "groups": n_groups,
                              "train": 0, "val": 0, "test": 0}
            for path, md5, _ in items:
                split, group = assign[md5]
                folder = "test" if split == "test" else "train"   # val lives inside data/train
                dst = out_root / folder / label / f"{label}_{md5[:12]}{path.suffix.lower()}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
                rows.append({"path": dst.relative_to(out_root.parent).as_posix(), "label": label,
                             "group": group, "split": split, "md5": md5, "source": "mendeley"})
                summary[label][split] += 1
    write_manifest(rows, out_root / MANIFEST_PATH.name)
    (out_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True, help="zip files or folders downloaded from Mendeley")
    args = ap.parse_args()
    summary = build([Path(s) for s in args.sources])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
