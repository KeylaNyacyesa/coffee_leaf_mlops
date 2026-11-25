"""Create ready-made files for the video demo.

samples/predict/<Class>_N.jpg   - held-out TEST images (never trained on) for the prediction demo
samples/retrain_batch.zip       - 60 images (12 per class) for the bulk-upload demo; 60 >= the threshold
                                  of 50, so uploading it starts retraining AUTOMATICALLY
samples/retrain_batch_small.zip - 25 images (5 per class); below the threshold, so you start
                                  retraining with the "Retrain model" BUTTON

The retrain batch is made from training images with new random augmentations (rotation + tilt,
crop, colour jitter), so the files are new to the system and pass the duplicate check.
For a real retraining, use genuinely new field photos instead (Healthy most of all).

    python scripts/make_demo_samples.py --per-class 12
"""
import argparse
import io
import random
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import CLASS_NAMES  # noqa: E402
from src.preprocessing import read_manifest  # noqa: E402


def jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    # 90-degree multiple + small tilt; the crop below removes the empty corners of the tilt
    img = img.rotate(90 * rng.randint(0, 3) + rng.uniform(-8, 8), resample=Image.BILINEAR, expand=False)
    w, h = img.size
    c = rng.uniform(0.12, 0.18)
    img = img.crop((int(w * c / 2), int(h * c / 2), int(w * (1 - c / 2)), int(h * (1 - c / 2)))).resize((w, h))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.15))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--predict-per-class", type=int, default=2)
    a = ap.parse_args()
    rng = random.Random(123)
    man = read_manifest()
    out = ROOT / "samples"
    if (out / "predict").exists():
        shutil.rmtree(out / "predict")
    (out / "predict").mkdir(parents=True)

    for c in CLASS_NAMES:
        tests = [r for r in man if r["label"] == c and r["split"] == "test"]
        for i, r in enumerate(rng.sample(tests, a.predict_per_class)):
            shutil.copy(ROOT / r["path"], out / "predict" / f"{c}_{i + 1}{Path(r['path']).suffix}")

    write_batch(out / "retrain_batch.zip", man, a.per_class, rng, tag="new")
    write_batch(out / "retrain_batch_small.zip", man, 5, rng, tag="extra")
    print(f"wrote {out / 'predict'}, retrain_batch.zip and retrain_batch_small.zip")


def write_batch(path: Path, manifest: list[dict], per_class: int, rng: random.Random, tag: str) -> None:
    """A zip with one folder per class, made of freshly augmented training images."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for c in CLASS_NAMES:
            train = [r for r in manifest if r["label"] == c and r["split"] == "train"]
            for i, r in enumerate(rng.sample(train, min(per_class, len(train)))):
                img = jitter(Image.open(ROOT / r["path"]).convert("RGB"), rng)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=92)
                z.writestr(f"{c}/{c}_{tag}_{i + 1:03d}.jpg", buf.getvalue())


if __name__ == "__main__":
    main()
