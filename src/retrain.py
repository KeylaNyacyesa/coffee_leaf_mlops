"""Upload ingestion + retraining pipeline.

How it fits together
--------------------
  UPLOAD      save_upload() / save_zip()
              New labelled images are saved to data/uploads/<label>/ and registered in
              the SQLite database with status "pending".

  TRIGGER     run_retraining() is called by the "Retrain model" button (POST /retrain),
              automatically by POST /upload once RETRAIN_THRESHOLD images are pending,
              or from the command line:  python -m src.retrain --upload-dir path/to/new_images

  RETRAIN     run_retraining() then runs these steps:
                1. preprocess the pending uploads (validate, RGB, resize, de-duplicate)
                2. extract their features with our trained MobileNetV2 (feature_extractor.tflite)
                3. load the CURRENT production classifier head as the pre-trained starting point
                4. fine-tune it on the stored features of the existing training images + the
                   new ones (Adam, class weights, L2, early stopping on the validation set)
                5. compare old vs new model on the held-out test set
                6. promote the new head only if it is not worse; archive the old one
                7. record everything in the database
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import database as db
from .config import (ARCHIVE_DIR, CLASS_NAMES, HEAD_PATH, METRICS_PATH, PROMOTION_TOLERANCE,
                     RETRAIN_EPOCHS, RETRAIN_LR, ROOT, TRAIN_DIR, UPLOAD_DIR)
from .preprocessing import (IMAGE_EXTS, class_weights, file_md5, image_to_array, is_valid_image,
                            read_manifest, write_manifest)
from .serving import (ClassifierHead, FeatureExtractor, load_embeddings, metrics_from_proba,
                      save_embeddings, train_head)


# ---- Upload ----
def normalise_label(label: str) -> str:
    """Accept 'leaf rust', 'Leaf-Rust', 'cerscospora' (the dataset's spelling), etc."""
    key = label.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"cerscospora": "Cercospora", "rust": "Leaf_rust"}
    if key in aliases:
        return aliases[key]
    for name in CLASS_NAMES:
        if name.lower() == key:
            return name
    raise ValueError(f"Unknown label '{label}'. Expected one of {CLASS_NAMES}")


def save_upload(filename: str, data: bytes, label: str) -> dict:
    """Store one uploaded image on disk and register it in the database."""
    label = normalise_label(label)
    ext = Path(filename).suffix.lower()

    if ext not in IMAGE_EXTS:
        return {"file": filename, "status": "rejected", "reason": "not an image"}
    if not is_valid_image(data):
        return {"file": filename, "status": "rejected", "reason": "unreadable image"}
    md5 = file_md5(data)
    if db.md5_exists(md5):
        return {"file": filename, "status": "rejected", "reason": "duplicate upload"}

    destination = UPLOAD_DIR / label / f"{md5[:16]}{ext}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    db.add_upload(filename, label, md5, destination.relative_to(ROOT).as_posix())
    return {"file": filename, "status": "saved", "label": label}


def save_zip(data: bytes) -> list[dict]:
    """Store every image in a zip whose folders are class names (Healthy/a.jpg, Leaf_rust/b.jpg, ...)."""
    results = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for entry in archive.infolist():
            path = Path(entry.filename)
            if entry.is_dir() or path.suffix.lower() not in IMAGE_EXTS or path.name.startswith("._"):
                continue
            label = _label_from_folders(path)
            if label is None:
                results.append({"file": entry.filename, "status": "rejected", "reason": "no class folder"})
            else:
                results.append(save_upload(path.name, archive.read(entry), label))
    return results


def _label_from_folders(path: Path) -> str | None:
    for folder in reversed(path.parts[:-1]):
        try:
            return normalise_label(folder)
        except ValueError:
            continue
    return None


# ---- Retrain ----
def run_retraining(trigger: str = "manual", epochs: int = RETRAIN_EPOCHS, lr: float = RETRAIN_LR,
                   on_progress=None) -> dict:
    """Retrain the production model on all pending uploads. Returns a result summary."""
    pending = db.pending_uploads()
    if not pending:
        return {"status": "skipped", "message": "No new uploaded data to train on"}

    run_id = db.claim_run(trigger, len(pending))
    if run_id is None:
        return {"status": "busy", "message": "A retraining run is already in progress"}

    def log(message: str):
        print(f"[retrain run {run_id}] {message}", flush=True)
        db.append_log(run_id, message)
        if on_progress:
            on_progress(message)

    started = time.time()
    try:
        log(f"retraining started ({trigger} trigger), {len(pending)} uploaded images pending")

        # 1. Preprocess the uploaded images
        manifest = read_manifest()
        new_rows, X_new, y_new, used_ids = preprocess_uploads(pending, manifest, run_id)
        log(f"step 1 - preprocessed {len(used_ids)} new images ({len(pending) - len(used_ids)} rejected)")
        if not used_ids:
            db.finish_run(run_id, status="skipped", message="all uploads were rejected during preprocessing")
            return {"status": "skipped", "message": "All uploads were rejected during preprocessing"}

        # 2. Features of the new images (from our trained MobileNetV2) + stored features of the dataset
        features_new = FeatureExtractor()(X_new)
        data = load_embeddings()
        is_train, is_val, is_test = (data["splits"] == split for split in ("train", "val", "test"))
        X_train = np.concatenate([data["features"][is_train], features_new])
        y_train = np.concatenate([data["labels"][is_train], y_new])
        X_val, y_val = data["features"][is_val], data["labels"][is_val]
        X_test, y_test = data["features"][is_test], data["labels"][is_test]
        log(f"step 2 - features extracted; training set: {is_train.sum()} existing + {len(y_new)} new images")

        # 3. The current production head is the pre-trained starting point
        current = ClassifierHead.load(HEAD_PATH)
        old = metrics_from_proba(y_test, current.predict_proba(X_test))
        log(f"step 3 - loaded current model (test accuracy {old['accuracy']:.4f})")

        # 4. Fine-tune a copy of it on old + new data
        candidate = current.copy()
        history = train_head(candidate, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr,
                             class_weight=class_weights(y_train),
                             on_epoch_end=lambda msg: log("step 4 - " + msg))

        # 5. Compare old vs new on the held-out test set
        new = metrics_from_proba(y_test, candidate.predict_proba(X_test))
        log(f"step 5 - test accuracy {old['accuracy']:.4f} -> {new['accuracy']:.4f}, "
            f"macro F1 {old['f1_macro']:.4f} -> {new['f1_macro']:.4f}")

        # 6. Promote only if neither accuracy nor macro-F1 got worse (beyond a small tolerance)
        promoted = (new["accuracy"] >= old["accuracy"] - PROMOTION_TOLERANCE
                    and new["f1_macro"] >= old["f1_macro"] - PROMOTION_TOLERANCE)
        version = time.strftime("%Y%m%d-%H%M%S") if promoted else None
        if promoted:
            promote_head(candidate, new, old, history, run_id, version, n_train=len(y_train))
            log(f"step 6 - new model PROMOTED (version {version})")
        else:
            log("step 6 - new model NOT promoted (it was worse) - keeping the current model")

        # 7. Record: the new images (and their features) join the training set either way
        write_manifest(manifest + new_rows)
        save_embeddings({
            "features": np.concatenate([data["features"], features_new]),
            "labels": np.concatenate([data["labels"], y_new]),
            "splits": np.concatenate([data["splits"], np.array(["train"] * len(y_new))]),
            "md5": np.concatenate([data["md5"], np.array([r["md5"] for r in new_rows])]),
        })
        db.set_upload_status(used_ids, "used", run_id)
        result = {
            "status": "completed", "run_id": run_id, "promoted": promoted, "model_version": version,
            "n_new_images": len(used_ids), "n_train_images": int(len(y_train)),
            "epochs_run": len(history["loss"]),
            "old_accuracy": old["accuracy"], "new_accuracy": new["accuracy"],
            "old_f1": old["f1_macro"], "new_f1": new["f1_macro"],
        }
        db.finish_run(run_id, status="completed", n_new_images=len(used_ids), n_train_images=int(len(y_train)),
                      epochs_run=result["epochs_run"], old_accuracy=old["accuracy"], new_accuracy=new["accuracy"],
                      old_f1=old["f1_macro"], new_f1=new["f1_macro"], promoted=int(promoted),
                      model_version=version, message="promoted" if promoted else "kept the current model")
        log(f"step 7 - done in {time.time() - started:.0f} s")
        return result

    except Exception as error:                              # noqa: BLE001
        db.finish_run(run_id, status="failed", message=str(error)[:500])
        log(f"FAILED: {error}")
        raise


def preprocess_uploads(pending: list[dict], manifest: list[dict], run_id: int):
    """Validate + standardise pending uploads and copy them into data/train/<label>/."""
    known_md5 = {row["md5"] for row in manifest}
    new_rows, images, labels, used_ids = [], [], [], []

    for upload in pending:
        try:
            if upload["md5"] in known_md5:
                raise ValueError("already in the dataset")
            image = image_to_array((ROOT / upload["path"]).read_bytes())    # RGB, resized to 128x128
        except Exception as error:                                          # noqa: BLE001
            db.set_upload_status([upload["id"]], "rejected", run_id, str(error)[:120])
            continue

        destination = TRAIN_DIR / upload["label"] / f"{upload['label']}_up_{upload['md5'][:12]}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(destination, quality=95)
        known_md5.add(upload["md5"])

        images.append(image)
        labels.append(CLASS_NAMES.index(upload["label"]))
        used_ids.append(upload["id"])
        new_rows.append({"path": destination.relative_to(ROOT).as_posix(), "label": upload["label"],
                         "group": f"upload-{upload['md5'][:12]}", "split": "train",
                         "md5": upload["md5"], "source": f"upload-run{run_id}"})

    X_new = np.stack(images) if images else np.zeros((0, 128, 128, 3), np.uint8)
    return new_rows, X_new, np.array(labels, dtype=np.int64), used_ids


def promote_head(head, new, old, history, run_id, version, n_train):
    """Archive the current head, then save the retrained one in its place."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HEAD_PATH, ARCHIVE_DIR / f"classifier_head_before_{version}.npz")
    head.save(HEAD_PATH)
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics.update({
        "version": version,
        "retrained_at": time.time(),
        "retrain_run_id": run_id,
        "test": new,
        "previous_test_accuracy": old["accuracy"],
        "retrain_history": history,
        "n_train_images": int(n_train),
    })
    tmp = METRICS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(metrics, indent=2))
    tmp.replace(METRICS_PATH)


# ---- Command line ----
def main():
    parser = argparse.ArgumentParser(description="Retrain the coffee-leaf model on newly uploaded data")
    parser.add_argument("--upload-dir", help="optional folder with <class>/<images> to upload first")
    parser.add_argument("--epochs", type=int, default=RETRAIN_EPOCHS)
    args = parser.parse_args()

    if args.upload_dir:
        for path in Path(args.upload_dir).rglob("*"):
            if path.suffix.lower() in IMAGE_EXTS:
                print(save_upload(path.name, path.read_bytes(), path.parent.name))
    print(json.dumps(run_retraining("cli", args.epochs), indent=2))


if __name__ == "__main__":
    main()
