"""Model creation, training, evaluation, saving - and the export for serving.

Used by the notebook (it needs TensorFlow). The API does not import this file: it runs the
exported lightweight version of the model (see src/serving.py).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .config import (BACKBONE_PATH, CLASS_NAMES, CLASS_NAMES_PATH, EMBEDDINGS_PATH, HEAD_PATH, IMG_SIZE,
                     METRICS_PATH, MODEL_PATH, SEED)
from .serving import ClassifierHead, metrics_from_proba, save_embeddings

BASE_NAME = "mobilenetv2_base"


# ---- Architecture ----
def augmentation_block(name="augmentation") -> keras.Sequential:
    """On-the-fly augmentation (only active while training). Leaves have no canonical
    orientation, so full rotations/flips are label-preserving."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(0.25, fill_mode="reflect", seed=SEED),
        layers.RandomZoom(0.15, fill_mode="reflect", seed=SEED),
        layers.RandomContrast(0.15, seed=SEED),
    ], name=name)


def build_transfer_model(n_classes: int = len(CLASS_NAMES), dropout: float = 0.4,
                         l2: float = 1e-4, weights: str | None = "imagenet") -> keras.Model:
    """MobileNetV2 (ImageNet) feature extractor + regularised classification head.

    Input: raw RGB pixels 0-255, shape (128,128,3). Scaling to [-1,1] is part of the model.
    """
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = augmentation_block()(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1, name="scale_to_pm1")(x)
    base = keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
                                          weights=weights, alpha=1.0)
    base._name = BASE_NAME
    base.trainable = False
    x = base(x, training=False)            # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2), name="dense_128")(x)
    x = layers.Dropout(dropout / 2, name="dropout_2")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name="coffee_leaf_mobilenetv2")


def build_baseline_cnn(n_classes: int = len(CLASS_NAMES)) -> keras.Model:
    """Small CNN trained from scratch - the un-optimised baseline for comparison."""
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1 / 255.0)(inputs)
    for f in (16, 32, 64):
        x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def get_base(model: keras.Model) -> keras.Model | None:
    for l in model.layers:
        if isinstance(l, keras.Model) and "mobilenet" in l.name:
            return l
    return None


def unfreeze_top(model: keras.Model, n_layers: int = 30) -> None:
    """Fine-tuning: unfreeze the last n layers of the backbone (BatchNorm stays frozen)."""
    base = get_base(model)
    if base is None:
        return
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def compile_model(model: keras.Model, lr: float = 1e-3) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def default_callbacks(patience: int = 4):
    """Early stopping (keeps the best weights) + learning-rate reduction on plateau."""
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]


def make_dataset(X, y, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(lambda a, b: (tf.cast(a, tf.float32), b)).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(model, X_tr, y_tr, X_val, y_val, epochs=15, batch_size=32, class_weight=None,
          patience=4, verbose=2):
    return model.fit(make_dataset(X_tr, y_tr, batch_size, shuffle=True),
                     validation_data=make_dataset(X_val, y_val, batch_size),
                     epochs=epochs, class_weight=class_weight,
                     callbacks=default_callbacks(patience), verbose=verbose)


# ---- Evaluation ----
def predict_proba(model, X, batch_size=64) -> np.ndarray:
    return model.predict(make_dataset(X, np.zeros(len(X), np.int64), batch_size), verbose=0)


def evaluate(model, X, y, class_names=CLASS_NAMES) -> dict:
    """Accuracy, loss, precision, recall, F1, ROC-AUC and confusion matrix (see serving.metrics_from_proba)."""
    return metrics_from_proba(y, predict_proba(model, X), class_names)


# ---- Persistence ----
def save(model, path: Path = MODEL_PATH, metrics: dict | None = None, class_names=CLASS_NAMES):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.h5")
    try:
        model.save(tmp)
        os.replace(tmp, path)              # atomic swap: running API workers never read half a file
    finally:
        if tmp.exists():
            tmp.unlink()
    CLASS_NAMES_PATH.write_text(json.dumps(list(class_names)))
    if metrics is not None:
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))


def load(path: Path = MODEL_PATH) -> keras.Model:
    return keras.models.load_model(path, compile=False)


# ---- Export for serving (see src/serving.py) ----
def build_feature_extractor(model: keras.Model) -> keras.Model:
    """The trained model up to global average pooling: image (0-255) -> 1280 features.
    The augmentation layers are left out (they only act during training anyway)."""
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = model.get_layer("scale_to_pm1")(inputs)
    x = get_base(model)(x, training=False)
    outputs = model.get_layer("gap")(x)
    return keras.Model(inputs, outputs, name="feature_extractor")


def export_for_serving(model: keras.Model, manifest: list[dict], root: Path) -> dict:
    """Split the trained model into the two files the API uses, and store the features
    of every dataset image (retraining reuses them instead of re-reading all images).

    Writes models/feature_extractor.tflite, models/classifier_head.npz, data/embeddings.npz.
    """
    from .preprocessing import image_to_array

    # 1. Feature extractor -> TensorFlow Lite
    extractor = build_feature_extractor(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(extractor)
    tf.get_logger().setLevel("ERROR")
    with contextlib.redirect_stdout(io.StringIO()):      # the converter prints a long technical dump
        BACKBONE_PATH.write_bytes(converter.convert())

    # 2. Classifier head -> NumPy weights
    W1, b1 = model.get_layer("dense_128").get_weights()
    W2, b2 = model.get_layer("predictions").get_weights()
    ClassifierHead(W1, b1, W2, b2).save(HEAD_PATH)

    # 3. Features of every image in the dataset
    X = np.stack([image_to_array(root / r["path"]) for r in manifest])
    features = extractor.predict(make_dataset(X, np.zeros(len(X), np.int64), 64), verbose=0)
    save_embeddings({
        "features": features,
        "labels": np.array([CLASS_NAMES.index(r["label"]) for r in manifest]),
        "splits": np.array([r["split"] for r in manifest]),
        "md5": np.array([r["md5"] for r in manifest]),
    }, EMBEDDINGS_PATH)
    return {"tflite_mb": round(BACKBONE_PATH.stat().st_size / 1e6, 1), "n_embeddings": int(len(X))}
