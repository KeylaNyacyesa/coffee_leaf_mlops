"""Lightweight model used by the API: no TensorFlow needed at run time.

The trained Keras model (models/coffee_leaf_model.h5) is split into two parts by
`src.model.export_for_serving()` (last section of the notebook):

    image (128x128x3, 0-255)
      -> FEATURE EXTRACTOR   models/feature_extractor.tflite
         (Rescaling + fine-tuned MobileNetV2 + global average pooling -> 1280 numbers)
      -> CLASSIFIER HEAD     models/classifier_head.npz
         (Dense 128 ReLU -> Dense 5 softmax, the trained head's weights)
      -> class probabilities

Both parts together give the same predictions as the .h5 model, but use ~10x less memory,
so the app runs on a free 512 MB cloud instance.

Retraining keeps the feature extractor fixed and fine-tunes the head, starting from the
current head's weights, on stored features of the existing images + the new uploads.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import (BACKBONE_PATH, CLASS_NAMES, EMBEDDINGS_PATH, HEAD_PATH, LEAF_DISTANCE_THRESHOLD,
                     LEAF_NEIGHBOURS)


# ---- Feature extractor (TensorFlow Lite) ----
class FeatureExtractor:
    """Runs the exported MobileNetV2 backbone: images -> 1280-dimensional features."""

    def __init__(self, path: Path = BACKBONE_PATH):
        from ai_edge_litert.interpreter import Interpreter
        threads = int(os.getenv("TF_NUM_THREADS", "0")) or None
        self.interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """images: uint8 or float array [N, 128, 128, 3] with pixel values 0-255."""
        features = []
        for image in images:                       # the exported model takes one image at a time
            self.interpreter.set_tensor(self.input_index, image[None].astype(np.float32))
            self.interpreter.invoke()
            features.append(self.interpreter.get_tensor(self.output_index)[0].copy())
        return np.stack(features).astype(np.float32)


# ---- Is it a coffee leaf at all? ----
NOT_A_LEAF_MESSAGE = ("This doesn't look like a coffee leaf photo (it is far from every image the model "
                      "was trained on), so no diagnosis is given.")


class LeafCheck:
    """The classifier always picks one of its 5 classes, even for a photo of a person.
    This check compares an image's features with those of the training images: a real
    coffee-leaf photo is close to some of them, anything else is far from all of them."""

    def __init__(self, embeddings: dict, threshold: float = LEAF_DISTANCE_THRESHOLD, k: int = LEAF_NEIGHBOURS):
        self.train = _unit_rows(embeddings["features"][embeddings["splits"] == "train"])
        self.threshold, self.k = threshold, k

    def distance(self, features: np.ndarray) -> np.ndarray:
        """Mean cosine distance to the k nearest training images (0 = identical)."""
        similarity = _unit_rows(features) @ self.train.T
        nearest = np.sort(similarity, axis=1)[:, -self.k:]
        return 1 - nearest.mean(axis=1)

    def is_leaf(self, features: np.ndarray) -> np.ndarray:
        return self.distance(features) <= self.threshold


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# ---- Classifier head (NumPy) ----
class ClassifierHead:
    """Dense(128, ReLU) -> Dense(5, softmax), with the weights of the trained Keras head."""

    def __init__(self, W1, b1, W2, b2):
        self.W1, self.b1, self.W2, self.b2 = (np.asarray(a, np.float32) for a in (W1, b1, W2, b2))

    @classmethod
    def load(cls, path: Path = HEAD_PATH) -> "ClassifierHead":
        with np.load(path) as f:
            return cls(f["W1"], f["b1"], f["W2"], f["b2"])

    def save(self, path: Path = HEAD_PATH) -> None:
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        os.replace(tmp, path)                      # atomic: other containers never read half a file

    def copy(self) -> "ClassifierHead":
        return ClassifierHead(self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        hidden = np.maximum(features @ self.W1 + self.b1, 0)
        return softmax(hidden @ self.W2 + self.b2)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


# ---- Fine-tuning the head (used by retraining) ----
def train_head(head: ClassifierHead, X_train, y_train, X_val, y_val, epochs=30, lr=1e-3, l2=1e-4,
               class_weight: dict | None = None, batch_size=64, patience=4, seed=42, on_epoch_end=None):
    """Fine-tune `head` in place with mini-batch Adam, weighted cross-entropy + L2,
    and early stopping on validation loss (the best weights are restored).

    Returns a Keras-style history dict: loss, accuracy, val_loss, val_accuracy per epoch."""
    rng = np.random.default_rng(seed)
    n_classes = head.W2.shape[1]
    weights = np.ones(n_classes, np.float32) if class_weight is None else \
        np.array([class_weight.get(i, 1.0) for i in range(n_classes)], np.float32)
    params = [head.W1, head.b1, head.W2, head.b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps, step = 0.9, 0.999, 1e-7, 0

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss, best_params, epochs_without_improvement = np.inf, None, 0

    for epoch in range(epochs):
        order = rng.permutation(len(X_train))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            X, y = X_train[idx], y_train[idx]
            sample_w = weights[y]

            # forward
            z1 = X @ head.W1 + head.b1
            h = np.maximum(z1, 0)
            p = softmax(h @ head.W2 + head.b2)

            # backward (weighted cross-entropy)
            d_logits = p.copy()
            d_logits[np.arange(len(y)), y] -= 1
            d_logits *= (sample_w / sample_w.sum())[:, None]
            dW2 = h.T @ d_logits + 2 * l2 * head.W2
            db2 = d_logits.sum(axis=0)
            d_hidden = (d_logits @ head.W2.T) * (z1 > 0)
            dW1 = X.T @ d_hidden + 2 * l2 * head.W1
            db1 = d_hidden.sum(axis=0)

            # Adam update
            step += 1
            for i, grad in enumerate((dW1, db1, dW2, db2)):
                m[i] = beta1 * m[i] + (1 - beta1) * grad
                v[i] = beta2 * v[i] + (1 - beta2) * grad ** 2
                m_hat = m[i] / (1 - beta1 ** step)
                v_hat = v[i] / (1 - beta2 ** step)
                params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        loss, acc = _loss_and_accuracy(head, X_train, y_train)
        val_loss, val_acc = _loss_and_accuracy(head, X_val, y_val)
        for key, value in zip(history, (loss, acc, val_loss, val_acc)):
            history[key].append(float(value))
        if on_epoch_end:
            on_epoch_end(f"epoch {epoch + 1}/{epochs}: loss {loss:.4f}, val_loss {val_loss:.4f}, "
                         f"val_accuracy {val_acc:.3f}")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss, epochs_without_improvement = val_loss, 0
            best_params = [p.copy() for p in params]
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_params is not None:
        for p, best in zip(params, best_params):
            p[...] = best
    return history


def _loss_and_accuracy(head: ClassifierHead, X, y):
    p = head.predict_proba(X)
    loss = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)))
    return loss, float((p.argmax(1) == y).mean())


# ---- Stored features of the whole dataset ----
def load_embeddings(path: Path = EMBEDDINGS_PATH) -> dict:
    """{'features': float32 [N,1280], 'labels': int [N], 'splits': str [N], 'md5': str [N]}"""
    with np.load(path, allow_pickle=False) as f:
        return {"features": f["features"].astype(np.float32), "labels": f["labels"],
                "splits": f["splits"], "md5": f["md5"]}


def save_embeddings(data: dict, path: Path = EMBEDDINGS_PATH) -> None:
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, features=data["features"].astype(np.float16), labels=data["labels"],
                        splits=data["splits"], md5=data["md5"])
    os.replace(tmp, path)


# ---- Evaluation metrics (shared by the notebook and retraining) ----
def metrics_from_proba(y_true: np.ndarray, proba: np.ndarray, class_names=CLASS_NAMES) -> dict:
    """Accuracy, loss, precision, recall, F1 (macro + weighted + per class),
    ROC-AUC (one-vs-rest) and confusion matrix."""
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score, roc_auc_score)
    pred = proba.argmax(1)
    labels = list(range(len(class_names)))
    present = sorted(set(y_true.tolist()))
    try:
        p = proba[:, present] / proba[:, present].sum(1, keepdims=True)
        auc = float(roc_auc_score(y_true, p, multi_class="ovr", average="macro", labels=present)) \
            if len(present) > 2 else None
    except ValueError:
        auc = None
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "loss": float(log_loss(y_true, np.clip(proba, 1e-7, 1), labels=labels)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", labels=present, zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", labels=present, zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", labels=present, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", labels=present, zero_division=0)),
        "roc_auc_ovr_macro": auc,
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
        "per_class": classification_report(y_true, pred, labels=labels, target_names=class_names,
                                           output_dict=True, zero_division=0),
    }
