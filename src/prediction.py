"""Prediction utilities used by the API (and usable from the command line).

    python -m src.prediction path/to/leaf.jpg

Uses the lightweight serving model (TFLite feature extractor + NumPy head, see
src/serving.py), which gives the same predictions as models/coffee_leaf_model.h5.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .config import BACKBONE_PATH, CLASS_NAMES, CLASS_NAMES_PATH, HEAD_PATH, METRICS_PATH
from .preprocessing import image_to_array
from .serving import ClassifierHead, FeatureExtractor

DISEASE_INFO = {
    "Cercospora": "Cercospora leaf spot (Cercospora coffeicola): brown spots with a pale centre and yellow halo.",
    "Healthy": "No disease detected.",
    "Leaf_rust": "Coffee leaf rust (Hemileia vastatrix): orange-yellow powdery pustules on the leaf underside.",
    "Miner": "Coffee leaf miner (Leucoptera coffeella): larvae tunnel inside the leaf, leaving brown necrotic patches.",
    "Phoma": "Phoma leaf spot / blight (Phoma spp.): dark necrotic lesions, often on leaf margins and tips.",
}


class Predictor:
    """Thread-safe model wrapper. It reloads the classifier head when the file on disk
    changes, e.g. after a retraining run in this or another container."""

    def __init__(self, backbone_path: Path = BACKBONE_PATH, head_path: Path = HEAD_PATH,
                 check_every: float = 10.0):
        self.head_path = Path(head_path)
        self.check_every = check_every
        self._lock = threading.Lock()
        self._last_check = 0.0
        self.class_names = CLASS_NAMES
        self.extractor = FeatureExtractor(backbone_path)     # fixed; retraining only changes the head
        self.load()

    def load(self):
        """(Re)load the classifier head, the class names and the model version."""
        with self._lock:
            self._mtime = self.head_path.stat().st_mtime
            self.head = ClassifierHead.load(self.head_path)
            if CLASS_NAMES_PATH.exists():
                self.class_names = json.loads(CLASS_NAMES_PATH.read_text())
            try:
                self._version = json.loads(METRICS_PATH.read_text()).get("version")
            except (FileNotFoundError, json.JSONDecodeError):
                self._version = None
            self.loaded_at = time.time()

    def maybe_reload(self):
        """Every `check_every` seconds, reload if the head file changed (e.g. after retraining)."""
        now = time.time()
        if now - self._last_check < self.check_every:
            return
        self._last_check = now
        try:
            if self.head_path.stat().st_mtime != self._mtime:
                self.load()
        except FileNotFoundError:
            pass

    @property
    def version(self) -> str:
        if self._version:
            return self._version
        return time.strftime("%Y%m%d-%H%M%S", time.localtime(self._mtime))

    def predict_array(self, image: np.ndarray) -> dict:
        self.maybe_reload()
        with self._lock:
            proba = self.head.predict_proba(self.extractor(image[None]))[0]
        idx = int(proba.argmax())
        label = self.class_names[idx]
        return {
            "prediction": label,
            "confidence": float(proba[idx]),
            "probabilities": {c: float(p) for c, p in zip(self.class_names, proba)},
            "description": DISEASE_INFO.get(label, ""),
            "model_version": self.version,
        }

    def predict(self, src) -> dict:
        """src: file path, bytes or file-like object."""
        started = time.perf_counter()
        result = self.predict_array(image_to_array(src))
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.prediction IMAGE [IMAGE ...]")
    predictor = Predictor()
    for f in sys.argv[1:]:
        r = predictor.predict(f)
        print(f"{f}: {r['prediction']} ({r['confidence']:.1%})")
