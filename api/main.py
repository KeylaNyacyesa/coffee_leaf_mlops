"""FastAPI service: prediction, bulk upload, retraining, monitoring + the web UI.

Run locally (from the project root):
    uvicorn api.main:app --port 8000
Then open http://localhost:8000 (web UI) or http://localhost:8000/docs (Swagger).

Sections in this file:
    1. App setup
    2. Model loading
    3. Monitoring      GET  /health, /stats, /model
    4. Prediction      POST /predict
    5. Data insights   GET  /insights
    6. Upload          POST /upload, GET /uploads
    7. Retraining      POST /retrain, GET /retrain/status
    8. Web UI          GET  /
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # so `src` imports work

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import database as db
from src.config import (AUTO_RETRAIN, CLASS_NAMES, HEAD_PATH, METRICS_PATH,
                        RETRAIN_EPOCHS, RETRAIN_THRESHOLD, ROOT)

UI_DIR = ROOT / "ui"
START_TIME = time.time()
CONTAINER = socket.gethostname()        # tells you which container answered (behind the load balancer)


# ---- 1. App setup ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    if HEAD_PATH.exists():
        get_predictor()                 # load the model before the first request arrives
    yield


app = FastAPI(
    title="CoffeeLeaf AI - Coffee Leaf Disease Classifier",
    description="Predict coffee leaf diseases from images, upload new labelled data and retrain the model.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---- 2. Model loading (lightweight export, see src/serving.py) ----
_predictor = None
_predictor_lock = threading.Lock()


def get_predictor():
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                from src.prediction import Predictor
                _predictor = Predictor()
    return _predictor


# ---- 3. Monitoring ----
STATS = {"requests": 0, "predictions": 0, "errors": 0, "not_a_leaf": 0,
         "class_counts": {c: 0 for c in CLASS_NAMES}}
LATENCIES_MS: deque = deque(maxlen=1000)         # last 1000 prediction latencies


@app.middleware("http")
async def count_requests(request: Request, call_next):
    STATS["requests"] += 1
    try:
        return await call_next(request)
    except Exception:
        STATS["errors"] += 1
        raise


@app.get("/health", tags=["monitoring"])
async def health():
    """Is the API up, how long has it been up, and which model is it serving?
    Async, so it answers immediately even while every worker thread is busy predicting."""
    predictor = _predictor
    return {
        "status": "ok" if predictor else "model_not_loaded",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "container": CONTAINER,
        "model_version": predictor.version if predictor else None,
        "model_loaded_at": predictor.loaded_at if predictor else None,
        "retraining": db.running_run() is not None,
    }


@app.get("/stats", tags=["monitoring"])
async def stats():
    """Request counts and prediction latency for THIS container."""
    latencies = sorted(LATENCIES_MS)

    def percentile(q):
        return round(latencies[min(len(latencies) - 1, int(q * len(latencies)))], 2) if latencies else None

    return {
        **STATS,
        "container": CONTAINER,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "latency_ms": {
            "avg": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99),
        },
    }


@app.get("/model", tags=["monitoring"])
def model_info():
    """Evaluation metrics and training history of the model currently in production."""
    if not METRICS_PATH.exists():
        raise HTTPException(404, "metrics.json not found - train the model with the notebook first")
    return json.loads(METRICS_PATH.read_text())


# ---- 4. Prediction ----
@app.post("/predict", tags=["prediction"])
def predict(file: UploadFile = File(..., description="A photo of one coffee leaf (jpg/png)")):
    data = file.file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        result = get_predictor().predict(data)
    except FileNotFoundError:
        raise HTTPException(503, "Model files not found - run the notebook (training + export) first")
    except Exception as error:                   # noqa: BLE001  (e.g. not an image)
        STATS["errors"] += 1
        raise HTTPException(400, f"Could not process the image: {error}")

    STATS["predictions"] += 1
    if result["is_leaf"]:
        STATS["class_counts"][result["prediction"]] += 1
    else:
        STATS["not_a_leaf"] += 1
    LATENCIES_MS.append(result["latency_ms"])
    return {**result, "filename": file.filename, "container": CONTAINER}


@app.get("/classes", tags=["prediction"])
def classes():
    return CLASS_NAMES


# ---- 5. Data insights ----
@app.get("/insights", tags=["data"])
def insights():
    from src.preprocessing import load_insights
    return load_insights()


# ---- 6. Upload new training data ----
@app.post("/upload", tags=["retraining"])
def upload(files: list[UploadFile] = File(..., description="Images, or .zip files with one folder per class"),
           label: str | None = Form(None, description=f"Class of ALL uploaded images (not needed for zips): {CLASS_NAMES}")):
    from src.retrain import normalise_label, save_upload, save_zip

    if label:
        try:
            label = normalise_label(label)
        except ValueError as error:
            raise HTTPException(400, str(error))

    results = []
    for f in files:
        data = f.file.read()
        if f.filename.lower().endswith(".zip"):
            results.extend(save_zip(data))
        elif label:
            results.append(save_upload(f.filename, data, label))
        else:
            results.append({"file": f.filename, "status": "rejected", "reason": "label required for single images"})

    saved = sum(r["status"] == "saved" for r in results)
    pending = len(db.pending_uploads())

    # automatic trigger: enough new data is waiting
    auto_started = AUTO_RETRAIN and pending >= RETRAIN_THRESHOLD and start_retraining_in_background("auto")

    return {
        "saved": saved,
        "rejected": len(results) - saved,
        "pending_total": pending,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "auto_retrain_started": bool(auto_started),
        "details": results[:200],
    }


@app.get("/uploads", tags=["retraining"])
def uploads():
    return {**db.upload_stats(), "retrain_threshold": RETRAIN_THRESHOLD, "auto_retrain": AUTO_RETRAIN}


# ---- 7. Retraining ----
def start_retraining_in_background(trigger: str, epochs: int = RETRAIN_EPOCHS) -> bool:
    """Start retraining in a background thread so the API keeps answering requests.
    Returns False if a run is already going (in this or any other container)."""
    if db.running_run():
        return False

    def job():
        from src.retrain import run_retraining
        try:
            result = run_retraining(trigger=trigger, epochs=epochs)
            if result.get("promoted"):
                get_predictor().load()          # other containers pick it up within 10 s
        except Exception as error:              # noqa: BLE001
            print(f"[retrain] failed: {error}", flush=True)

    threading.Thread(target=job, daemon=True).start()
    return True


@app.post("/retrain", tags=["retraining"])
def retrain(epochs: int = RETRAIN_EPOCHS):
    if not db.pending_uploads():
        raise HTTPException(400, "No new uploaded data. Upload labelled images first.")
    if not start_retraining_in_background("manual", epochs):
        raise HTTPException(409, "A retraining run is already in progress")
    return {"status": "started", "epochs": epochs}


@app.get("/retrain/status", tags=["retraining"])
def retrain_status():
    """Latest run with its live log (read from the shared database) + the run history."""
    runs = db.list_runs(20)
    latest = runs[0] if runs else None
    return {
        "state": "running" if latest and latest["status"] == "running" else "idle",
        "log": (latest["log"] or "").strip().splitlines()[-60:] if latest else [],
        "latest": latest,
        "runs": [{k: v for k, v in r.items() if k != "log"} for r in runs],
    }


# ---- 8. Web UI ----
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_DIR / "index.html")
