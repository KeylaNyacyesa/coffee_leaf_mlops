"""SQLite database for uploaded training images and retraining runs.

The database lives in data/mlops.db. With Docker Compose, the data/ folder is a shared
volume, so every API container sees the same uploads and the same retraining status.

Table `uploads`: one row per uploaded image
    status: pending  -> waiting for the next retraining run
            used     -> consumed by a retraining run (copied into data/train)
            rejected -> failed preprocessing (unreadable or already in the dataset)

Table `training_runs`: one row per retraining run, including its live progress log.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT,
    label       TEXT,
    md5         TEXT,
    path        TEXT,
    status      TEXT DEFAULT 'pending',
    note        TEXT,
    uploaded_at REAL,
    run_id      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status);

CREATE TABLE IF NOT EXISTS training_runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     REAL,
    finished_at    REAL,
    status         TEXT,      -- running | completed | skipped | failed
    trigger        TEXT,      -- manual | auto | cli
    n_new_images   INTEGER,
    n_train_images INTEGER,
    epochs_run     INTEGER,
    old_accuracy   REAL,
    new_accuracy   REAL,
    old_f1         REAL,
    new_f1         REAL,
    promoted       INTEGER,
    model_version  TEXT,
    message        TEXT,
    log            TEXT DEFAULT ''
);
"""

# A run that has been "running" for longer than this is treated as dead
# (e.g. its container was stopped mid-run), so it can't block retraining forever.
STALE_RUN_SECONDS = 2 * 60 * 60


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")      # lets several containers read while one writes
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


# ---- Uploads ----
def add_upload(filename: str, label: str, md5: str, path: str) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO uploads (filename, label, md5, path, status, uploaded_at) VALUES (?, ?, ?, ?, 'pending', ?)",
            (filename, label, md5, path, time.time()),
        )
        return cur.lastrowid


def md5_exists(md5: str) -> bool:
    with connect() as con:
        row = con.execute("SELECT 1 FROM uploads WHERE md5 = ? AND status != 'rejected' LIMIT 1", (md5,)).fetchone()
        return row is not None


def pending_uploads() -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM uploads WHERE status = 'pending' ORDER BY id")]


def set_upload_status(ids: list[int], status: str, run_id: int | None = None, note: str | None = None) -> None:
    with connect() as con:
        con.executemany(
            "UPDATE uploads SET status = ?, run_id = ?, note = COALESCE(?, note) WHERE id = ?",
            [(status, run_id, note, i) for i in ids],
        )


def upload_stats() -> dict:
    """Counts per status, and per class label."""
    with connect() as con:
        rows = con.execute("SELECT label, status, COUNT(*) AS n FROM uploads GROUP BY label, status").fetchall()
    stats = {"pending": 0, "used": 0, "rejected": 0, "by_label": {}}
    for r in rows:
        stats[r["status"]] += r["n"]
        stats["by_label"].setdefault(r["label"], {})[r["status"]] = r["n"]
    return stats


# ---- Training runs ----
def claim_run(trigger: str, n_new_images: int) -> int | None:
    """Start a new run unless one is already running. Returns the run id, or None if busy.

    BEGIN IMMEDIATE takes the write lock first, so two containers that try to start
    retraining at the same moment can't both succeed.
    """
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        busy = con.execute(
            "SELECT 1 FROM training_runs WHERE status = 'running' AND started_at > ?",
            (time.time() - STALE_RUN_SECONDS,),
        ).fetchone()
        if busy:
            return None
        cur = con.execute(
            "INSERT INTO training_runs (started_at, status, trigger, n_new_images) VALUES (?, 'running', ?, ?)",
            (time.time(), trigger, n_new_images),
        )
        return cur.lastrowid


def append_log(run_id: int, line: str) -> None:
    stamped = time.strftime("%H:%M:%S") + "  " + line + "\n"
    with connect() as con:
        con.execute("UPDATE training_runs SET log = COALESCE(log, '') || ? WHERE id = ?", (stamped, run_id))


def finish_run(run_id: int, **fields) -> None:
    fields["finished_at"] = time.time()
    columns = ", ".join(f"{name} = ?" for name in fields)
    with connect() as con:
        con.execute(f"UPDATE training_runs SET {columns} WHERE id = ?", (*fields.values(), run_id))


def running_run() -> dict | None:
    with connect() as con:
        row = con.execute(
            "SELECT * FROM training_runs WHERE status = 'running' AND started_at > ? ORDER BY id DESC LIMIT 1",
            (time.time() - STALE_RUN_SECONDS,),
        ).fetchone()
        return dict(row) if row else None


def list_runs(limit: int = 20) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,))]
