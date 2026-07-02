"""
db.py — SQLite job queue using aiosqlite.

jobs table columns:
  job_id        TEXT PRIMARY KEY  — Slurm job ID (or UUID before Slurm assigns one)
  user_id       TEXT              — JupyterHub username from X-User header
  status        TEXT              — queued | running | done | failed | cancelled
  catchments    TEXT              — JSON list of catchment ID strings
  submitted_at  TEXT              — ISO-8601 UTC
  finished_at   TEXT              — ISO-8601 UTC or NULL
  slurm_id      TEXT              — actual Slurm job ID once sbatch responds
  output_dir    TEXT              — absolute path to results on shared NFS
  error_msg     TEXT              — stderr snippet on failure
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import aiosqlite

DB_PATH = os.environ.get('HBV_DB_PATH', '/data/hbv/api/jobs.db')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id       TEXT PRIMARY KEY,
                user_id      TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'queued',
                catchments   TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                finished_at  TEXT,
                slurm_id     TEXT,
                output_dir   TEXT,
                error_msg    TEXT,
                n_tasks      INTEGER NOT NULL DEFAULT 1
            )
        """)
        await db.commit()


async def insert_job(job_id: str, user_id: str, catchment_ids: list[str],
                     slurm_id: str | None = None, output_dir: str | None = None,
                     n_tasks: int = 1) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO jobs (job_id, user_id, status, catchments, submitted_at, slurm_id, output_dir, n_tasks)
               VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)""",
            (job_id, user_id, json.dumps(catchment_ids), _now(), slurm_id, output_dir, n_tasks),
        )
        await db.commit()


async def get_job(job_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,)) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            d = dict(row)
            d['catchments'] = json.loads(d['catchments'])
            return d


async def get_user_jobs(user_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            'SELECT * FROM jobs WHERE user_id = ? ORDER BY submitted_at DESC', (user_id,)
        ) as cur:
            rows = await cur.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['catchments'] = json.loads(d['catchments'])
                result.append(d)
            return result


async def count_active_jobs(user_id: str) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            "SELECT COUNT(*) FROM jobs WHERE user_id = ? AND status IN ('queued', 'running')",
            (user_id,),
        ) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0


async def update_status(job_id: str, status: str, error_msg: str | None = None,
                        output_dir: str | None = None) -> None:
    finished_at = _now() if status in ('done', 'failed', 'cancelled') else None
    # Use existing finished_at if already set (avoid overwriting with None)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE jobs SET status = ?,
               finished_at = COALESCE(finished_at, ?),
               error_msg = COALESCE(?, error_msg),
               output_dir = COALESCE(?, output_dir)
               WHERE job_id = ?""",
            (status, finished_at, error_msg, output_dir, job_id),
        )
        await db.commit()


async def all_jobs(limit: int = 100) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            'SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?', (limit,)
        ) as cur:
            rows = await cur.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d['catchments'] = json.loads(d['catchments'])
                result.append(d)
            return result
