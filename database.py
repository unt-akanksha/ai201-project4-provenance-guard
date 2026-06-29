"""
database.py — SQLite setup and all database operations for Provenance Guard.
"""

import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "provenance.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS submissions (
                content_id    TEXT PRIMARY KEY,
                creator_id    TEXT NOT NULL,
                text_snippet  TEXT NOT NULL,
                result        TEXT NOT NULL,
                confidence    REAL NOT NULL,
                llm_score     REAL NOT NULL,
                stylo_score   REAL NOT NULL DEFAULT 0.0,
                burst_score   REAL NOT NULL DEFAULT 0.0,
                label         TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'classified',
                short_text_warning INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_type    TEXT NOT NULL,
                content_id    TEXT NOT NULL,
                payload       TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS appeals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id    TEXT NOT NULL UNIQUE,
                reason        TEXT NOT NULL,
                created_at    TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS verified_creators (
                creator_id    TEXT PRIMARY KEY,
                verified_at   TEXT NOT NULL
            );
        """)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def save_submission(content_id, creator_id, text, result, confidence,
                    llm_score, stylo_score, burst_score, label, short_text_warning):
    snippet = text[:200] + ("…" if len(text) > 200 else "")
    ts = now_iso()
    payload = {
        "entry_type": "decision",
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": ts,
        "result": result,
        "confidence": round(confidence, 4),
        "llm_score": round(llm_score, 4),
        "stylo_score": round(stylo_score, 4),
        "burst_score": round(burst_score, 4),
        "label": label,
        "status": "classified",
        "short_text_warning": short_text_warning,
    }
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO submissions
               (content_id, creator_id, text_snippet, result, confidence,
                llm_score, stylo_score, burst_score, label, status, short_text_warning, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (content_id, creator_id, snippet, result, confidence,
             llm_score, stylo_score, burst_score, label, "classified", int(short_text_warning), ts),
        )
        conn.execute(
            "INSERT INTO audit_log (entry_type, content_id, payload, created_at) VALUES (?,?,?,?)",
            ("decision", content_id, json.dumps(payload), ts),
        )
    return payload


def get_submission(content_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def update_status(content_id, status):
    with get_conn() as conn:
        conn.execute(
            "UPDATE submissions SET status = ? WHERE content_id = ?",
            (status, content_id),
        )


def save_appeal(content_id, reason, original):
    ts = now_iso()
    payload = {
        "entry_type": "appeal",
        "content_id": content_id,
        "appeal_reason": reason,
        "original_result": original["result"],
        "original_confidence": original["confidence"],
        "llm_score": original["llm_score"],
        "timestamp": ts,
    }
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO appeals (content_id, reason, created_at) VALUES (?,?,?)",
            (content_id, reason, ts),
        )
        conn.execute(
            "INSERT INTO audit_log (entry_type, content_id, payload, created_at) VALUES (?,?,?,?)",
            ("appeal", content_id, json.dumps(payload), ts),
        )
    return payload


def appeal_exists(content_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM appeals WHERE content_id = ?", (content_id,)
        ).fetchone()
    return row is not None


def get_log(limit=50, content_id=None, entry_type=None):
    query = "SELECT payload FROM audit_log WHERE 1=1"
    params = []
    if content_id:
        query += " AND content_id = ?"
        params.append(content_id)
    if entry_type:
        query += " AND entry_type = ?"
        params.append(entry_type)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def get_analytics():
    """Return aggregated detection statistics for the analytics dashboard."""
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        breakdown_rows = conn.execute(
            "SELECT result, COUNT(*) as cnt FROM submissions GROUP BY result"
        ).fetchall()
        appeal_count = conn.execute("SELECT COUNT(*) FROM appeals").fetchone()[0]
        avg_conf_row = conn.execute("SELECT AVG(confidence) FROM submissions").fetchone()
        short_count = conn.execute(
            "SELECT COUNT(*) FROM submissions WHERE short_text_warning = 1"
        ).fetchone()[0]
        verified_count = conn.execute(
            "SELECT COUNT(*) FROM verified_creators"
        ).fetchone()[0]

    breakdown = {"ai": 0, "uncertain": 0, "human": 0}
    for row in breakdown_rows:
        breakdown[row["result"]] = row["cnt"]

    avg_confidence = round(avg_conf_row[0], 4) if avg_conf_row[0] is not None else 0.0
    appeal_rate = round(appeal_count / total, 4) if total > 0 else 0.0
    short_text_rate = round(short_count / total, 4) if total > 0 else 0.0

    def pct(n):
        return round(n / total * 100, 1) if total > 0 else 0.0

    return {
        "total_submissions": total,
        "attribution_breakdown": {
            "ai":        {"count": breakdown["ai"],        "pct": pct(breakdown["ai"])},
            "uncertain": {"count": breakdown["uncertain"], "pct": pct(breakdown["uncertain"])},
            "human":     {"count": breakdown["human"],     "pct": pct(breakdown["human"])},
        },
        "appeal_count": appeal_count,
        "appeal_rate": appeal_rate,
        "avg_confidence": avg_confidence,
        "short_text_rate": short_text_rate,
        "verified_creators": verified_count,
    }


def is_verified(creator_id: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT creator_id FROM verified_creators WHERE creator_id = ?", (creator_id,)
        ).fetchone()
    return row is not None


def set_verified(creator_id: str):
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO verified_creators (creator_id, verified_at) VALUES (?,?)",
            (creator_id, ts),
        )
