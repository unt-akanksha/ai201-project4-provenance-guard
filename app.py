"""
app.py — Provenance Guard Flask application.

Endpoints:
  POST /submit              — Submit text for attribution analysis
  POST /appeal              — Contest a classification
  GET  /log                 — Retrieve audit log entries
  GET  /status/<content_id> — Get current status of a submission
"""

import uuid
import os

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

from database import init_db, save_submission, get_submission, update_status, save_appeal, appeal_exists, get_log
from signals import llm_signal, stylometric_signal, combine
from labels import get_label

load_dotenv()

app = Flask(__name__)

# ─────────────────────────────────────────────
# Rate limiting
# ─────────────────────────────────────────────
# Limits (justified in README):
#   - 10 submissions per minute per IP  (protects Groq quota, prevents flooding)
#   - 100 submissions per hour per IP   (allows heavy legitimate use)
#   - 5 appeals per hour per IP         (appeals are manual — no need for high throughput)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# ─────────────────────────────────────────────
# Initialise DB on startup
# ─────────────────────────────────────────────

with app.app_context():
    init_db()


# ─────────────────────────────────────────────
# POST /submit
# ─────────────────────────────────────────────

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute; 100 per hour")
def submit():
    """
    Accept a text submission, run both detection signals, return attribution result.

    Required JSON fields:
      text        (str) — the content to analyse
      creator_id  (str) — identifier for the submitting creator

    Optional:
      content_id  (str) — caller-supplied ID; generated if omitted
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    text = body.get("text", "").strip()
    creator_id = body.get("creator_id", "").strip()
    content_id = body.get("content_id") or str(uuid.uuid4())

    if not text:
        return jsonify({"error": "'text' field is required and must not be empty"}), 400
    if not creator_id:
        return jsonify({"error": "'creator_id' field is required"}), 400
    if len(text) > 50_000:
        return jsonify({"error": "'text' must be 50,000 characters or fewer"}), 400

    # ── Run detection pipeline ──
    score_llm = llm_signal(text)
    stylo_result = stylometric_signal(text)
    combined = combine(score_llm, stylo_result)

    result = combined["result"]
    confidence = combined["confidence"]
    label = get_label(result)

    # ── Persist to audit log ──
    save_submission(
        content_id=content_id,
        creator_id=creator_id,
        text=text,
        result=result,
        confidence=confidence,
        llm_score=combined["score_llm"],
        label=label,
        short_text_warning=combined["short_text_warning"],
    )

    return jsonify({
        "content_id": content_id,
        "result": result,
        "confidence": confidence,
        "label": label,
        "signals": {
            "llm": combined["score_llm"],
            "stylometric": combined["score_stylo"],
            "sub_signals": combined["sub_signals"],
        },
        "short_text_warning": combined["short_text_warning"],
    }), 200


# ─────────────────────────────────────────────
# POST /appeal
# ─────────────────────────────────────────────

@app.route("/appeal", methods=["POST"])
@limiter.limit("5 per hour")
def appeal():
    """
    Contest a classification.

    Required JSON fields:
      content_id  (str) — the ID from a previous /submit response
      reason      (str) — creator's explanation (min 10 characters)
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    content_id = body.get("content_id", "").strip()
    reason = body.get("reason", "").strip()

    if not content_id:
        return jsonify({"error": "'content_id' is required"}), 400
    if not reason or len(reason) < 10:
        return jsonify({"error": "'reason' must be at least 10 characters"}), 400

    submission = get_submission(content_id)
    if not submission:
        return jsonify({"error": f"No submission found for content_id '{content_id}'"}), 404

    if appeal_exists(content_id):
        return jsonify({"error": "An appeal has already been submitted for this content_id"}), 409

    update_status(content_id, "under_review")
    save_appeal(content_id, reason, submission)

    return jsonify({
        "status": "under_review",
        "content_id": content_id,
        "message": "Your appeal has been received. A human reviewer will examine your content.",
    }), 200


# ─────────────────────────────────────────────
# GET /log
# ─────────────────────────────────────────────

@app.route("/log", methods=["GET"])
def log():
    """
    Return recent audit log entries.

    Query params (all optional):
      limit       (int)  — max entries to return (default 50, max 200)
      content_id  (str)  — filter to a single submission
      type        (str)  — filter by entry type: "decision" or "appeal"
    """
    try:
        limit = min(int(request.args.get("limit", 50)), 200)
    except ValueError:
        limit = 50

    content_id = request.args.get("content_id") or None
    entry_type = request.args.get("type") or None

    if entry_type and entry_type not in ("decision", "appeal"):
        return jsonify({"error": "'type' must be 'decision' or 'appeal'"}), 400

    entries = get_log(limit=limit, content_id=content_id, entry_type=entry_type)
    return jsonify({"count": len(entries), "entries": entries}), 200


# ─────────────────────────────────────────────
# GET /status/<content_id>
# ─────────────────────────────────────────────

@app.route("/status/<content_id>", methods=["GET"])
def status(content_id):
    """Return the current status and label for a submission."""
    submission = get_submission(content_id)
    if not submission:
        return jsonify({"error": f"No submission found for content_id '{content_id}'"}), 404

    return jsonify({
        "content_id": content_id,
        "result": submission["result"],
        "confidence": submission["confidence"],
        "label": submission["label"],
        "status": submission["status"],
        "short_text_warning": bool(submission["short_text_warning"]),
        "created_at": submission["created_at"],
    }), 200


# ─────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────

@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
    }), 429


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
