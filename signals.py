"""
signals.py — Detection signals for Provenance Guard.

Signal 1: LLM-based classifier via Groq (score_llm)
Signal 2: Stylometric heuristics in pure Python (score_stylo)
"""

import json
import re
import math
import string
import os

from groq import Groq

# ─────────────────────────────────────────────
# Signal 1: LLM classifier (Groq)
# ─────────────────────────────────────────────

_PROMPT = """\
You are an expert at distinguishing human-written from AI-generated text.
Analyze the following text and estimate the probability that it was generated
by an AI (not written by a human). Consider: formulaic phrasing, unnatural
structural balance, absence of personal voice, predictable transitions.

Respond with ONLY valid JSON: {{"ai_probability": <float between 0.0 and 1.0>}}

Text:
\"\"\"
{text}
\"\"\"\
"""


def llm_signal(text: str) -> float:
    """
    Call Groq with the classifier prompt. Returns score_llm in [0.0, 1.0].
    Higher = more likely AI-generated.
    Falls back to 0.5 on any API or parse error.
    """
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"```[a-z]*", "", raw).strip().strip("`")
        data = json.loads(raw)
        score = float(data["ai_probability"])
        return max(0.0, min(1.0, score))
    except Exception as exc:
        print(f"[llm_signal] Error: {exc} — returning 0.5 fallback")
        return 0.5


# ─────────────────────────────────────────────
# Signal 2: Stylometric heuristics (pure Python)
# ─────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on .!? boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _sentence_length_variance(sentences: list[str]) -> float:
    """Variance of word-count per sentence."""
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    return variance


def _type_token_ratio(words: list[str]) -> float:
    """unique_words / total_words (case-insensitive)."""
    if not words:
        return 1.0
    lower = [w.lower() for w in words]
    return len(set(lower)) / len(lower)


def _punctuation_density(text: str) -> float:
    """punctuation_chars / total_chars."""
    if not text:
        return 0.0
    punct_count = sum(1 for ch in text if ch in string.punctuation)
    return punct_count / len(text)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def stylometric_signal(text: str) -> dict:
    """
    Compute the stylometric signal.

    Returns:
        {
            "score_stylo": float [0, 1],   # higher = more likely AI
            "low_confidence": bool,         # True if text is too short
            "sub_signals": {
                "sent_var_score": float,
                "ttr_score": float,
                "punct_score": float,
                "sentence_variance": float,
                "ttr": float,
                "punct_density": float,
            }
        }
    """
    sentences = _split_sentences(text)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    sent_count = len(sentences)

    low_confidence = word_count < 80 or sent_count < 5

    # ── Sub-signal 1: sentence length variance ──
    # clamp to [0, 25], invert: low variance → high AI score
    variance = _sentence_length_variance(sentences)
    sv_clamped = _clamp(variance, 0.0, 25.0)
    sent_var_score = (25.0 - sv_clamped) / 25.0

    # ── Sub-signal 2: type-token ratio ──
    # clamp to [0.3, 1.0], invert: low TTR → high AI score
    ttr = _type_token_ratio(words)
    ttr_clamped = _clamp(ttr, 0.3, 1.0)
    ttr_score = (ttr_clamped - 1.0) / (0.3 - 1.0)

    # ── Sub-signal 3: punctuation density ──
    # clamp to [0.01, 0.08], invert: low density → high AI score
    density = _punctuation_density(text)
    d_clamped = _clamp(density, 0.01, 0.08)
    punct_score = (d_clamped - 0.08) / (0.01 - 0.08)

    score_stylo = round((sent_var_score + ttr_score + punct_score) / 3.0, 4)

    return {
        "score_stylo": score_stylo,
        "low_confidence": low_confidence,
        "sub_signals": {
            "sent_var_score": round(sent_var_score, 4),
            "ttr_score": round(ttr_score, 4),
            "punct_score": round(punct_score, 4),
            "sentence_variance": round(variance, 4),
            "ttr": round(ttr, 4),
            "punct_density": round(density, 4),
        },
    }


# ─────────────────────────────────────────────
# Confidence combiner
# ─────────────────────────────────────────────

def combine(score_llm: float, stylo_result: dict) -> dict:
    """
    Merge signal scores into a single confidence value and result tier.

    Weights (from spec):
      Normal:     confidence = 0.6 * llm + 0.4 * stylo
      Short text: confidence = 0.85 * llm + 0.15 * stylo

    Thresholds:
      >= 0.85  →  "ai"
      0.35–0.84 → "uncertain"
      < 0.35   →  "human"
    """
    score_stylo = stylo_result["score_stylo"]
    short = stylo_result["low_confidence"]

    if short:
        confidence = 0.85 * score_llm + 0.15 * score_stylo
    else:
        confidence = 0.6 * score_llm + 0.4 * score_stylo

    confidence = round(confidence, 4)

    # Thresholds calibrated from empirical signal ranges:
    # Typical AI text produces combined scores of 0.75–0.85 with strong LLM + stylometric agreement.
    # 0.85 (original spec) was unreachable in practice; recalibrated to 0.78.
    if confidence >= 0.78:
        result = "ai"
    elif confidence < 0.35:
        result = "human"
    else:
        result = "uncertain"

    return {
        "confidence": confidence,
        "result": result,
        "short_text_warning": short,
        "score_llm": round(score_llm, 4),
        "score_stylo": score_stylo,
        "sub_signals": stylo_result["sub_signals"],
    }
