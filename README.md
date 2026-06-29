# Provenance Guard

A backend system any creative sharing platform can plug into to classify submitted text content, score confidence in that classification, surface a transparency label to users, and handle appeals from creators who believe they've been misclassified.

**Stack:** Flask · Groq (llama-3.3-70b-versatile) · SQLite · Flask-Limiter · pure Python  
**Required features:** 7/7 ✅  **Stretch features:** 3/4 ✅ (ensemble detection, analytics dashboard, provenance certificate)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Detection Signals](#detection-signals)
3. [Confidence Scoring](#confidence-scoring)
4. [Transparency Labels](#transparency-labels)
5. [Appeals Workflow](#appeals-workflow)
6. [Rate Limiting](#rate-limiting)
7. [Audit Log](#audit-log)
8. [Stretch Features](#stretch-features)
9. [Known Limitations](#known-limitations)
10. [Spec Reflection](#spec-reflection)
11. [AI Usage](#ai-usage)
12. [Setup & Running Locally](#setup--running-locally)

---

## Architecture Overview

### How a submission flows through the system

A piece of text enters at `POST /submit` with a `text` and `creator_id`. The request first passes through the **rate limiter** — if the caller has exceeded their per-IP limit, it's rejected with 429 before any work happens.

If it passes, the text enters the **detection pipeline**, which runs all three signals in sequence. Signal 1 (LLM via Groq) and Signal 2 (stylometric heuristics) have always been present; Signal 3 (burstiness) was added as a stretch feature. Each returns a score between 0 and 1. The **confidence combiner** merges them with documented weights into a single `confidence` value, which is mapped to one of three result tiers. The full decision is written to the **SQLite audit log** and returned to the caller.

If a creator disputes a result, `POST /appeal` immediately marks the content `under_review` in the database and appends the appeal reasoning alongside the original decision to the audit log. No automated re-classification occurs — a human reviewer resolves it.

```
① Submission flow

POST /submit ──► Rate limiter ──► ┌────────────────────────────────────┐
                                  │        Detection pipeline          │
                                  │                                    │
                                  │  Signal 1: LLM (Groq)             │
                                  │    score_llm ∈ [0, 1]             │
                                  │                                    │
                                  │  Signal 2: Stylometric (Python)   │
                                  │    score_stylo ∈ [0, 1]           │
                                  │                                    │
                                  │  Signal 3: Burstiness (Python)    │
                                  │    score_burst ∈ [0, 1]           │
                                  │       ↓        ↓        ↓         │
                                  │      Ensemble confidence combiner  │
                                  │  0.55·llm + 0.30·stylo            │
                                  │           + 0.15·burst            │
                                  └──────────────┬─────────────────────┘
                                                 ↓
                                      Transparency label selector
                                   (ai / uncertain / human)
                                                 ↓
                                        Audit log (SQLite)
                                                 │
                                    (response) ◄─┘

② Appeal flow

POST /appeal ──► Validate content_id ──► Update status = "under_review"
                                                  ↓
                                         Audit log (SQLite)
                                    (appeal reason + original decision)
                                                  ↓
                                      Return { status: "under_review" }
```

---

## Detection Signals

### Signal 1 — LLM-Based Classifier (Groq)

**What it measures:** Holistic semantic and stylistic coherence. The model (`llama-3.3-70b-versatile`) reads the full text and estimates whether it reads as human- or AI-authored — capturing formulaic phrasing, unnaturally balanced sentence structure, template-like transitions, and semantic predictability.

**Output:** A single float `score_llm ∈ [0.0, 1.0]`. The model is prompted to respond with only `{"ai_probability": <float>}`, so parsing is deterministic. Temperature is set to 0.0 for consistency. On any API or parse error the function falls back to `0.5` (uncertain) rather than crashing the pipeline.

**Why I chose it:** LLMs are uniquely positioned to detect other LLMs because they share the same vocabulary distribution. This gives the signal high sensitivity on typical AI prose. It also catches things pure statistics can't — semantic predictability, absence of personal voice, safe generic framing.

**What it misses:** Clean structured human writing (corporate memos, formal essays) can score as AI-like. AI text that has been lightly edited — a few deliberate typos, a colloquial phrase added — can reduce the score significantly. The model also cannot see the intent behind the writing, only its surface.

---

### Signal 2 — Stylometric Heuristics

**What it measures:** Three statistical properties of the text computed without any external libraries:

| Sub-signal | Computed as | AI tendency | Human tendency |
|---|---|---|---|
| Sentence length variance | variance of word-counts per sentence | Low — uniform sentences | High — erratic lengths |
| Type-token ratio (TTR) | unique\_words / total\_words | Lower — repetitive vocabulary | Higher — diverse vocabulary |
| Punctuation density | punct\_chars / total\_chars | Lower — conventional | Higher — expressive |

Each sub-signal is normalized to [0, 1] using empirically-set bounds, then averaged into `score_stylo`. If the text has fewer than 80 words or 5 sentences, `low_confidence` is flagged and the signal's weight drops in the combiner.

**Why I chose it:** Stylometrics are independent of meaning — they operate on raw character and token distributions. This makes them genuinely orthogonal to the LLM signal: one judges "does this feel AI-written?", the other asks "does the statistical fingerprint match AI writing?" A text that fools one signal is unlikely to fool both.

**What it misses:** Formal academic and legal human writing has naturally low sentence variance, high consistent vocabulary, and clean punctuation — all of which look AI-like to this signal. Poetry with deliberate repetition (villanelles, refrains) is a known false-positive case. Both are documented in Known Limitations.

---

### Signal 3 — Burstiness / Vocabulary Clustering *(Stretch: Ensemble Detection)*

**What it measures:** Variance in local type-token ratio computed over a sliding window of 10 words. Human writers naturally cluster unusual vocabulary in bursts — a passage of elevated vocabulary followed by plain prose. AI text tends to distribute vocabulary density uniformly throughout.

**How it's computed:** Window TTR variance is calculated across all 10-word windows in the text, clamped to [0, 0.04], then inverted: `score_burst = (0.04 − variance) / 0.04`. High variance → bursty/human-like → low AI score.

**Why this differs from Signal 2:** TTR in Signal 2 measures the overall vocabulary diversity of the whole text. Burstiness measures how that diversity is *distributed* — whether it clusters or spreads evenly. Two texts can have identical TTRs but opposite burstiness profiles.

**What it misses:** Texts under 80 words don't produce enough windows for reliable variance measurement. The signal is flagged `low_confidence` for short texts and weighted at only 5% in the combiner.

---

## Confidence Scoring

### Design philosophy

The confidence score is a design decision before it is a technical one. **A false positive — labeling a human's work as AI-generated — is worse than a false negative on a writing platform.** This asymmetry shapes every threshold: the "ai" tier requires ≥ 0.78, not just above 0.5. The uncertain band is intentionally wide (0.35–0.77) so the system defaults to "we don't know" rather than making an accusation.

A score of 0.50 means: the signals disagree or are weakly consistent — we genuinely don't know. It is not "slightly more AI than human."

### Ensemble weights

```
Normal text (≥ 80 words, ≥ 5 sentences):
  confidence = 0.55 × score_llm + 0.30 × score_stylo + 0.15 × score_burst

Short text (< 80 words or < 5 sentences):
  confidence = 0.85 × score_llm + 0.10 × score_stylo + 0.05 × score_burst
```

LLM carries the highest weight because it captures meaning, not just structure, and is the most discriminative signal across text types. Stylometric gets 30% as an independent structural check. Burstiness gets 15% — genuine signal on longer texts, too noisy on short ones. All non-LLM weights drop toward zero on short text because both statistical signals need sufficient word count to be meaningful.

### Threshold mapping

| Confidence | Result | Label variant |
|---|---|---|
| ≥ 0.78 | `"ai"` | Attribution: AI-generated |
| 0.35 – 0.77 | `"uncertain"` | Attribution: Uncertain |
| < 0.35 | `"human"` | Attribution: Human-written |

The original spec set the AI threshold at 0.85. During Milestone 4 testing this proved unreachable — typical clearly-AI text was scoring 0.75–0.82. The threshold was recalibrated to 0.78 to match the actual signal distribution. See Spec Reflection for details.

### Example submissions

**Example 1 — Clearly AI-generated** (input: `ai_clear.json`)

> "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications..."

| Signal | Score |
|---|---|
| LLM classifier | 0.95 |
| Stylometric | 0.82 |
| Burstiness | 0.88 |
| **Combined confidence** | **0.90** |

Result: `ai` → Attribution: AI-generated label.

---

**Example 2 — Borderline formal human writing** (input: `borderline_formal.json`)

> "The relationship between monetary policy and asset price inflation has been extensively studied in the literature. Central banks face a fundamental tension between their mandate for price stability..."

| Signal | Score |
|---|---|
| LLM classifier | 0.58 |
| Stylometric | 0.61 |
| Burstiness | 0.34 |
| **Combined confidence** | **0.54** |

Result: `uncertain` → Attribution: Uncertain label. The burstiness signal pulled the score down because this text shows human-like vocabulary clustering even though the other signals found it somewhat AI-like. This is the honest answer — formal academic writing sits in the uncertain zone by design.

---

## Transparency Labels

All label text is returned in the `"label"` field of the API response. The confidence score as a decimal is not shown to end users — only the label text.

### Variant 1 — Attribution: AI-generated (`confidence ≥ 0.78`)

```
Attribution: AI-generated

Our detection system found strong signals that this content was generated
by an AI writing tool. This label reflects pattern-based analysis — it is
not a definitive ruling.

If you wrote this yourself, you can dispute this classification using the
appeal button below.
```

### Variant 2 — Attribution: Uncertain (`confidence 0.35 – 0.77`)

```
Attribution: Uncertain

Our system could not confidently determine whether this content was written
by a human or an AI tool. This is not an accusation — attribution is
genuinely difficult, and this label simply means we don't know.

If you feel this label is wrong, you can submit an appeal to have a human
reviewer look at your content.
```

### Variant 3 — Attribution: Human-written (`confidence < 0.35`)

```
Attribution: Human-written

Our detection system found strong signals that this content was written by
a human author. This label reflects pattern-based analysis and is not a
guarantee.
```

### Variant 4 — Attribution: Verified Human ✓ *(Stretch: Provenance Certificate)*

```
Attribution: Verified Human ✓

This creator has completed Provenance Guard's human verification step.
They submitted a live writing sample that scored below the AI threshold,
and their identity has been confirmed under their verified account.

Verification reduces the likelihood of misclassification but does not
guarantee all content from this creator is human-written.
```

**Design notes:**
- All variants use plain language — no "confidence score", no "signal", no technical jargon.
- Variants 1 and 2 include an explicit path to appeal; Variant 3 does not (no false accusation to dispute).
- Variant 1 includes "not a definitive ruling" to soften the accusation.
- Variant 2 explicitly says "this is not an accusation" — the most important phrase in the entire label design.

---

## Appeals Workflow

### Who can submit

Any submitter with a `content_id` from a previous `/submit` response. The `content_id` acts as the access token — no separate authentication in this implementation.

### Request format

```json
{
  "content_id": "7eb001f5-3dce-4897-b591-b26fde6e575c",
  "reason": "I wrote this myself as a summary of my research. Please review my classification."
}
```

`reason` is required and must be at least 10 characters. Empty or trivial appeals are rejected with 400.

### What the system does

1. Validates `content_id` exists — returns 404 if not found.
2. Checks for a prior appeal on the same `content_id` — returns 409 Conflict if one exists (one appeal per submission).
3. Updates `status` from `"classified"` to `"under_review"` in the submissions table.
4. Writes a new `"appeal"` entry to the audit log containing: `content_id`, `appeal_reason`, `original_result`, `original_confidence`, `llm_score`, and `timestamp`.
5. Returns:

```json
{
  "status": "under_review",
  "content_id": "7eb001f5-...",
  "message": "Your appeal has been received. A human reviewer will examine your content."
}
```

### What a reviewer sees

`GET /log?type=appeal` returns all appeal entries:

```json
{
  "entry_type": "appeal",
  "content_id": "7eb001f5-...",
  "appeal_reason": "I wrote this myself as a summary of my research.",
  "original_result": "uncertain",
  "original_confidence": 0.54,
  "llm_score": 0.58,
  "timestamp": "2026-06-28T15:22:44.001Z"
}
```

---

## Rate Limiting

**Limits:**
- `POST /submit`: **10 per minute, 100 per hour** per IP
- `POST /appeal`: **5 per hour** per IP
- `POST /verify`: **3 per hour** per IP *(stretch feature)*

**Reasoning:**

*10 per minute on /submit:* A real writer submitting their own work might post two or three pieces in a sitting, but hitting 10 in 60 seconds is already aggressive legitimate use. This comfortably handles a creator batch-submitting a short story collection chapter by chapter, while stopping a scripted flood. The Groq API has its own rate limits — staying at 10/min also prevents burning through Groq quota on abuse traffic.

*100 per hour on /submit:* 100 is generous enough to never inconvenience any legitimate user, but low enough to prevent systematic scraping or enumeration attacks across a session.

*5 per hour on /appeal:* Appeals are a manual-review workflow. High throughput appeals would overwhelm a reviewer queue — 5 per hour is more than enough for any single legitimate creator while preventing appeal-flooding as a harassment vector.

**Rate limit in action** (12 rapid POST /submit requests):

```
200
200
200
200
200
200
200
200
200
200
429
429
```

The first 10 return 200 OK; requests 11 and 12 return 429 Too Many Requests.

**Flask-Limiter configuration:**

```python
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute; 100 per hour")
def submit(): ...

@app.route("/appeal", methods=["POST"])
@limiter.limit("5 per hour")
def appeal(): ...
```

---

## Audit Log

Every attribution decision — including all three signal scores, combined confidence, label, and any appeal — is written to a structured SQLite audit log. `GET /log` returns recent entries as JSON.

**Query params:** `?limit=N` (default 50, max 200) · `?content_id=X` · `?type=decision|appeal`

### Sample log output (`GET /log`)

```json
{
  "count": 3,
  "entries": [
    {
      "entry_type": "decision",
      "content_id": "3f7a2b1e-9c4d-4a1f-b823-d7e2f1a0c5b9",
      "creator_id": "m4-test",
      "timestamp": "2026-06-28T14:32:10.123Z",
      "result": "ai",
      "confidence": 0.90,
      "llm_score": 0.95,
      "stylo_score": 0.82,
      "burst_score": 0.88,
      "label": "Attribution: AI-generated\n\nOur detection system found strong signals...",
      "status": "classified",
      "short_text_warning": false
    },
    {
      "entry_type": "appeal",
      "content_id": "7eb001f5-3dce-4897-b591-b26fde6e575c",
      "appeal_reason": "I wrote this myself as a summary of my research. Please review my classification.",
      "original_result": "uncertain",
      "original_confidence": 0.54,
      "llm_score": 0.58,
      "timestamp": "2026-06-28T15:22:44.001Z"
    },
    {
      "entry_type": "decision",
      "content_id": "f9e8d7c6-b5a4-3210-fedc-ba9876543210",
      "creator_id": "test-user-1",
      "timestamp": "2026-06-28T15:02:44.887Z",
      "result": "human",
      "confidence": 0.18,
      "llm_score": 0.12,
      "stylo_score": 0.28,
      "burst_score": 0.09,
      "label": "Attribution: Human-written\n\nOur detection system found strong signals...",
      "status": "classified",
      "short_text_warning": true
    }
  ]
}
```

---

## Stretch Features

### ✅ Ensemble Detection (3 signals)

Added **Signal 3: Burstiness / vocabulary clustering** in `signals.py`. See [Detection Signals](#detection-signals) for the full description.

Updated the combiner from a 2-signal weighted average to a 3-signal ensemble with documented weights (55% LLM / 30% stylometric / 15% burstiness for normal text; 85% / 10% / 5% for short text). The weights are explained and justified in `planning.md` and the [Confidence Scoring](#confidence-scoring) section above.

---

### ✅ Analytics Dashboard

**Endpoint:** `GET /analytics`

Returns aggregated detection statistics from the SQLite database with no external dependencies — pure SQL aggregation queries.

```json
{
  "total_submissions": 12,
  "attribution_breakdown": {
    "ai":        { "count": 5, "pct": 41.7 },
    "uncertain": { "count": 4, "pct": 33.3 },
    "human":     { "count": 3, "pct": 25.0 }
  },
  "appeal_count": 1,
  "appeal_rate": 0.0833,
  "avg_confidence": 0.5812,
  "short_text_rate": 0.25,
  "verified_creators": 0
}
```

The **appeal rate** is the additional metric beyond the two required ones. A rising appeal rate is the most actionable early signal that detection quality is degrading — it tells a platform operator that creators are being misclassified before the misclassification count grows large enough to notice otherwise.

---

### ✅ Provenance Certificate

**Endpoint:** `POST /verify`

A creator submits a live writing sample (minimum 150 words). The full 3-signal pipeline runs on the sample. If `confidence < 0.40` — both the LLM and stylometric signals lean human — the `creator_id` is stored in a `verified_creators` table in SQLite. All future submissions from that creator display the Verified Human label (Variant 4 above) instead of the standard detection result.

**Request:**
```json
{
  "creator_id": "user-042",
  "sample": "... at least 150 words of live writing ..."
}
```

**Response (passed):**
```json
{
  "creator_id": "user-042",
  "verified": true,
  "confidence": 0.22,
  "message": "Verification passed. Your content will now display a Verified Human badge.",
  "signals": { "llm": 0.18, "stylometric": 0.31, "burstiness": 0.14 }
}
```

**Response (failed):**
```json
{
  "creator_id": "user-042",
  "verified": false,
  "confidence": 0.61,
  "message": "Verification did not pass. The sample scored above the human threshold (confidence: 0.61). You may rewrite and try again.",
  "signals": { "llm": 0.67, "stylometric": 0.52, "burstiness": 0.58 }
}
```

**Constraints and design decisions:**
- Minimum 150 words enforced — too short and the pipeline can't make a reliable judgment
- Rate limited to 3 attempts per hour to prevent brute-forcing with different samples
- A creator already verified gets a clean 200 (`"verified": true`) rather than an error
- One appeal per `content_id` enforced via UNIQUE constraint in the appeals table
- Known limitation: `creator_id` is caller-supplied in this implementation; a production system would authenticate the session before granting a verified credential

---

## Known Limitations

### 1 — Formal human writing triggers stylometric false positives

Formal academic, legal, and technical writing by humans has naturally low sentence-length variance, high but consistent vocabulary, and clean punctuation — exactly the statistical fingerprint the stylometric signal associates with AI. A PhD thesis or legal brief can produce `score_stylo` values of 0.65–0.75, which pushes the combined confidence into the uncertain zone even when the LLM signal leans human.

This is a property of the signal's design, not a data problem — any stylometric approach that penalizes uniformity will disadvantage disciplined formal writers. The burstiness signal partially mitigates this (formal human writing tends to show bursty vocabulary clustering), but cannot fully compensate. The uncertain label is the honest output for this class of content.

### 2 — Lightly edited AI output lands in the uncertain band

A creator who generates text with an AI and then edits it — changing a few words, introducing typos, adding personal anecdotes — will often score in the uncertain band rather than the AI band. Both signals measure the final text, not the generative process. If the editing was thoughtful, the system will label the result "uncertain," which is technically accurate but may frustrate readers expecting it to catch assisted content.

This is a fundamental limitation of any text-based detection approach. Without provenance metadata (e.g., generation-time watermarking), distinguishing "AI-generated and edited" from "human-written in a polished style" is not reliably solvable at the text level.

---

## Spec Reflection

### Where the spec helped

Writing the three transparency label variants verbatim before any implementation was the most valuable spec decision. When I reached Milestone 5 and wrote `get_label()`, the exact text was already decided — I just encoded it. Without that prior decision I would have written the text inline in the route handler and rationalized it post-hoc. The spec forced the UX decision to happen before the technical one, which is the right order.

The false positive scenario trace in Milestone 1 also shaped the threshold design concretely. Tracing the scenario — "a poet submits a villanelle; stylometrics scores it AI-like; what does the user see?" — before writing any code led directly to the wide uncertain band (0.35–0.77) and the explicit "this is not an accusation" language in the uncertain label.

### Where implementation diverged from the spec

The original spec set the AI threshold at ≥ 0.85. During Milestone 4 testing with the four provided sample inputs, clearly AI-generated text was consistently scoring 0.75–0.82 under the original 2-signal 0.6/0.4 formula — the 0.85 threshold was effectively unreachable in practice. I recalibrated to 0.78 after running the test suite and confirming that this threshold correctly classifies the clearly AI inputs as `"ai"` while keeping borderline inputs in `"uncertain"`.

The spec also assumed a simple 2-signal weighted average throughout. The ensemble approach (3 signals, separate short-text weights) emerged from realizing during testing that the stylometric signal degrades significantly on short texts, while the LLM signal doesn't — a single combined weight for both cases was producing misleading results on short submissions.

---

## AI Usage

### Instance 1 — Flask skeleton and Signal 1

**Directed:** Provided the detection signals section and architecture diagram from `planning.md`, asked for (1) a Flask app skeleton with a `POST /submit` route stub and SQLite audit log setup, and (2) the LLM signal function using the exact prompt template from the spec.

**Produced:** A working skeleton and a `classify_with_groq()` function. The function correctly used `temperature=0.0` and the JSON-only prompt. However, it called `json.loads()` directly on the model response with no error handling — a single preamble word before the JSON object would crash the pipeline.

**Revised:** Added a try/except around JSON parsing and a pre-processing step that strips everything before the first `{`. Also added the fallback to `0.5` on any exception, so a Groq API error degrades gracefully rather than taking down the endpoint. Renamed the function to `llm_signal()` to match the naming convention in the spec.

### Instance 2 — Stylometric signal and confidence combiner

**Directed:** Provided the Signal 2 section (all three sub-signals with normalization formulas) and the confidence combiner section (both weighted formulas, threshold table) from `planning.md`. Asked for the stylometric function and the combiner.

**Produced:** Both functions were structurally correct. The stylometric normalization matched the spec. The combiner function, however, used equal 0.5/0.5 weights rather than the 0.6/0.4 split specified — the model implemented "reasonable-looking" weights rather than reading the spec carefully.

**Revised:** Corrected the weights to 0.6/0.4 to match the spec, then updated them again to 0.55/0.30/0.15 when Signal 3 was added. The short-text fallback (0.85/0.15) was also missing from the generated code — the AI had read the combiner formula but skipped the short-text adjustment documented two paragraphs above it. Added that manually.

---

## Setup & Running Locally

```bash
git clone https://github.com/<your-username>/ai201-project4-provenance-guard
cd ai201-project4-provenance-guard
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Create .env (never commit this)
echo "GROQ_API_KEY=your_key_here" > .env

python app.py
```

### API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/submit` | Submit text for attribution analysis |
| `POST` | `/appeal` | Contest a classification |
| `GET` | `/log` | View audit log (`?limit`, `?content_id`, `?type`) |
| `GET` | `/status/<content_id>` | Current status of a submission |
| `GET` | `/analytics` | Detection statistics dashboard |
| `POST` | `/verify` | Submit sample for provenance certificate |

### Example curl commands

```bash
# Submit content
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d @ai_clear.json

# Submit an appeal
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d @appeal1.json

# View audit log (all entries)
curl -s http://localhost:5000/log

# View only appeal entries
curl -s "http://localhost:5000/log?type=appeal"

# Analytics dashboard
curl -s http://localhost:5000/analytics

# Submit for provenance certificate
curl -s -X POST http://localhost:5000/verify \
  -H "Content-Type: application/json" \
  -d '{"creator_id": "user-001", "sample": "... 150+ word writing sample ..."}'
```
