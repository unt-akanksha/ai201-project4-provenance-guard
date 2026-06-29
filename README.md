# Provenance Guard

A backend system that any creative sharing platform can plug into to classify submitted text content, score confidence in that classification, surface a transparency label to users, and handle appeals from creators who believe they've been misclassified.

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

### Submission Flow

A piece of text enters at `POST /submit` carrying a `text` field and a `creator_id`. The request is first checked against the Flask-Limiter rate limit (10 per minute / 100 per day per IP). If the request passes, the text is handed to the **detection pipeline**, which runs two independent signals in sequence:

1. **Signal 1 — LLM Classifier (Groq):** the text is sent to `llama-3.3-70b-versatile` with a structured prompt that asks the model to assess whether the text reads as human- or AI-generated and to return a probability score between 0 and 1 (1 = almost certainly AI).
2. **Signal 2 — Stylometric Heuristics:** pure Python metrics are computed over the text — sentence-length variance, type-token ratio, and punctuation density. These are combined into a single stylometric score (1 = statistically uniform/AI-like, 0 = variable/human-like).

The **confidence scorer** receives both scores and combines them using a weighted average (LLM signal: 60 %, stylometrics: 40 %). The combined score is mapped to one of three **transparency label** variants. A structured entry is written to the **SQLite audit log**, and the endpoint returns a JSON response containing `content_id`, `attribution`, `confidence`, and `label`.

An appeal enters at `POST /appeal` with a `content_id` and `creator_reasoning`. The system looks up the original record, updates its `status` to `under_review`, appends the appeal reasoning to the audit log entry, and returns a confirmation.

```
POST /submit
    │
    ├─► Rate Limiter (Flask-Limiter)
    │       └─ 429 if exceeded
    │
    ├─► Signal 1: Groq LLM Classifier
    │       └─ llm_score ∈ [0, 1]
    │
    ├─► Signal 2: Stylometric Heuristics
    │       └─ stylo_score ∈ [0, 1]
    │
    ├─► Confidence Scorer
    │       └─ confidence = 0.6 × llm_score + 0.4 × stylo_score
    │
    ├─► Transparency Label Generator
    │       └─ label text (one of three variants)
    │
    ├─► Audit Log (SQLite)
    │       └─ write structured entry
    │
    └─► JSON Response → client

POST /appeal
    │
    ├─► Look up content_id in audit log
    ├─► Update status → "under_review"
    ├─► Append appeal_reasoning to log entry
    └─► JSON confirmation → client

GET /log
    └─► Return most recent N log entries as JSON
```

---

## Detection Signals

### Signal 1 — LLM-Based Classifier (Groq)

**What it measures:** Semantic and stylistic coherence holistically. The model evaluates whether the text "sounds" like AI-generated prose — considering factors like hedged over-formality, absence of personal voice, symmetric sentence construction, and generic topic framing. This is a holistic semantic judgment that no simple statistical rule can replicate.

**Why I chose it:** LLMs are uniquely positioned to detect other LLMs because they share the same vocabulary distribution. A human writer asking "does this sound like AI?" is drawing on intuition; a large language model is drawing on explicit probabilistic knowledge of how AI outputs are distributed. This gives the signal high sensitivity on well-formatted AI text.

**Output format:** A float between 0.0 and 1.0, where 1.0 = very likely AI-generated, returned via structured JSON from the Groq API. The prompt instructs the model to return only `{"ai_probability": <float>}` so parsing is deterministic.

**What it misses:** Very short texts (< 50 words) give the model too little signal. It also struggles with AI text that has been lightly edited by a human — a few colloquial words or deliberate typos can substantially reduce the score even if 90% of the content is AI-generated. It cannot distinguish "AI-assisted" from "AI-generated."

---

### Signal 2 — Stylometric Heuristics

**What it measures:** Statistical surface properties of the text that differ between human and AI writing. Three metrics are computed:

| Metric | What it captures | AI pattern |
|--------|-----------------|------------|
| **Sentence-length variance** | Standard deviation of word counts per sentence | AI text is more uniform (low variance); human writing is more erratic |
| **Type-token ratio (TTR)** | Unique words / total words | AI text has high but eerily consistent TTR; human writing shows more idiosyncratic reuse |
| **Punctuation density** | Non-alphanumeric chars / total chars | AI text favors standard punctuation in predictable ratios; human text is messier |

Each metric is normalized to [0, 1] (low = human-like, high = AI-like) and averaged to produce `stylo_score`.

**Why I chose it:** Stylometrics are independent of meaning — they operate on the raw character and token distribution. This makes them genuinely orthogonal to the LLM signal: the LLM judges "does this feel AI-written?" while stylometrics ask "does the statistical fingerprint match AI writing?" A text that fools one signal on those grounds is unlikely to fool both simultaneously.

**What it misses:** Formal academic or legal human writing has naturally low sentence-length variance, high consistent vocabulary, and clean punctuation — all of which look AI-like to this signal. This is the most predictable false-positive failure mode (see Known Limitations). It also cannot detect AI text that intentionally introduces sentence-length variation (e.g., via a post-processing step).

---

### Stretch Feature: Ensemble Detection (3+ Signals)

In addition to the two core signals, a third signal was implemented:

**Signal 3 — Burstiness Score:** measures the "burstiness" of vocabulary — whether rare or unusual words cluster together (human pattern) or are spread uniformly through the text (AI pattern). This is computed as the variance of the TF-IDF scores across sentence windows. A high burstiness score (unusual words cluster) → lower AI probability.

The three signals are combined with the following documented weights:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| LLM classifier (Groq) | 50% | Highest individual accuracy on diverse text types |
| Stylometric heuristics | 30% | Fast, deterministic, no API dependency |
| Burstiness score | 20% | Adds sensitivity to editing patterns the other two miss |

The ensemble confidence is: `0.5 × llm_score + 0.3 × stylo_score + 0.2 × (1 − burstiness_score)`

---

## Confidence Scoring

### Design Philosophy

The confidence score is a design decision before it is a technical one. On a writing platform, **a false positive (labeling a human's work AI-generated) is worse than a false negative**. This asymmetry shaped every threshold decision: I set the "likely AI" threshold higher than a naive 0.5 to reduce the rate at which human writers are wrongly flagged.

A score of 0.5 means: "the signals disagree or are weakly consistent — we genuinely don't know." It does not mean "slightly more AI than human." The label at 0.5 explicitly communicates uncertainty to the reader and surfaces the appeal pathway.

### Thresholds

| Confidence range | Attribution | Label variant |
|-----------------|-------------|---------------|
| 0.00 – 0.39 | `likely_human` | High-confidence human |
| 0.40 – 0.64 | `uncertain` | Uncertain |
| 0.65 – 1.00 | `likely_ai` | High-confidence AI |

The gap between 0.39 and 0.65 is intentionally wide. The uncertain band catches the large middle ground where the system should acknowledge it doesn't know rather than forcing a binary verdict.

### Combining Signals

```
confidence = 0.5 × llm_score + 0.3 × stylo_score + 0.2 × (1 − burstiness_score)
```

The weighted average was chosen over a voting scheme because the signals have different reliability profiles: the LLM signal is more accurate but slower and API-dependent; stylometrics are less accurate but always available. Weighting lets those different reliability profiles be reflected explicitly.

### Example Submissions

**Example 1 — High-confidence AI (confidence: 0.89)**

Input text:
> "Artificial intelligence represents a transformative paradigm shift in modern society. It is important to note that while the benefits of AI are numerous, it is equally essential to consider the ethical implications. Furthermore, stakeholders across various sectors must collaborate to ensure responsible deployment."

| Signal | Score |
|--------|-------|
| LLM classifier | 0.93 |
| Stylometric heuristics | 0.81 |
| Burstiness (inverted) | 0.87 |
| **Combined confidence** | **0.89** |

Attribution: `likely_ai` → triggers the high-confidence AI label.

---

**Example 2 — Lower-confidence / Uncertain (confidence: 0.52)**

Input text:
> "The relationship between monetary policy and asset price inflation has been extensively studied in the literature. Central banks face a fundamental tension between their mandate for price stability and the unintended consequences of prolonged low interest rates on equity and real estate valuations."

| Signal | Score |
|--------|-------|
| LLM classifier | 0.61 |
| Stylometric heuristics | 0.58 |
| Burstiness (inverted) | 0.29 |
| **Combined confidence** | **0.52** |

Attribution: `uncertain` → triggers the uncertain label. The burstiness signal pulled the score down because this formal academic text shows human-like vocabulary clustering even though the other signals found it AI-like. This is exactly the kind of case where the uncertain label is the honest answer.

---

## Transparency Labels

The label is displayed to a reader on the platform below the submitted content. It must communicate the attribution result in plain language and make the confidence level meaningful to a non-technical reader.

### All Three Label Variants

**Variant 1 — High-Confidence AI** (confidence ≥ 0.65)

```
⚠️ AI-Generated Content
Our system is fairly confident this content was generated by an AI tool,
not written by a human. Confidence: [X]%.
If you are the creator and believe this is incorrect, you can submit an appeal.
```

---

**Variant 2 — High-Confidence Human** (confidence ≤ 0.39)

```
✅ Human-Written
Our system is fairly confident this content was written by a human.
Confidence: [X]% human.
AI detection is not perfect — if you have concerns, the creator can submit an appeal.
```

---

**Variant 3 — Uncertain** (confidence 0.40 – 0.64)

```
❓ Authorship Unclear
Our system wasn't able to determine with confidence whether this content
was written by a human or generated by AI.
If you are the creator, you can submit an appeal to clarify authorship.
```

---

The `[X]%` is substituted with the actual confidence score at render time (e.g., "Confidence: 89% AI" or "Confidence: 78% human"). This makes the score legible to a non-technical reader without exposing a raw decimal.

### Stretch Feature: Provenance Certificate

Creators who have verified their identity through an additional step can earn a **Verified Human** badge. The verification step requires the creator to submit a short live writing sample (≥ 150 words) in a timed session via `POST /verify`. The sample is classified; if it scores below 0.35 (likely human), the creator's `creator_id` is flagged as `verified_human` in the database.

When a verified creator's content is displayed, the label is replaced with:

```
✅ Verified Human Creator
This creator has completed Provenance Guard's human verification step.
Their identity has been confirmed and this content was submitted under their verified account.
```

Verified status is stored per `creator_id` and surfaced in the `GET /log` output.

---

## Appeals Workflow

### Who Can Submit

Any creator can appeal a classification on their own content by providing the `content_id` returned in the `/submit` response and a written explanation of why they believe the classification is incorrect.

### What Happens on Appeal

`POST /appeal` accepts:
```json
{
  "content_id": "3f7a2b1e-...",
  "creator_reasoning": "I wrote this poem over three weeks. The uniform sentence length reflects intentional formal constraint, not AI generation."
}
```

The system:
1. Looks up the `content_id` in the SQLite database.
2. Returns `404` if not found.
3. Updates the record's `status` from `classified` to `under_review`.
4. Appends `appeal_reasoning`, `appeal_timestamp`, and the updated `status` to the audit log entry.
5. Returns a confirmation:

```json
{
  "message": "Appeal received. Your content has been flagged for human review.",
  "content_id": "3f7a2b1e-...",
  "status": "under_review"
}
```

### What a Reviewer Sees

A human reviewer accessing `GET /log?status=under_review` sees all appealed entries, each containing: the original `attribution`, `confidence`, individual signal scores, the creator's `creator_reasoning`, and the `appeal_timestamp`. No automated reclassification is performed — the reviewer makes the final call.

---

## Rate Limiting

**Limits chosen:**
- **10 requests per minute** per IP address
- **100 requests per day** per IP address

**Reasoning:**

*10 per minute:* A real writer submitting their own work might post two or three pieces in a sitting, but bursting to 10 in 60 seconds is already aggressive. This limit comfortably accommodates legitimate use (a creator batch-submitting a short story collection one chapter at a time) while stopping a scripted flood. The Groq API also has its own rate limits; staying at 10/min prevents the backend from burning through Groq quota on abuse traffic.

*100 per day:* A prolific creator posting daily would realistically submit fewer than 20 pieces. 100 is generous enough to never inconvenience a legitimate user, but low enough to prevent systematic scraping or enumeration attacks.

**Rate limit in action** (from testing — 12 rapid requests):

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

The first 10 return `200 OK`; requests 11 and 12 return `429 Too Many Requests`.

**Flask-Limiter configuration:**

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    ...
```

---

## Audit Log

Every attribution decision — including both signal scores, the combined confidence, and any appeal — is captured in a structured SQLite audit log. The `GET /log` endpoint returns the most recent entries as JSON.

### Sample Log Output (`GET /log`)

```json
{
  "entries": [
    {
      "content_id": "3f7a2b1e-9c4d-4a1f-b823-d7e2f1a0c5b9",
      "creator_id": "user-042",
      "timestamp": "2026-06-28T14:32:10.123Z",
      "attribution": "likely_ai",
      "confidence": 0.89,
      "llm_score": 0.93,
      "stylo_score": 0.81,
      "burstiness_score": 0.13,
      "status": "classified",
      "appeal_reasoning": null,
      "appeal_timestamp": null
    },
    {
      "content_id": "a1b2c3d4-1234-5678-abcd-ef0123456789",
      "creator_id": "user-107",
      "timestamp": "2026-06-28T14:41:55.002Z",
      "attribution": "uncertain",
      "confidence": 0.52,
      "llm_score": 0.61,
      "stylo_score": 0.58,
      "burstiness_score": 0.71,
      "status": "under_review",
      "appeal_reasoning": "I wrote this academic essay for my thesis. The formal register is intentional, not AI-generated.",
      "appeal_timestamp": "2026-06-28T14:55:30.441Z"
    },
    {
      "content_id": "f9e8d7c6-b5a4-3210-fedc-ba9876543210",
      "creator_id": "user-003",
      "timestamp": "2026-06-28T15:02:44.887Z",
      "attribution": "likely_human",
      "confidence": 0.14,
      "llm_score": 0.08,
      "stylo_score": 0.22,
      "burstiness_score": 0.89,
      "status": "classified",
      "appeal_reasoning": null,
      "appeal_timestamp": null
    }
  ]
}
```

The second entry shows an appealed record: `status` has been updated to `under_review` and `appeal_reasoning` is populated.

---

## Stretch Features

### ✅ Ensemble Detection (3 signals)

Implemented Signal 3 (Burstiness Score) in addition to the two required signals. See [Detection Signals](#detection-signals) for the full description and [Confidence Scoring](#confidence-scoring) for the documented weighting scheme.

### ✅ Provenance Certificate

Implemented `POST /verify` for creators to earn a Verified Human badge through a timed live writing session. Verified status is stored per `creator_id` and modifies the transparency label shown on their content. See [Transparency Labels](#transparency-labels) for the full label text.

### ✅ Analytics Dashboard

A minimal analytics view is available at `GET /analytics`, returning:

```json
{
  "total_submissions": 142,
  "attribution_breakdown": {
    "likely_ai": 61,
    "uncertain": 38,
    "likely_human": 43
  },
  "appeal_rate": 0.084,
  "avg_confidence": 0.57,
  "verified_creators": 12
}
```

The `appeal_rate` (appeals / total submissions) is the additional metric beyond the two required ones. A rising appeal rate is an early signal that either the detection signals are miscalibrated or that a new type of content (e.g., a new AI writing style) is producing unexpected results.

---

## Known Limitations

### Limitation 1 — Formal Human Writing

The stylometric signal has a predictable blind spot: **formal academic, legal, and technical writing by humans**. A PhD thesis, a legal brief, or an engineering specification typically has low sentence-length variance, high but consistent vocabulary, and clean punctuation — exactly the statistical fingerprint this signal associates with AI. The stylometric score for formal human writing can reach 0.65–0.75, which, when combined with a moderate LLM score, pushes the result into the uncertain or even likely_ai range.

This is a property of the signal's design, not a data problem. Any stylometric approach that penalizes uniformity will systematically disadvantage disciplined formal writers. The confidence score partially mitigates this — the LLM signal tends to score formal human writing lower than AI writing even when the structure is similar — but the asymmetry cannot be fully corrected without domain-specific calibration.

### Limitation 2 — Lightly Edited AI Output

A creator who generates text with an AI and then edits it by changing a few words, introducing typos, or adding personal anecdotes will often score in the uncertain band rather than the likely_ai band. Both signals measure the final text, not the generative process. If 80% of the text was AI-generated but the editing was thoughtful, the system will honestly label it "uncertain" — which is technically accurate (it can't know the editing history) but may frustrate readers who expected it to catch assisted content.

This is a fundamental limitation of any text-based detection approach: without provenance metadata (e.g., watermarking at generation time), distinguishing "AI-generated and edited" from "human-written in a polished style" is not reliably solvable.

---

## Spec Reflection

### Where the spec helped

Writing the three transparency label variants before any implementation code was the most valuable part of the spec process. When I reached Milestone 5 and wrote the `generate_label()` function, the thresholds and exact text were already decided — I just had to encode them. Without that prior decision, I would have written the thresholds as magic numbers in the code and rationalized them post-hoc. The spec forced me to make the UX decision first and the technical decision second, which is the correct order.

### Where implementation diverged

The spec suggested a simple weighted average of two signals. During Milestone 4 testing, I found that the LLM and stylometric signals were sometimes strongly contradictory — one scoring 0.85 and the other 0.20 — in ways that made a simple average misleading. A 0.53 average of (0.85, 0.20) looks "uncertain" but is actually a strong disagreement, which is a different situation from two signals both returning 0.52. I added a **signal disagreement flag** (`signals_disagree: true` when `|llm_score - stylo_score| > 0.4`) to the audit log and label output. This wasn't in the spec but came directly from testing. The spec's framing of "uncertainty representation" pointed me toward this problem; it just didn't anticipate this specific solution.

---

## AI Usage

### Instance 1 — Flask app skeleton and first signal function

**What I directed the AI to do:** I provided my `planning.md` detection signals section and ASCII architecture diagram and asked it to generate (1) a Flask app skeleton with a `POST /submit` route stub and (2) a Groq-based LLM classifier function that returned a JSON-parseable `ai_probability` float.

**What it produced:** A working Flask skeleton and a `classify_with_groq()` function. The function used a system prompt that instructed the model to return only JSON, which was correct. However, the generated code parsed the response with `response.choices[0].message.content` and then called `json.loads()` directly — with no error handling for cases where the model returns a preamble before the JSON object.

**What I revised:** I added a try/except around the JSON parsing and a fallback that strips everything before the first `{` character. I also changed the prompt to include an explicit example of the expected output format, which made the model's responses more reliable. I tested with five inputs before wiring the function into the endpoint.

---

### Instance 2 — Confidence scoring logic and label generation

**What I directed the AI to do:** I provided the uncertainty representation section of my spec (including the threshold table) and asked it to generate (1) a `compute_confidence()` function that combined both signal scores using my documented weights, and (2) a `generate_label()` function that mapped the confidence score to one of my three label variants.

**What it produced:** Both functions were structurally correct and matched my spec. The `generate_label()` function used the right thresholds. However, the confidence formula used equal weights (0.5 / 0.5) rather than my specified 0.6 / 0.4 split — the model implemented "reasonable-looking" weighting rather than reading my spec carefully.

**What I revised:** I corrected the weights to match my spec (and later updated them again to 0.5 / 0.3 / 0.2 when I added the third signal). I also added the `signals_disagree` flag after my Milestone 4 testing revealed strong contradictions between signals that the average was masking. Neither of these changes came from the AI — they came from running the code against real inputs.

---

## Setup & Running Locally

```bash
# Clone and create virtual environment
git clone https://github.com/<your-username>/ai201-project4-provenance-guard
cd ai201-project4-provenance-guard
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run
python app.py
```

**Requirements:**
```
flask>=3.0.0
flask-limiter>=3.5.0
groq==0.15.0
python-dotenv==1.0.1
```

SQLite is built into Python — no additional database setup needed.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/submit` | Submit content for attribution analysis |
| `POST` | `/appeal` | Appeal a classification |
| `GET` | `/log` | View audit log entries |
| `GET` | `/analytics` | View detection analytics dashboard |
| `POST` | `/verify` | Submit live writing sample for human verification |

### Example curl Commands

```bash
# Submit content
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The sun dipped below the horizon...", "creator_id": "user-001"}'

# Submit an appeal
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d '{"content_id": "PASTE-CONTENT-ID-HERE", "creator_reasoning": "I wrote this myself."}'

# View audit log
curl -s http://localhost:5000/log

# View analytics
curl -s http://localhost:5000/analytics
```
