# planning.md — Provenance Guard

---

## Architecture

### Narrative

A piece of text enters the system via `POST /submit`. The request is first checked by the **rate limiter** (Flask-Limiter); if the caller has exceeded the per-IP limit, it's rejected with 429 before any work happens.

If the request passes rate limiting, the raw text is handed to the **detection pipeline**, which runs two independent signals in sequence: an LLM-based classifier (Groq) and a stylometric heuristic analyzer (pure Python). Each signal returns a score from 0–1 (higher = more likely AI-generated). Those two scores are merged by the **confidence combiner** into a single weighted confidence value, which is mapped to one of three transparency label tiers. The full decision — including both signal scores, the combined confidence, the label, and a timestamp — is written to the **audit log** (SQLite) before the response is returned.

If a creator disputes a result, `POST /appeal` immediately marks the content as `"under_review"` in the database and appends the appeal and the original decision to the audit log. No automated re-classification occurs; a human reviewer handles it.

### Diagram

```
① Submission flow

POST /submit ──► Rate limiter ──► ┌──────────────────────────────────┐
                                  │        Detection pipeline        │
                                  │                                  │
                                  │  Signal 1: LLM (Groq)           │
                                  │    score_llm ∈ [0, 1]           │
                                  │                                  │
                                  │  Signal 2: Stylometric (Python)  │
                                  │    score_stylo ∈ [0, 1]         │
                                  │           ↓          ↓           │
                                  │      Confidence combiner         │
                                  │  confidence = 0.6·llm            │
                                  │            + 0.4·stylo           │
                                  └──────────────┬───────────────────┘
                                                 ↓
                                      Transparency label selector
                                   (high-AI / uncertain / high-human)
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

## Detection signals

### Signal 1: LLM-based classifier (Groq)

**What it measures:** Holistic semantic and stylistic coherence. The model reads the full text and estimates whether it reads as human- or AI-authored — capturing formulaic phrasing, unnaturally balanced sentence structure, template-like transitions, and semantic predictability.

**Output format:** A single float `score_llm ∈ [0.0, 1.0]`. The model is prompted to respond with only a JSON object `{"ai_probability": <float>}` so the score is parsed directly — no free-text parsing needed.

**Prompt strategy:**
```
You are an expert at distinguishing human-written from AI-generated text.
Analyze the following text and estimate the probability that it was generated
by an AI (not written by a human). Consider: formulaic phrasing, unnatural
structural balance, absence of personal voice, predictable transitions.

Respond with ONLY valid JSON: {"ai_probability": <float between 0.0 and 1.0>}

Text:
"""
{text}
"""
```

**Why it differs between human and AI writing:** LLMs optimized by RLHF toward readability produce fluent, balanced, consistent output. Human writing carries idiosyncratic voice, tangents, imperfection, and genuine surprise.

**Blind spots:** Clean, structured human writing (corporate memos, formal essays) scores as AI-like. AI writing with deliberate noise can fool it. The model may underidentify output from models similar to itself.

---

### Signal 2: Stylometric heuristics (pure Python)

**What it measures:** Three statistical properties of the text computed without any external libraries:

| Sub-signal | How computed | AI tendency | Human tendency |
|---|---|---|---|
| Sentence length variance | `stdev(word counts per sentence)` | Low variance, clustered | High variance, spread |
| Type-token ratio (TTR) | `unique_words / total_words` | Lower TTR (repetitive vocab) | Higher TTR (diverse vocab) |
| Punctuation density | `punctuation_chars / total_chars` | Conventional, lower density | More expressive, higher density |

**Output format:** Each sub-signal is normalized to [0, 1] using empirically-set bounds, then averaged into a single `score_stylo ∈ [0.0, 1.0]`. Higher = more likely AI. Normalization details:

- **Sentence length variance:** clamp to [0, 25], then invert: `(25 - variance) / 25`. Low variance → high AI score.
- **TTR:** clamp to [0.3, 1.0], invert: `(TTR - 1.0) / (0.3 - 1.0)`. Low TTR → high AI score.
- **Punctuation density:** clamp to [0.01, 0.08], invert: `(density - 0.08) / (0.01 - 0.08)`. Low density → high AI score.

**Short text adjustment:** If the text contains fewer than 5 sentences or fewer than 80 words, the stylometric score is flagged as `low_confidence` and its weight in the combiner is reduced (see confidence combiner section).

**Why it differs:** RLHF training smooths statistical irregularities. Human writing is messier by nature.

**Blind spots:** Short texts lack sufficient data. Heavily edited human text (polished op-eds) can appear statistically AI-like. Poetry and stream-of-consciousness writing have atypical statistics that may produce false positives.

---

## Confidence combiner

**Formula (normal case):**
```
confidence = 0.6 * score_llm + 0.4 * score_stylo
```

**Formula (short text — stylometric flagged as low_confidence):**
```
confidence = 0.85 * score_llm + 0.15 * score_stylo
```

**Rationale for weights:** The LLM signal is the stronger signal for most text lengths because it captures meaning, not just structure. The stylometric signal adds independent structural evidence but is unreliable on short texts, so its weight drops when the text is short. A 60/40 split was chosen (not 50/50) because the LLM signal has demonstrated better discrimination in testing, while still giving the structural signal meaningful influence.

**Threshold mapping:**

| Confidence range | Result label | Interpretation |
|---|---|---|
| ≥ 0.85 | `"ai"` | High confidence this is AI-generated |
| 0.35 – 0.84 | `"uncertain"` | System cannot confidently attribute the content |
| < 0.35 | `"human"` | High confidence this is human-written |

The uncertain zone is intentionally wide (0.35–0.84) because false positives (misclassifying human work as AI) are more harmful than false negatives on a writing platform. A high-confidence AI label requires 0.85+, not just above 0.5.

---

## Uncertainty representation

**What does a score of 0.60 mean?**
It means both signals have moderate evidence pointing toward AI, but neither is strongly convincing. The LLM might say "this reads somewhat formulaic" while the stylometrics show slightly low variance. The system should not accuse anyone at 0.60 — it falls squarely in the uncertain zone and the label should reflect that.

**What does a score of 0.90 mean?**
Both signals are aligned and strong. The LLM says the text reads unmistakably AI-generated, and the stylometrics show low variance, low TTR, and low punctuation density. The system is confident enough to show the AI label.

**What does a score of 0.20 mean?**
Both signals point strongly toward human. The text has high sentence variance, rich vocabulary, expressive punctuation, and reads with personal voice. The system is confident enough to show the human label.

**Calibration approach:** Rather than trusting raw signal outputs as calibrated probabilities, the system maps them through empirically-set normalization bounds. The bounds are chosen so that typical AI-generated text from major models (GPT-4, Claude, Gemini) produces `score_llm` values of 0.75–0.95, and typical human creative writing produces 0.10–0.45. The stylometric bounds are set from similar empirical observation. Combined scores should cluster near the extremes for clearly-AI or clearly-human text, and near the middle for genuinely ambiguous text.

---

## Transparency label design

All three label variants are shown to users as plain-language text. The confidence score itself is not shown to end users — only the label text. The label text is returned in the API response under `"label"`.

### Variant 1 — High-confidence AI (`confidence ≥ 0.85`)

> **Attribution: AI-generated**
> Our detection system found strong signals that this content was generated by an AI writing tool. This label reflects pattern-based analysis — it is not a definitive ruling.
> If you wrote this yourself, you can dispute this classification using the appeal button below.

### Variant 2 — Uncertain (`confidence 0.35–0.84`)

> **Attribution: Uncertain**
> Our system could not confidently determine whether this content was written by a human or an AI tool. This is not an accusation — attribution is genuinely difficult, and this label simply means we don't know.
> If you feel this label is wrong, you can submit an appeal to have a human reviewer look at your content.

### Variant 3 — High-confidence human (`confidence < 0.35`)

> **Attribution: Human-written**
> Our detection system found strong signals that this content was written by a human author. This label reflects pattern-based analysis and is not a guarantee.

**Design notes:**
- All three variants use plain language with no technical jargon (no "confidence score," no "signal").
- Variants 1 and 2 include an explicit path to appeal; Variant 3 does not (no false accusation to dispute).
- Variant 1 includes the phrase "not a definitive ruling" to soften the accusation and pre-empt hostility.
- Variant 2 explicitly says "this is not an accusation" — the most important phrase in the entire design.

---

## Appeals workflow

**Who can submit an appeal:** Any submitter who has a `content_id` from a previous `/submit` call. There is no authentication in this implementation — the content_id acts as the token. (A production system would authenticate the creator.)

**What information they provide:**
```json
{
  "content_id": "<uuid>",
  "reason": "Free text from the creator explaining why they believe the classification is wrong."
}
```
The `reason` field is required and must be at least 10 characters. Empty appeals are rejected with 400.

**What the system does on receipt:**
1. Validates that `content_id` exists in the database.
2. Updates the content's `status` from `"decided"` to `"under_review"`.
3. Writes a new audit log entry of type `"appeal"` that includes:
   - `content_id`
   - `appeal_reason` (the creator's text)
   - `original_result` (the original decision that's being disputed)
   - `original_confidence` (the score at time of original decision)
   - `timestamp`
4. Returns `{ "status": "under_review", "content_id": "..." }`.

**What a human reviewer sees in the appeal queue (`GET /log?type=appeal`):**
```json
{
  "entry_type": "appeal",
  "content_id": "abc-123",
  "appeal_reason": "I wrote this essay myself over three days...",
  "original_result": "ai",
  "original_confidence": 0.91,
  "signal_scores": { "llm": 0.93, "stylometric": 0.87 },
  "timestamp": "2025-06-01T14:22:00Z"
}
```

**Constraints:**
- A content_id can only be appealed once. A second appeal attempt returns 409 Conflict.
- Appeals do not trigger automated re-classification. A human moderator resolves them manually (out of scope for this implementation).

---

## Anticipated edge cases

### Edge case 1: Very short texts (< 80 words)

A haiku, a tweet-length blurb, or a one-paragraph excerpt won't give the stylometric signal enough data to be meaningful. Sentence variance from 2–3 sentences is statistically noise; TTR from 20 words is unreliable. The system handles this by flagging the stylometric score as `low_confidence` and shifting the combiner weights to 85/15 in favor of the LLM signal. The API response will include `"short_text_warning": true` so callers know the result is based primarily on one signal.

### Edge case 2: Poetry and experimental writing

A poem with heavy repetition ("Do not go gentle into that good night" — a villanelle's refrain structure) will have low TTR and low sentence length variance by design. These are features of the form, not signals of AI generation. The stylometric signal will falsely score this as AI-like. The LLM signal is more likely to recognize the poetic form, but not guaranteed. Result: the combined score may land in the uncertain zone even for clearly human poetry, producing a "we don't know" label. This is an acceptable false negative (labeling human work as uncertain rather than human) — better than a false positive.

### Edge case 3: AI-assisted human writing

A human who uses AI to edit or polish their draft — but who wrote the core ideas — will produce text with mixed signals. The stylometrics may look human (the underlying structure is the writer's) while the LLM signal may lean AI (the surface polish is from a model). The system will likely produce an uncertain result, which is the honest answer. There is no clean classification for AI-assisted work; the uncertain label is the most honest output.

### Edge case 4: Translated text

Text translated from another language often has unusual stylometric properties — sentence structures that don't match native-English norms, low punctuation density if translated from languages like Japanese or Chinese. The stylometric signal may score translated human text as AI-like. The system has no translation detection; this is a known gap.

---

## API surface

| Method | Endpoint | Input | Output |
|--------|----------|-------|--------|
| `POST` | `/submit` | `{ "text": "...", "content_id": "..." }` | `{ "result": "ai\|human\|uncertain", "confidence": 0.0–1.0, "label": "...", "signals": { "llm": 0.0, "stylometric": 0.0 }, "short_text_warning": bool }` |
| `POST` | `/appeal` | `{ "content_id": "...", "reason": "..." }` | `{ "status": "under_review", "content_id": "..." }` |
| `GET` | `/log` | `?limit=N&content_id=X&type=decision\|appeal` (optional) | array of audit log entries |
| `GET` | `/status/<content_id>` | — | `{ "content_id", "result", "confidence", "label", "status", "appeal": null\|{...} }` |

---

## AI Tool Plan

### M3 — Submission endpoint + Signal 1 (LLM)

**Spec sections to provide to AI tool:**
- Architecture diagram (both flows)
- Detection signals → Signal 1 (including prompt template and output format)
- API surface table (just the `/submit` row)

**What to ask for:**
> "Generate a Flask app skeleton with a `POST /submit` endpoint and SQLite audit log setup. Then implement the LLM signal function using the Groq client. Use the exact prompt template in the spec. The function should return a float score_llm ∈ [0, 1]. Do not implement the second signal or the combiner yet — just return score_llm as the confidence for now."

**How to verify:**
- Call `POST /submit` with 3 test inputs: clearly AI text, clearly human text, and an ambiguous paragraph.
- Check that `score_llm` varies meaningfully (e.g., > 0.80 for AI text, < 0.40 for human text).
- Check that the audit log entry is written to SQLite after each call.
- Check that the endpoint returns a 429 when called > rate limit threshold in one minute.

---

### M4 — Signal 2 + confidence scoring

**Spec sections to provide to AI tool:**
- Architecture diagram
- Detection signals → Signal 2 (all three sub-signals, normalization formulas, short-text logic)
- Confidence combiner (both formulas, threshold table)
- Uncertainty representation section

**What to ask for:**
> "Implement the stylometric signal function. It should compute sentence length variance, TTR, and punctuation density, normalize each to [0, 1] using the bounds in the spec, average them into score_stylo, and flag low_confidence if the text has fewer than 5 sentences or 80 words. Then implement the confidence combiner using the weighted formulas in the spec. Wire both signals into the existing /submit endpoint and update the response to include both signal scores, the combined confidence, and short_text_warning."

**How to verify:**
- Run both signals on 5+ text samples (at least 2 clearly AI, 2 clearly human, 1 short).
- Check that `score_stylo` moves in the expected direction (low for human creative writing, high for AI text).
- Check that the combined confidence produces the correct label tier (high-AI / uncertain / human) for each sample.
- Specifically test a short text (< 80 words) and confirm `short_text_warning: true` appears and weights shift.

---

### M5 — Production layer: labels, appeals, rate limiting, audit log

**Spec sections to provide to AI tool:**
- Architecture diagram (appeal flow)
- Transparency label design (all three variant texts, verbatim)
- Appeals workflow (full section)
- Rate limiting section (from README plan)
- API surface table (all four endpoints)

**What to ask for:**
> "Implement the label selector function that maps the confidence score to one of the three label variant texts from the spec. Then implement POST /appeal with the validation, status update, and audit log write described in the spec. Implement GET /log with optional query params. Implement GET /status/<content_id>. Ensure all three transparency label variants are reachable (not dead code paths)."

**How to verify:**
- Call `/submit` with text that should score in each tier; confirm the correct label text appears verbatim in the response.
- Submit an appeal on a content_id and confirm status changes to `"under_review"` in `/status/<content_id>`.
- Confirm a second appeal on the same content_id returns 409.
- Check `/log` returns at least 3 entries with the correct structure after several submissions and an appeal.

---

## Stretch feature planning

*(Update this section before starting each stretch feature)*

---

## Milestone checklist

- [x] M1: Architecture narrative written
- [x] M1: Two detection signals chosen and documented
- [x] M1: False positive scenario traced
- [x] M1: API surface defined
- [x] M1: Architecture diagram created
- [x] M2: Five spec questions answered with implementation-ready detail
- [x] M2: Three label variants written out verbatim
- [x] M2: Confidence thresholds defined (not a binary flip at 0.5)
- [x] M2: AI Tool Plan written for M3, M4, M5
- [x] M2: Edge cases documented (4 specific scenarios)
- [ ] M3: Flask app + LLM signal implemented and tested
- [ ] M4: Stylometric signal + combiner implemented and tested
- [ ] M5: Labels, appeals, rate limiting, GET /log and /status implemented
- [ ] Stretch features (if attempted)
