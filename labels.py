"""
labels.py — Transparency label text for Provenance Guard.

Four variants, verbatim from planning.md:
  "ai"       — high-confidence AI-generated
  "uncertain" — system cannot confidently attribute
  "human"    — high-confidence human-written
  "verified" — creator has passed the provenance certificate step (stretch feature)
"""


def get_label(result: str) -> str:
    """
    Return the transparency label text for a given result tier.
    result must be one of: "ai", "uncertain", "human", "verified"
    """
    if result == "ai":
        return (
            "Attribution: AI-generated\n\n"
            "Our detection system found strong signals that this content was generated "
            "by an AI writing tool. This label reflects pattern-based analysis — it is "
            "not a definitive ruling.\n\n"
            "If you wrote this yourself, you can dispute this classification using the "
            "appeal button below."
        )
    elif result == "uncertain":
        return (
            "Attribution: Uncertain\n\n"
            "Our system could not confidently determine whether this content was written "
            "by a human or an AI tool. This is not an accusation — attribution is "
            "genuinely difficult, and this label simply means we don't know.\n\n"
            "If you feel this label is wrong, you can submit an appeal to have a human "
            "reviewer look at your content."
        )
    elif result == "verified":
        return (
            "Attribution: Verified Human ✓\n\n"
            "This creator has completed Provenance Guard's human verification step. "
            "They submitted a live writing sample that scored below the AI threshold, "
            "and their identity has been confirmed under their verified account.\n\n"
            "Verification reduces the likelihood of misclassification but does not "
            "guarantee all content from this creator is human-written."
        )
    else:  # "human"
        return (
            "Attribution: Human-written\n\n"
            "Our detection system found strong signals that this content was written by "
            "a human author. This label reflects pattern-based analysis and is not a "
            "guarantee."
        )
