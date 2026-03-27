"""
mqm_pipeline.py
===============
LangGraph pipeline for the 5-category MQM MT evaluation framework.

Architecture (3 stages + audit loop)
--------------------------------------

  START
    │ (parallel)
    ├─ Accuracy Scout   ─┐
    ├─ Fluency Scout    ─┤
    ├─ Terminology Scout─┤  Stage 1 (super-category first-pass)
    ├─ Style Scout      ─┤
    └─ Locale Scout     ─┘
         │ (parallel per category)
    ┌────┴──────────────────────────────────────────────────────┐
    │  Accuracy: addition, omission, mistranslation, untranslated│
    │  Fluency:  grammar, spelling, punctuation, register,       │
    │            morphology, word_order                          │
    │  Terminology: incorrect_term, inconsistent_term            │  Stage 2
    │  Style: awkward_phrasing, unnatural_flow                   │
    │  Locale: number_format, date_format, currency_format       │
    └────┬──────────────────────────────────────────────────────┘
         │ (parallel per category)
    ┌────┴──────────────────┐
    │  5 × Stage-3 verifiers │  Stage 3 (meta-evaluators)
    └────┬──────────────────┘
         │
    Missing-Errors Audit
         │
    ┌────┴──────────────────────────┐
    │  errors missed?               │
    │  YES & round < max_rounds     ├──► loop_controller ──► Stage 1 (all 5)
    │  NO  or  max_rounds reached   ├──► Aggregation
    └───────────────────────────────┘
         │
        END
"""

from __future__ import annotations

import json

from langgraph.graph import StateGraph, START, END

from mqm_models import MTState
from mqm_aggregation import aggregate_mt_quality
from mqm_agents import (
    make_stage1_agent,
    make_stage2_agent,
    make_stage3_agent,
    make_missing_errors_agent,
)
from mqm_prompts import (
    # Stage 1
    ACCURACY_S1_PROMPT,
    FLUENCY_S1_PROMPT,
    TERMINOLOGY_S1_PROMPT,
    STYLE_S1_PROMPT,
    LOCALE_S1_PROMPT,
    # Stage 2 – Accuracy
    ADDITION_S2_PROMPT,
    OMISSION_S2_PROMPT,
    MISTRANSLATION_S2_PROMPT,
    UNTRANSLATED_S2_PROMPT,
    # Stage 2 – Fluency
    GRAMMAR_S2_PROMPT,
    SPELLING_S2_PROMPT,
    PUNCTUATION_S2_PROMPT,
    REGISTER_S2_PROMPT,
    MORPHOLOGY_S2_PROMPT,
    WORD_ORDER_S2_PROMPT,
    # Stage 2 – Terminology
    INCORRECT_TERM_S2_PROMPT,
    INCONSISTENT_TERM_S2_PROMPT,
    # Stage 2 – Style
    AWKWARD_PHRASING_S2_PROMPT,
    UNNATURAL_FLOW_S2_PROMPT,
    # Stage 2 – Locale
    NUMBER_FORMAT_S2_PROMPT,
    DATE_FORMAT_S2_PROMPT,
    CURRENCY_FORMAT_S2_PROMPT,
    # Stage 3
    ACCURACY_S3_PROMPT,
    FLUENCY_S3_PROMPT,
    TERMINOLOGY_S3_PROMPT,
    STYLE_S3_PROMPT,
    LOCALE_S3_PROMPT,
    # Audit
    MISSING_ERRORS_PROMPT,
)


# ===========================================================================
# Build all agent callables
# ===========================================================================

# ── Stage 1 ─────────────────────────────────────────────────────────────────
accuracy_scout    = make_stage1_agent(ACCURACY_S1_PROMPT,    "accuracyStage1")
fluency_scout     = make_stage1_agent(FLUENCY_S1_PROMPT,     "fluencyStage1")
terminology_scout = make_stage1_agent(TERMINOLOGY_S1_PROMPT, "terminologyStage1")
style_scout       = make_stage1_agent(STYLE_S1_PROMPT,       "styleStage1")
locale_scout      = make_stage1_agent(LOCALE_S1_PROMPT,      "localeStage1")

# ── Stage 2 – Accuracy ──────────────────────────────────────────────────────
addition_agent        = make_stage2_agent(ADDITION_S2_PROMPT,      "addition",        "accuracyStage1")
omission_agent        = make_stage2_agent(OMISSION_S2_PROMPT,      "omission",        "accuracyStage1")
mistranslation_agent  = make_stage2_agent(MISTRANSLATION_S2_PROMPT,"mistranslation",  "accuracyStage1")
untranslated_agent    = make_stage2_agent(UNTRANSLATED_S2_PROMPT,  "untranslated_text","accuracyStage1")

# ── Stage 2 – Fluency ───────────────────────────────────────────────────────
grammar_agent     = make_stage2_agent(GRAMMAR_S2_PROMPT,     "grammar",     "fluencyStage1")
spelling_agent    = make_stage2_agent(SPELLING_S2_PROMPT,    "spelling",    "fluencyStage1")
punctuation_agent = make_stage2_agent(PUNCTUATION_S2_PROMPT, "punctuation", "fluencyStage1")
register_agent    = make_stage2_agent(REGISTER_S2_PROMPT,    "register",    "fluencyStage1")
morphology_agent  = make_stage2_agent(MORPHOLOGY_S2_PROMPT,  "morphology",  "fluencyStage1")
word_order_agent  = make_stage2_agent(WORD_ORDER_S2_PROMPT,  "word_order",  "fluencyStage1")

# ── Stage 2 – Terminology ───────────────────────────────────────────────────
incorrect_term_agent   = make_stage2_agent(INCORRECT_TERM_S2_PROMPT,   "incorrect_term",   "terminologyStage1")
inconsistent_term_agent= make_stage2_agent(INCONSISTENT_TERM_S2_PROMPT,"inconsistent_term","terminologyStage1")

# ── Stage 2 – Style ─────────────────────────────────────────────────────────
awkward_phrasing_agent = make_stage2_agent(AWKWARD_PHRASING_S2_PROMPT,"awkward_phrasing","styleStage1")
unnatural_flow_agent   = make_stage2_agent(UNNATURAL_FLOW_S2_PROMPT,  "unnatural_flow",  "styleStage1")

# ── Stage 2 – Locale ────────────────────────────────────────────────────────
number_format_agent   = make_stage2_agent(NUMBER_FORMAT_S2_PROMPT,  "number_format",  "localeStage1")
date_format_agent     = make_stage2_agent(DATE_FORMAT_S2_PROMPT,    "date_format",    "localeStage1")
currency_format_agent = make_stage2_agent(CURRENCY_FORMAT_S2_PROMPT,"currency_format","localeStage1")

# ── Stage 3 ─────────────────────────────────────────────────────────────────
accuracy_verifier    = make_stage3_agent(ACCURACY_S3_PROMPT,    "accuracyStage3",    "accuracyStage1")
fluency_verifier     = make_stage3_agent(FLUENCY_S3_PROMPT,     "fluencyStage3",     "fluencyStage1")
terminology_verifier = make_stage3_agent(TERMINOLOGY_S3_PROMPT, "terminologyStage3", "terminologyStage1")
style_verifier       = make_stage3_agent(STYLE_S3_PROMPT,       "styleStage3",       "styleStage1")
locale_verifier      = make_stage3_agent(LOCALE_S3_PROMPT,      "localeStage3",      "localeStage1")

# ── Audit agent ─────────────────────────────────────────────────────────────
missing_errors_auditor = make_missing_errors_agent(MISSING_ERRORS_PROMPT, "missingErrors")


# ===========================================================================
# Loop control helpers
# ===========================================================================

def increment_round(state: MTState) -> dict:
    return {"round": (state.get("round") or 1) + 1}


def route_after_audit(state: MTState) -> str:
    missing   = state.get("missingErrors")
    round_    = state.get("round") or 1
    max_rnd   = state.get("max_rounds") or 2  # default: allow 1 re-run

    if (
        missing is not None
        and (missing.missing_errors_exist if hasattr(missing, "missing_errors_exist")
             else missing.get("missing_errors_exist", "NO")) == "YES"
        and round_ < max_rnd
    ):
        return "loop"
    return "done"


# ===========================================================================
# Build the LangGraph StateGraph
# ===========================================================================

def build_graph() -> StateGraph:
    g = StateGraph(MTState)

    # ── Register nodes ──────────────────────────────────────────────────────
    # Stage 1
    g.add_node("accuracy_s1",    accuracy_scout)
    g.add_node("fluency_s1",     fluency_scout)
    g.add_node("terminology_s1", terminology_scout)
    g.add_node("style_s1",       style_scout)
    g.add_node("locale_s1",      locale_scout)

    # Stage 2 – Accuracy
    g.add_node("addition",         addition_agent)
    g.add_node("omission",         omission_agent)
    g.add_node("mistranslation",   mistranslation_agent)
    g.add_node("untranslated_text",untranslated_agent)

    # Stage 2 – Fluency
    g.add_node("grammar",     grammar_agent)
    g.add_node("spelling",    spelling_agent)
    g.add_node("punctuation", punctuation_agent)
    g.add_node("register",    register_agent)
    g.add_node("morphology",  morphology_agent)
    g.add_node("word_order",  word_order_agent)

    # Stage 2 – Terminology
    g.add_node("incorrect_term",    incorrect_term_agent)
    g.add_node("inconsistent_term", inconsistent_term_agent)

    # Stage 2 – Style
    g.add_node("awkward_phrasing", awkward_phrasing_agent)
    g.add_node("unnatural_flow",   unnatural_flow_agent)

    # Stage 2 – Locale
    g.add_node("number_format",   number_format_agent)
    g.add_node("date_format",     date_format_agent)
    g.add_node("currency_format", currency_format_agent)

    # Stage 3
    g.add_node("accuracy_s3",    accuracy_verifier)
    g.add_node("fluency_s3",     fluency_verifier)
    g.add_node("terminology_s3", terminology_verifier)
    g.add_node("style_s3",       style_verifier)
    g.add_node("locale_s3",      locale_verifier)

    # Control nodes
    g.add_node("missing_errors_audit", missing_errors_auditor)
    g.add_node("loop_controller",      increment_round)
    g.add_node("aggregation",          aggregate_mt_quality)

    # ── Edges ────────────────────────────────────────────────────────────────

    # START → Stage 1 (5 parallel scouts)
    for s1 in ["accuracy_s1", "fluency_s1", "terminology_s1", "style_s1", "locale_s1"]:
        g.add_edge(START, s1)

    # Stage 1 → Stage 2 (fan-out per category)
    for s2 in ["addition", "omission", "mistranslation", "untranslated_text"]:
        g.add_edge("accuracy_s1", s2)
    for s2 in ["grammar", "spelling", "punctuation", "register", "morphology", "word_order"]:
        g.add_edge("fluency_s1", s2)
    for s2 in ["incorrect_term", "inconsistent_term"]:
        g.add_edge("terminology_s1", s2)
    for s2 in ["awkward_phrasing", "unnatural_flow"]:
        g.add_edge("style_s1", s2)
    for s2 in ["number_format", "date_format", "currency_format"]:
        g.add_edge("locale_s1", s2)

    # Stage 2 → Stage 3 (fan-in per category)
    for s2 in ["addition", "omission", "mistranslation", "untranslated_text"]:
        g.add_edge(s2, "accuracy_s3")
    for s2 in ["grammar", "spelling", "punctuation", "register", "morphology", "word_order"]:
        g.add_edge(s2, "fluency_s3")
    for s2 in ["incorrect_term", "inconsistent_term"]:
        g.add_edge(s2, "terminology_s3")
    for s2 in ["awkward_phrasing", "unnatural_flow"]:
        g.add_edge(s2, "style_s3")
    for s2 in ["number_format", "date_format", "currency_format"]:
        g.add_edge(s2, "locale_s3")

    # Stage 3 → Audit
    for s3 in ["accuracy_s3", "fluency_s3", "terminology_s3", "style_s3", "locale_s3"]:
        g.add_edge(s3, "missing_errors_audit")

    # Conditional loop or done
    g.add_conditional_edges(
        "missing_errors_audit",
        route_after_audit,
        {"loop": "loop_controller", "done": "aggregation"},
    )

    # Loop back to Stage 1
    for s1 in ["accuracy_s1", "fluency_s1", "terminology_s1", "style_s1", "locale_s1"]:
        g.add_edge("loop_controller", s1)

    # Aggregation → END
    g.add_edge("aggregation", END)

    return g


# Compile once at module level for reuse
app = build_graph().compile()
print("[mqm_pipeline] Graph compiled successfully.")


# ===========================================================================
# Serialisation helper
# ===========================================================================

def _serialise(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


# ===========================================================================
# Main — example invocation
# ===========================================================================

if __name__ == "__main__":
    input_state: MTState = {
        # ── English → Hindi example ──────────────────────────────────────────
        "source": (
            "The qualities that determine a subculture as distinct may be linguistic, "
            "aesthetic, religious, political, sexual, geographical, or a combination of factors."
        ),
        "mt": (
            "वे गुण जो किसी उप-संस्कृति को अलग बनाते हैं, जैसे कि भाषा, सौंदर्य, "
            "धर्म, राजनीति, यौन, भूगोल या कई सारे कारकों का मिश्रण हो सकते हैं."
        ),
        "reference": (
            "उपसंस्कृति को विशिष्ट रूप से निर्धारित करने वाले गुण भाषाई, सौंदर्य, "
            "धार्मिक, राजनीतिक, यौन, भौगोलिक या कारकों का संयोजन हो सकते हैं।"
        ),
        # ── Pipeline control ─────────────────────────────────────────────────
        "round":      1,
        "max_rounds": 2,   # allow one re-run if audit detects missed errors
    }

    print("\n[mqm_pipeline] Running evaluation …\n")
    result = app.invoke(input_state)
    serialised = _serialise(result)

    # ── Pretty print aggregation summary ────────────────────────────────────
    agg = serialised.get("aggregation", {})
    print("=" * 60)
    print("  MQM EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Final Quality Score   : {agg.get('final_quality_score', 'N/A')} / 100")
    print(f"  Overall Error Prob    : {agg.get('overall_error_probability', 'N/A'):.3f}")
    print(f"  MQM Penalty Points    : {agg.get('mqm_penalty', 'N/A')}")
    print("-" * 60)
    print(f"  Accuracy  error prob  : {agg.get('accuracy_error_prob', 'N/A'):.3f}")
    print(f"  Fluency   error prob  : {agg.get('fluency_error_prob', 'N/A'):.3f}")
    print(f"  Terminology error prob: {agg.get('terminology_error_prob', 'N/A'):.3f}")
    print(f"  Style     error prob  : {agg.get('style_error_prob', 'N/A'):.3f}")
    print(f"  Locale    error prob  : {agg.get('locale_error_prob', 'N/A'):.3f}")
    print("-" * 60)
    print(f"  CRITICAL errors: {agg.get('critical_count', 0)}")
    print(f"  MAJOR    errors: {agg.get('major_count', 0)}")
    print(f"  MINOR    errors: {agg.get('minor_count', 0)}")
    print("=" * 60)

    # ── Persist full result ──────────────────────────────────────────────────
    with open("mqm_result.json", "w", encoding="utf-8") as f:
        json.dump(serialised, f, indent=2, ensure_ascii=False)
    print("\n[mqm_pipeline] Full result saved → mqm_result.json")
