"""
mqm_models.py  (v2)
====================
Pydantic models + LangGraph state.

Key additions over v1:
  - ErrorSpan: character-level index ranges for each detected error
  - Stage2Output now carries a List[ErrorSpan] instead of a plain string
  - max_rounds default is documented here: DEFAULT_MAX_ROUNDS = 2
    (one initial pass + one optional re-run if the audit fires)

Why max_rounds = 2?
  Round 1 runs the full pipeline cold.
  Round 2 (if triggered) re-runs with the missing-errors audit result injected
  into every Stage-2 and Stage-3 prompt, so agents specifically look for what
  was missed. A third round almost never improves results and doubles API cost.
  Set max_rounds=1 for speed/cost mode, max_rounds=3 for maximum recall on
  difficult sentence pairs.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Severity & existence literals
# ---------------------------------------------------------------------------
Severity   = Literal["CRITICAL", "MAJOR", "MINOR", "NEUTRAL", "NONE"]
ErrorExists = Literal["YES", "NO"]

# Exposed constant so callers can import the default instead of hard-coding it
DEFAULT_MAX_ROUNDS: int = 2


# ---------------------------------------------------------------------------
# Error Span  –  character-level index range in the MT string
# ---------------------------------------------------------------------------
class ErrorSpan(BaseModel):
    """A single detected error span within the machine-translated sentence."""

    start: int = Field(
        ...,
        description=(
            "0-based character index where the error span STARTS in the MT string. "
            "Count from the very first character of the MT sentence."
        ),
    )
    end: int = Field(
        ...,
        description=(
            "0-based character index where the error span ENDS (exclusive) in the MT string. "
            "mt[start:end] should reproduce the exact offending text."
        ),
    )
    span_text: str = Field(
        ...,
        description=(
            "The exact substring mt[start:end]. "
            "Must match the MT string exactly — no paraphrasing."
        ),
    )
    error_type: str = Field(
        ...,
        description=(
            "The MQM sub-category label for this span, e.g. "
            "'accuracy:omission', 'fluency:grammar', 'terminology:incorrect_term', "
            "'style:awkward_phrasing', 'locale_convention:number_format'."
        ),
    )
    severity: Severity = Field(
        ...,
        description="MQM severity of this specific span.",
    )
    explanation: str = Field(
        ...,
        description="One-sentence explanation of why this span is an error.",
    )


# ---------------------------------------------------------------------------
# Stage 1 — super-category scout
# ---------------------------------------------------------------------------
class Stage1Output(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0,
        description="Bayesian probability that at least one error of this category is present.")
    top_severity: Severity = Field(...,
        description="Worst-case severity you expect. Use NONE if probability < 0.15.")
    reason: str = Field(...,
        description="Step-by-step CoT justification. Cite exact source/MT tokens.")
    confidence: float = Field(..., ge=0.0, le=100.0,
        description="Confidence in this assessment (0-100).")


# ---------------------------------------------------------------------------
# Stage 2 — sub-category specialist  (now with error spans)
# ---------------------------------------------------------------------------
class Stage2Output(BaseModel):
    error_found: ErrorExists = Field(...,
        description="YES if you found a concrete instance of this sub-category error.")
    severity: Severity = Field(...,
        description=(
            "MQM severity (CRITICAL/MAJOR/MINOR/NEUTRAL/NONE). "
            "CRITICAL=meaning completely broken, MAJOR=clearly wrong/impedes comprehension, "
            "MINOR=noticeable but meaning intact, NEUTRAL=style preference only."
        ))
    re_evaluated_prob: float = Field(..., ge=0.0, le=1.0,
        description="Refined probability for THIS specific sub-category error.")
    agreement_with_stage1: Literal["AGREE", "PARTIALLY_AGREE", "DISAGREE"] = Field(...,
        description="Does your finding align with the Stage-1 broad signal?")
    error_spans: List[ErrorSpan] = Field(
        default_factory=list,
        description=(
            "List of character-level error spans in the MT string. "
            "Empty list if error_found=NO. "
            "Each span must satisfy: mt_string[span.start:span.end] == span.span_text."
        ),
    )
    reasoning: str = Field(...,
        description=(
            "Chain-of-thought: "
            "Step 1 – quote relevant MT span. "
            "Step 2 – explain error type. "
            "Step 3 – assess impact on meaning. "
            "Step 4 – assign severity."
        ))
    re_evaluated_confidence: float = Field(..., ge=0.0, le=100.0,
        description="Confidence in your sub-category assessment (0-100).")


# ---------------------------------------------------------------------------
# Stage 3 — meta-evaluator / verifier
# ---------------------------------------------------------------------------
class Stage3Output(BaseModel):
    errors_verified: ErrorExists = Field(...,
        description="YES if at least one sub-category error is verified to truly exist.")
    consistency_score: float = Field(..., ge=0.0, le=100.0,
        description="Inter-agent consistency (0-100). 100=perfect agreement.")
    worst_severity_confirmed: Severity = Field(...,
        description="Worst confirmed severity across verified sub-category errors.")
    false_positives: List[str] = Field(default_factory=list,
        description="Sub-category keys that appear to be false positives.")
    verified_spans: List[ErrorSpan] = Field(
        default_factory=list,
        description=(
            "Consolidated, deduplicated list of verified error spans "
            "across all Stage-2 agents for this category. "
            "Only include spans from sub-categories NOT in false_positives."
        ),
    )
    verification_reasoning: str = Field(...,
        description="Brief evidence-based justification for your verification decision.")


# ---------------------------------------------------------------------------
# Missing-errors audit
# ---------------------------------------------------------------------------
class MissingErrorsOutput(BaseModel):
    missing_errors_exist: ErrorExists = Field(...,
        description=(
            "YES only if you have concrete textual evidence of an error all prior agents missed. "
            "Default to NO — be very conservative."
        ))
    missing_error_types: List[str] = Field(default_factory=list,
        description="e.g. ['accuracy:omission', 'fluency:grammar']. Empty if NO.")
    evidence: str = Field(...,
        description="Concrete token-level evidence from source and MT for each missed error.")
    reasoning: str = Field(...,
        description="Why these errors were not caught in the previous pass.")


# ---------------------------------------------------------------------------
# Aggregation output  (deterministic, no LLM)
# ---------------------------------------------------------------------------
class AggregationOutput(TypedDict):
    # Per-category error probabilities
    accuracy_error_prob:    float
    fluency_error_prob:     float
    terminology_error_prob: float
    style_error_prob:       float
    locale_error_prob:      float

    # MQM penalty (0–100+ scale, higher = worse)
    mqm_penalty: float

    # Overall error probability (0–1)
    overall_error_probability: float

    # Final quality score (0–100, higher = better)
    final_quality_score: float

    # Severity counts
    critical_count: int
    major_count:    int
    minor_count:    int
    neutral_count:  int

    # All verified error spans from Stage-3 outputs (deduplicated)
    all_error_spans: List[dict]


# ---------------------------------------------------------------------------
# Global LangGraph state
# ---------------------------------------------------------------------------
class MTState(TypedDict):
    # ── Inputs ──────────────────────────────────────────────────────────────
    source:    str
    mt:        str
    reference: str

    # ── Stage 1 ─────────────────────────────────────────────────────────────
    accuracyStage1:    Optional[Stage1Output]
    fluencyStage1:     Optional[Stage1Output]
    terminologyStage1: Optional[Stage1Output]
    styleStage1:       Optional[Stage1Output]
    localeStage1:      Optional[Stage1Output]

    # ── Stage 2 – Accuracy ──────────────────────────────────────────────────
    addition:         Optional[Stage2Output]
    omission:         Optional[Stage2Output]
    mistranslation:   Optional[Stage2Output]
    untranslated_text:Optional[Stage2Output]

    # ── Stage 2 – Fluency ───────────────────────────────────────────────────
    grammar:     Optional[Stage2Output]
    spelling:    Optional[Stage2Output]
    punctuation: Optional[Stage2Output]
    register:    Optional[Stage2Output]
    morphology:  Optional[Stage2Output]
    word_order:  Optional[Stage2Output]

    # ── Stage 2 – Terminology ───────────────────────────────────────────────
    incorrect_term:    Optional[Stage2Output]
    inconsistent_term: Optional[Stage2Output]

    # ── Stage 2 – Style ─────────────────────────────────────────────────────
    awkward_phrasing: Optional[Stage2Output]
    unnatural_flow:   Optional[Stage2Output]

    # ── Stage 2 – Locale Convention ─────────────────────────────────────────
    number_format:   Optional[Stage2Output]
    date_format:     Optional[Stage2Output]
    currency_format: Optional[Stage2Output]

    # ── Stage 3 ─────────────────────────────────────────────────────────────
    accuracyStage3:    Optional[Stage3Output]
    fluencyStage3:     Optional[Stage3Output]
    terminologyStage3: Optional[Stage3Output]
    styleStage3:       Optional[Stage3Output]
    localeStage3:      Optional[Stage3Output]

    # ── Audit & loop control ─────────────────────────────────────────────────
    missingErrors: Optional[MissingErrorsOutput]
    round:         Optional[int]   # current round (starts at 1)
    max_rounds:    Optional[int]   # default: DEFAULT_MAX_ROUNDS = 2

    # ── Final result ─────────────────────────────────────────────────────────
    aggregation: Optional[AggregationOutput]
