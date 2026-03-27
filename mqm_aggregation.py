"""
mqm_aggregation.py
==================
Deterministic scoring layer — no LLM calls.

Scoring approach
----------------
1.  Collect Stage-2 sub-category outputs (refined probabilities + confidence).
2.  Compute a confidence-weighted mean probability per super-category.
3.  Apply Stage-3 consistency / verification adjustment:
      - errors_verified=NO  → scale down by 0.25 (likely false-positive pass)
      - errors_verified=YES → scale by consistency_score / 100
4.  Map confirmed severity findings to MQM penalty points (Critical=25, Major=5, Minor=1).
5.  Compute overall error probability as a weighted sum across 5 categories.
6.  Derive final quality score as (1 - overall_error_prob) * 100, further
    reduced by normalised MQM penalty.

Category weights (MQM-inspired defaults, can be overridden):
  Accuracy          40 %
  Fluency           25 %
  Terminology       20 %
  Style             10 %
  Locale Convention  5 %
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from mqm_models import MTState, AggregationOutput, Stage2Output, Stage3Output


# ---------------------------------------------------------------------------
# Severity → MQM penalty mapping
# ---------------------------------------------------------------------------
_SEVERITY_PENALTY: Dict[str, int] = {
    "CRITICAL": 25,
    "MAJOR":     5,
    "MINOR":     1,
    "NEUTRAL":   0,
    "NONE":      0,
}

# Category weights must sum to 1.0
_CATEGORY_WEIGHTS: Dict[str, float] = {
    "accuracy":    0.40,
    "fluency":     0.25,
    "terminology": 0.20,
    "style":       0.10,
    "locale":      0.05,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_weighted_mean(probs: List[float], confs: List[float]) -> float:
    """Epistemic weighting: higher-confidence agents carry more weight."""
    if not probs or not confs:
        return 0.0
    weights = [c / 100.0 for c in confs]
    total_w = sum(weights)
    if total_w == 0.0:
        return sum(probs) / len(probs)   # fallback: plain mean
    return sum(p * w for p, w in zip(probs, weights)) / total_w


def _collect_stage2(state: MTState, sub_keys: List[str]) -> Tuple[List[float], List[float]]:
    """Pull (re_evaluated_prob, re_evaluated_confidence) from stage-2 outputs."""
    probs, confs = [], []
    for key in sub_keys:
        out: Stage2Output | None = state.get(key)   # type: ignore[assignment]
        if out is not None:
            probs.append(out["re_evaluated_prob"] if isinstance(out, dict) else out.re_evaluated_prob)
            confs.append(out["re_evaluated_confidence"] if isinstance(out, dict) else out.re_evaluated_confidence)
    return probs, confs


def _apply_stage3_adjustment(base_score: float, stage3: Stage3Output | None) -> float:
    """Adjust base probability with Stage-3 verification signal."""
    if stage3 is None:
        return base_score
    ev = stage3.errors_verified if hasattr(stage3, "errors_verified") else stage3.get("errors_verified", "YES")  # type: ignore
    if ev == "NO":
        return base_score * 0.25      # strong discount — agents found no real errors
    cs = stage3.consistency_score if hasattr(stage3, "consistency_score") else stage3.get("consistency_score", 50.0)  # type: ignore
    return base_score * (cs / 100.0)  # weight by inter-agent agreement


def _category_mqm_penalty(state: MTState, sub_keys: List[str], stage3_key: str) -> int:
    """
    Collect confirmed severity findings and convert to MQM penalty points.
    A sub-category contributes its penalty only when:
      - error_found == YES
      - The parent Stage-3 also confirmed errors_verified == YES
    """
    stage3: Stage3Output | None = state.get(stage3_key)  # type: ignore[assignment]
    if stage3 is not None:
        ev = stage3.errors_verified if hasattr(stage3, "errors_verified") else stage3.get("errors_verified", "NO")  # type: ignore
        false_positives = stage3.false_positives if hasattr(stage3, "false_positives") else stage3.get("false_positives", [])  # type: ignore
        if ev == "NO":
            return 0
    else:
        false_positives = []

    penalty = 0
    for key in sub_keys:
        if key in false_positives:
            continue
        out: Stage2Output | None = state.get(key)  # type: ignore[assignment]
        if out is None:
            continue
        ef = out.error_found if hasattr(out, "error_found") else out.get("error_found", "NO")  # type: ignore
        sev = out.severity if hasattr(out, "severity") else out.get("severity", "NONE")  # type: ignore
        if ef == "YES":
            penalty += _SEVERITY_PENALTY.get(sev, 0)
    return penalty


def _count_severities(state: MTState, all_sub_keys: List[str]) -> Dict[str, int]:
    counts = {"CRITICAL": 0, "MAJOR": 0, "MINOR": 0, "NEUTRAL": 0}
    for key in all_sub_keys:
        out = state.get(key)
        if out is None:
            continue
        ef = out.error_found if hasattr(out, "error_found") else out.get("error_found", "NO")  # type: ignore
        sev = out.severity if hasattr(out, "severity") else out.get("severity", "NONE")  # type: ignore
        if ef == "YES" and sev in counts:
            counts[sev] += 1
    return counts


# ---------------------------------------------------------------------------
# Public aggregation function  (LangGraph node)
# ---------------------------------------------------------------------------

# Sub-key lists per category
_SUB_KEYS = {
    "accuracy":    ["addition", "omission", "mistranslation", "untranslated_text"],
    "fluency":     ["grammar", "spelling", "punctuation", "register", "morphology", "word_order"],
    "terminology": ["incorrect_term", "inconsistent_term"],
    "style":       ["awkward_phrasing", "unnatural_flow"],
    "locale":      ["number_format", "date_format", "currency_format"],
}

_STAGE3_KEYS = {
    "accuracy":    "accuracyStage3",
    "fluency":     "fluencyStage3",
    "terminology": "terminologyStage3",
    "style":       "styleStage3",
    "locale":      "localeStage3",
}

_ALL_SUB_KEYS = [k for keys in _SUB_KEYS.values() for k in keys]


def aggregate_mt_quality(state: MTState) -> Dict[str, AggregationOutput]:
    """
    Compute all scores and return a dict updating the 'aggregation' key in MTState.
    This is a deterministic function — no LLM is called.
    """
    category_error_probs: Dict[str, float] = {}
    total_mqm_penalty = 0

    for cat, sub_keys in _SUB_KEYS.items():
        s3_key = _STAGE3_KEYS[cat]
        probs, confs = _collect_stage2(state, sub_keys)
        base_score = _confidence_weighted_mean(probs, confs)
        stage3 = state.get(s3_key)   # type: ignore[assignment]
        adjusted = _apply_stage3_adjustment(base_score, stage3)
        category_error_probs[cat] = round(adjusted, 4)
        total_mqm_penalty += _category_mqm_penalty(state, sub_keys, s3_key)

    # Weighted overall error probability
    overall_error_prob = sum(
        _CATEGORY_WEIGHTS[cat] * prob
        for cat, prob in category_error_probs.items()
    )

    # MQM-informed quality score
    # Base: (1 - overall_error_prob) * 100
    # Further penalised by MQM points (capped at 100 deduction)
    base_quality = (1.0 - overall_error_prob) * 100.0
    mqm_deduction = min(total_mqm_penalty, 100)
    final_quality = round(max(0.0, base_quality - (mqm_deduction * 0.3)), 2)
    # Note: we scale MQM deduction by 0.3 to avoid double-counting with prob-based reduction.
    # Adjust this blend factor based on your quality threshold calibration.

    # Severity counts
    sevs = _count_severities(state, _ALL_SUB_KEYS)

    result: AggregationOutput = {
        "accuracy_error_prob":     category_error_probs["accuracy"],
        "fluency_error_prob":      category_error_probs["fluency"],
        "terminology_error_prob":  category_error_probs["terminology"],
        "style_error_prob":        category_error_probs["style"],
        "locale_error_prob":       category_error_probs["locale"],
        "mqm_penalty":             float(total_mqm_penalty),
        "overall_error_probability": round(overall_error_prob, 4),
        "final_quality_score":     final_quality,
        "critical_count":          sevs["CRITICAL"],
        "major_count":             sevs["MAJOR"],
        "minor_count":             sevs["MINOR"],
        "neutral_count":           sevs["NEUTRAL"],
    }

    return {"aggregation": result}


# ---------------------------------------------------------------------------
# Convenience analysis helpers (call after pipeline)
# ---------------------------------------------------------------------------

def get_error_breakdown(state: MTState) -> Dict[str, Dict[str, float]]:
    """Return per-category dict of sub-category error probabilities."""
    breakdown: Dict[str, Dict[str, float]] = {}
    for cat, sub_keys in _SUB_KEYS.items():
        breakdown[cat] = {}
        for key in sub_keys:
            out = state.get(key)
            if out is not None:
                prob = out.re_evaluated_prob if hasattr(out, "re_evaluated_prob") else out.get("re_evaluated_prob", 0.0)  # type: ignore
                breakdown[cat][key] = round(prob, 4)
    return breakdown


def get_severity_breakdown(state: MTState) -> Dict[str, Dict[str, str]]:
    """Return per-sub-category severity findings."""
    result: Dict[str, Dict[str, str]] = {}
    for cat, sub_keys in _SUB_KEYS.items():
        result[cat] = {}
        for key in sub_keys:
            out = state.get(key)
            if out is not None:
                ef = out.error_found if hasattr(out, "error_found") else out.get("error_found", "NO")  # type: ignore
                sev = out.severity if hasattr(out, "severity") else out.get("severity", "NONE")  # type: ignore
                result[cat][key] = f"{ef} | {sev}"
    return result


def get_verified_errors(state: MTState) -> Dict[str, bool]:
    """Return which super-categories had at least one verified error."""
    return {
        cat: (
            (s3 := state.get(s3_key)) is not None  # type: ignore[assignment]
            and (s3.errors_verified if hasattr(s3, "errors_verified") else s3.get("errors_verified", "NO")) == "YES"  # type: ignore
        )
        for cat, s3_key in _STAGE3_KEYS.items()
    }
