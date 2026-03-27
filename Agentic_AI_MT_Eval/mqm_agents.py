"""
mqm_agents.py  (v2)
====================
Factory functions for all three pipeline stages.

Key addition over v1: Stage-2 agents now receive explicit span-detection
instructions and must return List[ErrorSpan] with character indices.
"""

from __future__ import annotations

import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from mqm_models import (
    MTState, Stage1Output, Stage2Output, Stage3Output, MissingErrorsOutput,
)

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


# ---------------------------------------------------------------------------
# Shared serialiser
# ---------------------------------------------------------------------------
def _s(obj) -> str | dict | list:
    if obj is None:
        return "Not available"
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_s(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Span-detection instruction block  (injected into every Stage-2 prompt)
# ---------------------------------------------------------------------------
_SPAN_INSTRUCTIONS = """
─── ERROR SPAN DETECTION INSTRUCTIONS ──────────────────────────────────────
You MUST populate the error_spans list if error_found = YES.

For each error span:
  1. Find the EXACT substring in the MT string that is erroneous.
  2. Count characters from the START of the MT string (0-based index).
     - Spaces, punctuation, and Unicode characters each count as 1.
     - Devanagari, Chinese, Arabic etc.: each Unicode codepoint = 1 character.
  3. Set start = index of first character of the offending span.
  4. Set end   = index AFTER the last character  (Python slice convention).
  5. Set span_text = exactly mt_string[start:end]  (copy-paste, no changes).
  6. Set error_type = "category:subcategory"  (e.g. "accuracy:omission").
  7. Set severity   = same severity you assigned for this sub-category.
  8. Set explanation = one sentence describing the error.

Verification: mt_string[start:end] must equal span_text exactly.
If you cannot identify the exact span, set error_found = NO instead of guessing.

Example (MT = "He go to school yesterday."):
  start=3, end=5, span_text="go", error_type="fluency:grammar",
  severity="MINOR", explanation="Subject-verb disagreement: should be 'went'."
─────────────────────────────────────────────────────────────────────────────
"""


# ===========================================================================
# STAGE 1 factory
# ===========================================================================
def make_stage1_agent(system_prompt: str, state_key: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
SOURCE SENTENCE:
{source}

MACHINE TRANSLATED SENTENCE:
{mt}

REFERENCE SENTENCE:
{reference}

Evaluate whether this MT contains errors of the category defined in your instructions.
Follow your chain-of-thought steps exactly.
        """),
    ])
    chain = prompt_template | llm.with_structured_output(Stage1Output)

    def agent_fn(state: MTState) -> Dict[str, Stage1Output]:
        return {state_key: chain.invoke({
            "source":    state["source"],
            "mt":        state["mt"],
            "reference": state["reference"],
        })}

    agent_fn.__name__ = f"stage1_{state_key}"
    return agent_fn


# ===========================================================================
# STAGE 2 factory  (with span detection)
# ===========================================================================
def make_stage2_agent(system_prompt: str, state_key: str, parent_stage1_key: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt + _SPAN_INSTRUCTIONS),
        ("human", """
SOURCE SENTENCE:
{source}

MACHINE TRANSLATED SENTENCE (index your spans against THIS string exactly):
{mt}

REFERENCE SENTENCE:
{reference}

─── STAGE-1 PARENT EVALUATION ───────────────────────────────────────────────
{stage1_eval}

─── EVALUATION CONTEXT ──────────────────────────────────────────────────────
Current Round  : {round}
Missing-Errors Audit (prior round — empty on round 1):
{missing_errors}
─────────────────────────────────────────────────────────────────────────────

Evaluate ONLY the sub-category defined in your instructions.
Follow Chain-of-Thought steps. Then populate error_spans precisely.
        """),
    ])
    chain = prompt_template | llm.with_structured_output(Stage2Output)

    def agent_fn(state: MTState) -> Dict[str, Stage2Output]:
        output = chain.invoke({
            "source":         state["source"],
            "mt":             state["mt"],
            "reference":      state["reference"],
            "stage1_eval":    _s(state.get(parent_stage1_key)),
            "round":          state.get("round", 1),
            "missing_errors": _s(state.get("missingErrors")),
        })
        # Post-process: validate span indices against MT string
        mt_str = state["mt"]
        validated_spans = []
        for span in (output.error_spans or []):
            extracted = mt_str[span.start:span.end] if 0 <= span.start < span.end <= len(mt_str) else ""
            if extracted == span.span_text:
                validated_spans.append(span)
            # If mismatch, drop the span (don't hallucinate)
        output.error_spans = validated_spans
        return {state_key: output}

    agent_fn.__name__ = f"stage2_{state_key}"
    return agent_fn


# ===========================================================================
# STAGE 3 factory  (verifies + consolidates spans)
# ===========================================================================
_SUB_KEY_MAP: Dict[str, List[str]] = {
    "accuracyStage1":    ["addition", "omission", "mistranslation", "untranslated_text"],
    "fluencyStage1":     ["grammar", "spelling", "punctuation", "register", "morphology", "word_order"],
    "terminologyStage1": ["incorrect_term", "inconsistent_term"],
    "styleStage1":       ["awkward_phrasing", "unnatural_flow"],
    "localeStage1":      ["number_format", "date_format", "currency_format"],
}


def make_stage3_agent(system_prompt: str, state_key: str, parent_stage1_key: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
SOURCE SENTENCE:
{source}

MACHINE TRANSLATED SENTENCE:
{mt}

REFERENCE SENTENCE:
{reference}

─── STAGE-1 SUPER-CATEGORY EVALUATION ───────────────────────────────────────
{stage1_eval}

─── STAGE-2 SUB-CATEGORY EVALUATIONS (include error_spans) ──────────────────
{stage2_evals}

─── EVALUATION CONTEXT ──────────────────────────────────────────────────────
Current Round  : {round}
Missing-Errors Audit (prior round):
{missing_errors}
─────────────────────────────────────────────────────────────────────────────

Verify consistency and error existence.
In verified_spans, consolidate ALL confirmed error spans from Stage-2 agents
(excluding any sub-categories you list in false_positives).
Deduplicate overlapping spans — keep the one with higher severity.
        """),
    ])
    chain = prompt_template | llm.with_structured_output(Stage3Output)
    sub_keys = _SUB_KEY_MAP.get(parent_stage1_key, [])

    def agent_fn(state: MTState) -> Dict[str, Stage3Output]:
        stage2_outputs = {k: _s(state.get(k)) for k in sub_keys}
        return {state_key: chain.invoke({
            "source":         state["source"],
            "mt":             state["mt"],
            "reference":      state["reference"],
            "stage1_eval":    _s(state.get(parent_stage1_key)),
            "stage2_evals":   stage2_outputs,
            "round":          state.get("round", 1),
            "missing_errors": _s(state.get("missingErrors")),
        })}

    agent_fn.__name__ = f"stage3_{state_key}"
    return agent_fn


# ===========================================================================
# Missing-errors audit agent
# ===========================================================================
def make_missing_errors_agent(system_prompt: str, state_key: str = "missingErrors"):
    _ALL_STAGE_KEYS = [
        "accuracyStage1","fluencyStage1","terminologyStage1","styleStage1","localeStage1",
        "addition","omission","mistranslation","untranslated_text",
        "grammar","spelling","punctuation","register","morphology","word_order",
        "incorrect_term","inconsistent_term",
        "awkward_phrasing","unnatural_flow",
        "number_format","date_format","currency_format",
        "accuracyStage3","fluencyStage3","terminologyStage3","styleStage3","localeStage3",
    ]
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
SOURCE SENTENCE:
{source}

MACHINE TRANSLATED SENTENCE:
{mt}

REFERENCE SENTENCE:
{reference}

─── CURRENT ROUND ────────────────────────────────────────────────────────────
Round: {round}

─── ALL PRIOR PIPELINE OUTPUTS ───────────────────────────────────────────────
{prior_state}
─────────────────────────────────────────────────────────────────────────────

Review and decide if any significant error was missed. Be CONSERVATIVE.
        """),
    ])
    chain = prompt_template | llm.with_structured_output(MissingErrorsOutput)

    def agent_fn(state: MTState) -> Dict[str, MissingErrorsOutput]:
        prior = {k: _s(state.get(k)) for k in _ALL_STAGE_KEYS}
        return {state_key: chain.invoke({
            "source":      state["source"],
            "mt":          state["mt"],
            "reference":   state["reference"],
            "round":       state.get("round", 1),
            "prior_state": prior,
        })}

    agent_fn.__name__ = "missing_errors_audit"
    return agent_fn
