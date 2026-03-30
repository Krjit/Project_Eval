from typing import Dict, List
from own_framework import MTState, AggregationOutput

def weighted_mean(probs: List[float], confs: List[float]) -> float:
    if not probs or not confs:
        return 0.0
    
    weights = [c / 100.0 for c in confs]
    total_weight = sum(weights)

    if total_weight == 0:
        return 0.0
    
    return sum(p * w for p, w in zip(probs, weights)) / total_weight

def aggregate_super_category(state: MTState, sub_keys: List[str], stage3_key: str) -> float:
    cross = state.get("cross_reasoning")
    
    probs = []
    confs = []

    if cross is not None:
        retained = cross.retained_errors
        dropped = cross.dropped_errors
        for key in sub_keys:
            if key in retained:
                agent_output = state.get(key)
                if agent_output is not None:
                    probs.append(agent_output.reEvaluatedProb)
                    confs.append(agent_output.reEvaluatedConfidence)
            elif key in dropped:
                agent_output = state.get(key)
                if agent_output is not None:
                    probs.append(0.0)
                    confs.append(0.0)
    else:
        for key in sub_keys:
            agent_output = state.get(key)
            if agent_output is not None:
                probs.append(agent_output.reEvaluatedProb)
                confs.append(agent_output.reEvaluatedConfidence)

    base_score = weighted_mean(probs, confs)

    stage3 = state.get(stage3_key)

    if stage3 is None:
        return base_score

    if stage3.errorsExists == "NO":
        return base_score * 0.1

    consistency_factor = stage3.consistencyScore / 100.0
    return base_score * consistency_factor

def aggregate_mt_quality(state: MTState) -> Dict[str, AggregationOutput]:
    accuracy_subs = ["addition", "omission", "mistranslation", "untranslated_text", "transliteration", "non_translation"]
    fluency_subs = ["punctuation", "spelling", "grammar", "register", "inconsistency", "characterEncoding"]
    terminology_subs = ["inappropriate_for_context", "inconsistency_use"]
    style_subs = ["awkward"]

    acc_score = aggregate_super_category(state, accuracy_subs, "accuracyStage3")
    flu_score = aggregate_super_category(state, fluency_subs, "fluencyStage3")
    term_score = aggregate_super_category(state, terminology_subs, "terminologyStage3")
    style_score = aggregate_super_category(state, style_subs, "styleStage3")

    weights = {
        "accuracy": 0.4,
        "fluency": 0.3,
        "terminology": 0.2,
        "style": 0.1,
    }

    overall_error_prob = (
        weights["accuracy"] * acc_score +
        weights["fluency"] * flu_score +
        weights["terminology"] * term_score +
        weights["style"] * style_score
    )

    final_quality_score_100 = (1 - overall_error_prob) * 100

    return {"aggregation": {
        "accuracy_error": acc_score,
        "fluency_error": flu_score,
        "terminology_error": term_score,
        "style_error": style_score,
        "overall_error_probability": overall_error_prob,
        "final_quality_score_100": final_quality_score_100
    }}