"""
mqm_run.py
==========
End-to-end runner: load a dataset → run the MQM pipeline on each sample →
save results as JSONL → optionally compute correlation with human MQM scores.

Usage examples
--------------

# 1. Run on WMT en-de (HuggingFace, 50 samples)
python mqm_run.py --dataset wmt_ende --split 2022 --max_samples 50

# 2. Run on WMT en-de from local TSV (HiMATE setup)
python mqm_run.py --dataset wmt_ende_tsv \
    --tsv_path data/mqm_general_MT2022_ende.tsv \
    --max_samples 50

# 3. Run on IndicMQM en-hi (HuggingFace IN22-Gen, no MQM labels)
python mqm_run.py --dataset indic_hf --max_samples 50

# 4. Run on IndicMQM en-hi from local annotations file
python mqm_run.py --dataset indic_local \
    --annotations_path data/indic_mqm_en_hi.csv \
    --max_samples 50

# 5. Run with custom max_rounds (default = 2)
python mqm_run.py --dataset wmt_ende --max_rounds 1 --max_samples 20

All results are saved to:
  results/<dataset>_<timestamp>.jsonl   (one JSON line per sample)
  results/<dataset>_<timestamp>_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mqm_datasets import (
    load_wmt_ende,
    load_wmt_ende_tsv,
    load_indicmqm_hf,
    load_indicmqm_local,
)
from mqm_models import DEFAULT_MAX_ROUNDS


# ---------------------------------------------------------------------------
# Lazy-import the compiled graph (avoids heavy import at module load)
# ---------------------------------------------------------------------------
def _get_app():
    from mqm_pipeline import app
    return app


# ---------------------------------------------------------------------------
# Serialiser for Pydantic objects
# ---------------------------------------------------------------------------
def _serialise(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Run one sample through the pipeline
# ---------------------------------------------------------------------------
def run_sample(
    app,
    sample: Dict[str, Any],
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Dict[str, Any]:
    """
    Invoke the MQM pipeline on a single sample dict.

    Returns the serialised pipeline state merged with dataset metadata.
    """
    input_state = {
        "source":     sample["source"],
        "mt":         sample["mt"],
        "reference":  sample["reference"],
        "round":      1,
        "max_rounds": max_rounds,
    }

    t0 = time.time()
    try:
        result = app.invoke(input_state)
        elapsed = round(time.time() - t0, 2)
        serialised = _serialise(result)
        serialised["_meta"] = {
            "system":          sample.get("system", ""),
            "lp":              sample.get("lp", ""),
            "domain":          sample.get("domain", ""),
            "human_mqm_score": sample.get("human_mqm_score"),
            "human_spans":     sample.get("human_spans"),
            "elapsed_s":       elapsed,
            "error":           None,
        }
    except Exception as e:
        serialised = {
            "_meta": {
                "system":          sample.get("system", ""),
                "lp":              sample.get("lp", ""),
                "domain":          sample.get("domain", ""),
                "human_mqm_score": sample.get("human_mqm_score"),
                "human_spans":     sample.get("human_spans"),
                "elapsed_s":       round(time.time() - t0, 2),
                "error":           str(e),
            },
            "source":    sample["source"],
            "mt":        sample["mt"],
            "reference": sample["reference"],
            "aggregation": None,
        }

    return serialised


# ---------------------------------------------------------------------------
# Extract predicted error spans from pipeline result
# ---------------------------------------------------------------------------
def extract_predicted_spans(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pull all verified error spans from the Stage-3 outputs.
    These are the spans the pipeline is most confident about.
    """
    spans = []
    for s3_key in ["accuracyStage3", "fluencyStage3", "terminologyStage3",
                   "styleStage3", "localeStage3"]:
        s3 = result.get(s3_key)
        if not s3:
            continue
        for span in s3.get("verified_spans", []):
            spans.append(span)

    # Also collect from Stage-2 outputs directly (unverified but useful)
    stage2_keys = [
        "addition", "omission", "mistranslation", "untranslated_text",
        "grammar", "spelling", "punctuation", "register", "morphology", "word_order",
        "incorrect_term", "inconsistent_term",
        "awkward_phrasing", "unnatural_flow",
        "number_format", "date_format", "currency_format",
    ]
    for s2_key in stage2_keys:
        s2 = result.get(s2_key)
        if not s2 or s2.get("error_found") != "YES":
            continue
        for span in s2.get("error_spans", []):
            spans.append(span)

    # Deduplicate by (start, end, error_type)
    seen = set()
    unique = []
    for sp in spans:
        k = (sp.get("start"), sp.get("end"), sp.get("error_type"))
        if k not in seen:
            seen.add(k)
            unique.append(sp)

    return sorted(unique, key=lambda s: s.get("start", 0))


# ---------------------------------------------------------------------------
# Compute Pearson / Spearman correlation with human MQM scores
# ---------------------------------------------------------------------------
def compute_correlations(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute segment-level Pearson and Spearman correlations between
    the framework's final_quality_score and human MQM scores.
    Requires scipy.
    """
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError:
        return {"error": "pip install scipy to compute correlations"}

    pred_scores, human_scores = [], []
    for r in results:
        agg  = r.get("aggregation") or {}
        meta = r.get("_meta") or {}
        pred = agg.get("final_quality_score")
        human = meta.get("human_mqm_score")
        if pred is not None and human is not None:
            pred_scores.append(float(pred))
            human_scores.append(float(human))

    if len(pred_scores) < 3:
        return {"warning": "Not enough paired samples for correlation", "n": len(pred_scores)}

    pear, _ = pearsonr(pred_scores, human_scores)
    spear, _ = spearmanr(pred_scores, human_scores)
    return {
        "n":        len(pred_scores),
        "pearson":  round(pear, 4),
        "spearman": round(spear, 4),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_batch(
    samples: List[Dict[str, Any]],
    dataset_name: str,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    output_dir: str = "results",
) -> List[Dict[str, Any]]:

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(output_dir, f"{dataset_name}_{ts}.jsonl")
    summary_path = os.path.join(output_dir, f"{dataset_name}_{ts}_summary.json")

    app = _get_app()
    results = []

    print(f"\n{'='*60}")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Samples      : {len(samples)}")
    print(f"  max_rounds   : {max_rounds}  (default={DEFAULT_MAX_ROUNDS})")
    print(f"  Output JSONL : {jsonl_path}")
    print(f"{'='*60}\n")

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for i, sample in enumerate(samples, 1):
            print(f"[{i:03d}/{len(samples)}] source: {sample['source'][:60]}…")
            result = run_sample(app, sample, max_rounds=max_rounds)

            # Attach predicted spans summary to result
            result["predicted_error_spans"] = extract_predicted_spans(result)

            agg = result.get("aggregation") or {}
            print(f"         quality={agg.get('final_quality_score','?')}  "
                  f"penalty={agg.get('mqm_penalty','?')}  "
                  f"spans={len(result['predicted_error_spans'])}  "
                  f"t={result['_meta']['elapsed_s']}s")

            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    # Summary stats
    quality_scores = [
        r["aggregation"]["final_quality_score"]
        for r in results
        if r.get("aggregation")
    ]
    corr = compute_correlations(results)
    summary = {
        "dataset":        dataset_name,
        "n_samples":      len(results),
        "max_rounds":     max_rounds,
        "avg_quality":    round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else None,
        "min_quality":    round(min(quality_scores), 2) if quality_scores else None,
        "max_quality":    round(max(quality_scores), 2) if quality_scores else None,
        "correlations":   corr,
        "errors":         sum(1 for r in results if r["_meta"]["error"]),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  Done. Avg quality score : {summary['avg_quality']}")
    print(f"  Correlations            : {corr}")
    print(f"  Summary saved           → {summary_path}")
    print(f"{'='*60}\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run MQM framework on benchmark datasets.")
    p.add_argument("--dataset", required=True,
                   choices=["wmt_ende", "wmt_ende_tsv", "indic_hf", "indic_local"],
                   help="Which dataset to use.")
    p.add_argument("--split", default="2022",
                   help="WMT year split: 2022 or 2023  (wmt_ende only).")
    p.add_argument("--tsv_path", default="data/mqm_general_MT2022_ende.tsv",
                   help="Path to WMT MQM TSV file  (wmt_ende_tsv only).")
    p.add_argument("--annotations_path", default="data/indic_mqm_en_hi.csv",
                   help="Path to local IndicMQM annotations file.")
    p.add_argument("--max_samples", type=int, default=50,
                   help="Max number of samples to evaluate.")
    p.add_argument("--max_rounds", type=int, default=DEFAULT_MAX_ROUNDS,
                   help=f"Max re-run rounds for the audit loop (default={DEFAULT_MAX_ROUNDS}).")
    p.add_argument("--output_dir", default="results",
                   help="Directory to save JSONL results.")
    p.add_argument("--system_filter", default=None,
                   help="Only evaluate one MT system (wmt_ende only).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == "wmt_ende":
        samples = load_wmt_ende(
            split=args.split,
            max_samples=args.max_samples,
            system_filter=args.system_filter,
        )
    elif args.dataset == "wmt_ende_tsv":
        samples = load_wmt_ende_tsv(
            tsv_path=args.tsv_path,
            max_samples=args.max_samples,
        )
    elif args.dataset == "indic_hf":
        samples = load_indicmqm_hf(max_samples=args.max_samples)
    elif args.dataset == "indic_local":
        samples = load_indicmqm_local(
            annotations_path=args.annotations_path,
            max_samples=args.max_samples,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    run_batch(
        samples=samples,
        dataset_name=args.dataset,
        max_rounds=args.max_rounds,
        output_dir=args.output_dir,
    )
