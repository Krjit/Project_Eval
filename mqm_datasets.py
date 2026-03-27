"""
mqm_datasets.py
===============
Dataset loaders for the two benchmarks:

  1. IndicMT-Eval / IndicMQM  (en→hi)
     Source : AI4Bharat IndicMT-Eval   (Sai et al., ACL 2023)
     HuggingFace: ai4bharat/IndicMT-Eval  OR  ai4bharat/IN22-Gen (no MQM labels)
     The MQM-annotated en-hi split ships as a CSV/JSON in the GitHub release.
     We provide two loaders:
       a) load_indicmqm_hf()     – loads from HuggingFace (IN22-Gen, no MQM labels)
       b) load_indicmqm_local()  – loads from a local TSV/CSV with MQM annotations

  2. WMT en-de MQM  (used by HiMATE 2025)
     Source : google/wmt-mqm-human-evaluation  (WMT 2022 / 2023)
     HuggingFace: RicardoRei/wmt-mqm-human-evaluation
     The HiMATE paper uses the WMT 2022 en-de split (mqm_general_MT2022_ende.tsv)

Both loaders return a list of dicts with keys:
  {source, mt, reference, system, lp, domain, human_mqm_score, human_spans}
  human_spans may be None if the dataset doesn't include span-level annotations.

Quick-start
-----------
  from mqm_datasets import load_wmt_ende, load_indicmqm_hf
  samples = load_wmt_ende(split="2022", max_samples=50)
  for s in samples[:3]:
      print(s["source"][:80])
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# WMT en-de MQM  (used by HiMATE 2025)
# ---------------------------------------------------------------------------

def load_wmt_ende(
    split: str = "2022",
    max_samples: Optional[int] = None,
    system_filter: Optional[str] = None,
    domain_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load WMT en-de MQM data from HuggingFace (RicardoRei/wmt-mqm-human-evaluation).

    Parameters
    ----------
    split       : "2022" or "2023"  (year of the WMT shared task)
    max_samples : cap the number of returned samples (None = all)
    system_filter : keep only rows from this MT system (e.g. "refA")
    domain_filter : keep only rows from this domain (e.g. "news")

    Returns
    -------
    List of dicts, each with keys:
      source, mt, reference, system, lp, domain, human_mqm_score, human_spans
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train")

    # Filter to en-de and the requested year
    filtered = ds.filter(
        lambda ex: ex["lp"] == "en-de" and str(ex.get("year", "")) == str(split)
    )
    if system_filter:
        filtered = filtered.filter(lambda ex: ex.get("system", "") == system_filter)
    if domain_filter:
        filtered = filtered.filter(lambda ex: ex.get("domain", "") == domain_filter)

    # Deduplicate by (source, system) to get one row per sentence per system
    seen = set()
    samples: List[Dict[str, Any]] = []
    for row in filtered:
        key = (row["src"], row.get("system", ""))
        if key in seen:
            continue
        seen.add(key)
        samples.append({
            "source":          row["src"],
            "mt":              row["mt"],
            "reference":       row.get("ref", ""),
            "system":          row.get("system", ""),
            "lp":              "en-de",
            "domain":          row.get("domain", ""),
            "human_mqm_score": row.get("score", None),
            "human_spans":     None,   # span-level data lives in the TSV file, see below
        })
        if max_samples and len(samples) >= max_samples:
            break

    print(f"[load_wmt_ende] Loaded {len(samples)} en-de samples (WMT {split}).")
    return samples


def load_wmt_ende_tsv(
    tsv_path: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load from the original WMT MQM TSV files (include error spans).

    Download from:
      https://github.com/google/wmt-mqm-human-evaluation
      → mqm_general_MT2022_ende.tsv   (used by HiMATE 2025)

    TSV columns (tab-separated):
      system, doc, docid, seg_id, rater, source, target, category, severity

    Returns one entry per (system, seg_id) with aggregated spans.
    """
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"{tsv_path} not found.\n"
            "Download from: https://github.com/google/wmt-mqm-human-evaluation"
        )

    # # Group rows by (system, seg_id)
    # groups: Dict[tuple, Dict] = {}
    # with open(tsv_path, encoding="utf-8") as f:
    #     reader = csv.DictReader(f, delimiter="\t")
    #     for row in reader:
    #         key = (row["system"], row["seg_id"])
    #         if key not in groups:
    #             groups[key] = {
    #                 "source":    row["source"],
    #                 "mt":        row["target"],
    #                 "reference": "",        # TSV doesn't include reference
    #                 "system":    row["system"],
    #                 "lp":        "en-de",
    #                 "domain":    row.get("doc", ""),
    #                 "human_mqm_score": None,
    #                 "human_spans": [],
    #             }
    #         # Append span annotation (skip "no-error" rows)
    #         cat = row.get("category", "").strip()
    #         sev = row.get("severity", "").strip()
    #         if cat and cat.lower() not in ("", "no-error"):
    #             groups[key]["human_spans"].append({
    #                 "category": cat,
    #                 "severity": sev,
    #                 "rater":    row.get("rater", ""),
    #             })

    # samples = list(groups.values())
    # if max_samples:
    #     samples = samples[:max_samples]
    # print(f"[load_wmt_ende_tsv] Loaded {len(samples)} en-de samples from {tsv_path}.")
    # return samples
    

    # Group rows by (system, seg_id)
    groups = {}

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            key = (row["system"], row["seg_id"])

            if key not in groups:
                groups[key] = {
                    "source": row["source"],
                    "mt": row["target"],
                    "reference": None,
                    "system": row["system"],
                    "lp": "en-de",
                    "domain": row.get("doc", ""),
                    "human_spans": [],
                }

            cat = (row.get("category") or "").strip()
            sev = (row.get("severity") or "").strip()

            if cat and cat.lower() != "no-error":
                groups[key]["human_spans"].append({
                    "category": cat,
                    "severity": sev,
                    "rater": row.get("rater", ""),
                })

    samples = list(groups.values())
    if max_samples:
        samples = samples[:max_samples]

    return samples


# ---------------------------------------------------------------------------
# IndicMT-Eval / IndicMQM  (en→hi)
# ---------------------------------------------------------------------------

def load_indicmqm_hf(
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load en→hi sentence pairs from ai4bharat/IN22-Gen (HuggingFace).

    NOTE: IN22-Gen does NOT include MQM human annotations — it provides
    source + reference translations only. Use this to run the framework
    on real Indic MT data and generate automatic MQM scores.

    For MQM-annotated en-hi data, use load_indicmqm_local() after downloading
    the annotations from: https://github.com/AI4Bharat/IndicMT-Eval
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("ai4bharat/IN22-Gen", "eng_Latn-hin_Deva", split="gen")

    samples: List[Dict[str, Any]] = []
    for row in ds:
        samples.append({
            "source":          row["sentence_eng_Latn"],
            "mt":              row.get("sentence_hin_Deva", ""),   # reference acts as MT here
            "reference":       row.get("sentence_hin_Deva", ""),
            "system":          "reference",
            "lp":              "en-hi",
            "domain":          row.get("domain", ""),
            "human_mqm_score": None,
            "human_spans":     None,
        })
        if max_samples and len(samples) >= max_samples:
            break

    print(f"[load_indicmqm_hf] Loaded {len(samples)} en-hi samples from IN22-Gen.")
    return samples


def load_indicmqm_local(
    annotations_path: str,
    format: str = "csv",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load IndicMQM en-hi data from a local file downloaded from the
    AI4Bharat IndicMT-Eval repository.

    Download from: https://github.com/AI4Bharat/IndicMT-Eval
    Expected CSV columns:
      source, translation, reference, system, mqm_score, error_spans (JSON string)

    Parameters
    ----------
    annotations_path : path to the downloaded CSV/JSON file
    format           : "csv" or "json"
    max_samples      : cap returned samples

    Returns same schema as other loaders.
    """
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            f"{annotations_path} not found.\n"
            "Download MQM annotations from:\n"
            "  https://github.com/AI4Bharat/IndicMT-Eval\n"
            "Look for files tagged 'hindi' or 'en-hi' in the releases."
        )

    samples: List[Dict[str, Any]] = []

    if format == "json":
        with open(annotations_path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("data", [])
        for row in rows:
            samples.append({
                "source":          row.get("source", ""),
                "mt":              row.get("translation", row.get("mt", "")),
                "reference":       row.get("reference", ""),
                "system":          row.get("system", ""),
                "lp":              "en-hi",
                "domain":          row.get("domain", ""),
                "human_mqm_score": row.get("mqm_score", None),
                "human_spans":     row.get("error_spans", None),
            })
            if max_samples and len(samples) >= max_samples:
                break

    elif format == "csv":
        with open(annotations_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                spans = None
                if "error_spans" in row and row["error_spans"]:
                    try:
                        spans = json.loads(row["error_spans"])
                    except json.JSONDecodeError:
                        spans = None
                samples.append({
                    "source":          row.get("source", ""),
                    "mt":              row.get("translation", row.get("mt", "")),
                    "reference":       row.get("reference", ""),
                    "system":          row.get("system", ""),
                    "lp":              "en-hi",
                    "domain":          row.get("domain", ""),
                    "human_mqm_score": float(row["mqm_score"]) if row.get("mqm_score") else None,
                    "human_spans":     spans,
                })
                if max_samples and len(samples) >= max_samples:
                    break
    else:
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'json'.")

    print(f"[load_indicmqm_local] Loaded {len(samples)} en-hi samples from {annotations_path}.")
    return samples


# ---------------------------------------------------------------------------
# Generic TSV loader (handles any language pair in WMT format)
# ---------------------------------------------------------------------------

def load_mqm_tsv_generic(
    tsv_path: str,
    lp: str = "en-de",
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generic loader for any WMT-format MQM TSV file."""
    return load_wmt_ende_tsv(tsv_path, max_samples)
