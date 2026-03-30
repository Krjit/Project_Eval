import json
import pandas as pd
from collections import Counter
import os
import traceback

from own_framework_pipeline import app


CSV_PATH = "Hindi_Indic_MQM_MT_data_Own_Cat_Map - All_Original.csv"
OUTPUT_PATH = "top5_error_match_results.csv"
SUMMARY_PATH = "top5_error_match_summary.json"
FAILURE_PATH = "top5_error_match_failures.csv"

MODEL_ERROR_KEYS = [
    "addition", "omission", "mistranslation", "untranslated_text", "transliteration", "non_translation",
    "punctuation", "spelling", "grammar", "register",
    "inconsistency", "characterEncoding",
    "inappropriate_for_context", "inconsistency_use", "awkward",
]

GOLD_TO_MODEL = {
    "Accuracy_Addition": "addition",
    "Accuracy_Omission": "omission",
    "Accuracy_Mistranslation": "mistranslation",
    "Accuracy_Untranslated_text": "untranslated_text",
    "Fluency_Grammar": "grammar",
    "Fluency_Register": "register",
    "Fluency_Spelling": "spelling",
    "Style_Awkward": "awkward",
    "Terminology_Inappropriate": "inappropriate_for_context",
    "Non-translation": "non_translation",
    "Transliteration": "transliteration",
}

IGNORE_GOLD = {"Default", "Other", "Source_error"}


def log(msg):
    print(msg, flush=True)


def serialize(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    return obj


def get_gold_errors(row):
    gold = []
    for i in range(1, 6):
        col = f"Error{i}_Type"
        if col not in row.index:
            continue
        label = str(row[col]).strip()
        if label in IGNORE_GOLD:
            continue
        mapped = GOLD_TO_MODEL.get(label)
        if mapped is not None:
            gold.append(mapped)
    return sorted(set(gold))


def get_top5_predictions(result):
    scored = []
    for key in MODEL_ERROR_KEYS:
        node = result.get(key)
        if node is None:
            continue

        prob = node.get("reEvaluatedProb", 0.0)
        conf = node.get("reEvaluatedConfidence", 0.0)
        score = prob * (conf / 100.0)

        scored.append((key, score, prob, conf))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:5], scored


def evaluate_row(row, result):
    gold = set(get_gold_errors(row))
    top5, all_scored = get_top5_predictions(result)

    pred_top5 = [x[0] for x in top5]
    pred_set = set(pred_top5)
    hits = sorted(gold & pred_set)

    recall = len(hits) / len(gold) if gold else None
    precision_at_5 = len(hits) / 5.0
    exact_match = pred_set == gold if gold else None
    exact_containment = gold.issubset(pred_set) if gold else None

    return {
        "gold_errors": sorted(gold),
        "top5_predicted": pred_top5,
        "hits": hits,
        "num_gold": len(gold),
        "num_hits": len(hits),
        "recall_at_5": recall,
        "precision_at_5": precision_at_5,
        "exact_match": exact_match,
        "exact_containment": exact_containment,
        "all_scores_json": json.dumps(
            [{"error": k, "score": s, "prob": p, "conf": c} for (k, s, p, c) in all_scored],
            ensure_ascii=False
        ),
    }


def append_csv_row(path, row_dict, write_header):
    pd.DataFrame([row_dict]).to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig"
    )


def main():
    log("Script started")
    log(f"Current working directory: {os.getcwd()}")
    log(f"Looking for CSV at: {os.path.abspath(CSV_PATH)}")

    if not os.path.exists(CSV_PATH):
        log("ERROR: CSV file not found.")
        return

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        log(f"ERROR while reading CSV: {e}")
        log(traceback.format_exc())
        return

    log(f"CSV loaded successfully. Rows: {len(df)}")
    log(f"Results will be written to: {os.path.abspath(OUTPUT_PATH)}")
    log(f"Failures will be written to: {os.path.abspath(FAILURE_PATH)}")

    per_error_total = Counter()
    per_error_hit = Counter()

    write_header_results = not os.path.exists(OUTPUT_PATH)
    write_header_failures = not os.path.exists(FAILURE_PATH)

    for idx, row in df.iterrows():
        log(f"\nStarting row {idx}")

        try:
            state = {
                "source": str(row["Source"]),
                "mt": str(row["Translation"]),
                "reference": str(row["Reference"]),
                "accuracyStage1_round": 0,
                "fluencyStage1_round": 0,
                "terminologyStage1_round": 0,
                "styleStage1_round": 0
            }

            log(f"Calling app.invoke for row {idx} ...")
            result = app.invoke(state)
            log(f"app.invoke finished for row {idx}")

            result = serialize(result)
            eval_out = evaluate_row(row, result)

            for g in eval_out["gold_errors"]:
                per_error_total[g] += 1
            for h in eval_out["hits"]:
                per_error_hit[h] += 1

            row_dict = {
                "row_id": idx,
                "Source": row["Source"],
                "Reference": row["Reference"],
                "Translation": row["Translation"],
                "gold_errors": json.dumps(eval_out["gold_errors"], ensure_ascii=False),
                "top5_predicted": json.dumps(eval_out["top5_predicted"], ensure_ascii=False),
                "hits": json.dumps(eval_out["hits"], ensure_ascii=False),
                "num_gold": eval_out["num_gold"],
                "num_hits": eval_out["num_hits"],
                "recall_at_5": eval_out["recall_at_5"],
                "precision_at_5": eval_out["precision_at_5"],
                "exact_match": eval_out["exact_match"],
                "exact_containment": eval_out["exact_containment"],
                "all_scores_json": eval_out["all_scores_json"],
            }

            append_csv_row(OUTPUT_PATH, row_dict, write_header_results)
            write_header_results = False

            log(f"Saved row {idx}")

        except Exception as e:
            log(f"FAILED row {idx}: {e}")
            log(traceback.format_exc())

            fail_row = {
                "row_id": idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            append_csv_row(FAILURE_PATH, fail_row, write_header_failures)
            write_header_failures = False

    if os.path.exists(OUTPUT_PATH):
        try:
            out_df = pd.read_csv(OUTPUT_PATH)
            valid = out_df[out_df["num_gold"] > 0]

            summary = {
                "rows_total": int(len(out_df)),
                "rows_with_gold_errors": int(len(valid)),
                "hit_at_5": float((valid["num_hits"] > 0).mean()) if len(valid) else None,
                "mean_recall_at_5": float(valid["recall_at_5"].mean()) if len(valid) else None,
                "mean_precision_at_5": float(valid["precision_at_5"].mean()) if len(valid) else None,
                "exact_match_rate": float(valid["exact_match"].mean()) if len(valid) else None,
                "exact_containment_rate": float(valid["exact_containment"].mean()) if len(valid) else None,
                "per_error_recall": {
                    err: (per_error_hit[err] / per_error_total[err]) if per_error_total[err] else None
                    for err in sorted(per_error_total)
                }
            }

            with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            log(f"\nSummary saved to: {os.path.abspath(SUMMARY_PATH)}")
        except Exception as e:
            log(f"Could not create summary: {e}")
            log(traceback.format_exc())

    log("Script finished")


if __name__ == "__main__":
    main()