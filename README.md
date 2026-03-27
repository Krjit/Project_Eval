<<<<<<< HEAD
# Project_Eval
=======
# MQM Framework v2 — Setup & Run Guide

## Files

| File | Purpose |
|------|---------|
| `mqm_models.py` | Pydantic models + LangGraph state (includes `ErrorSpan`) |
| `mqm_prompts.py` | All 25+ system prompts |
| `mqm_agents.py` | Stage 1/2/3 agent factories (Stage 2 now detects spans) |
| `mqm_aggregation.py` | Deterministic scoring |
| `mqm_pipeline.py` | Full LangGraph graph wiring |
| `mqm_datasets.py` | Dataset loaders for en-hi and en-de |
| `mqm_run.py` | Batch runner CLI |

---

## `max_rounds` explained

```
DEFAULT_MAX_ROUNDS = 2   (defined in mqm_models.py)
```

| Value | Meaning |
|-------|---------|
| `1` | Single pass — fastest, cheapest. ~18 LLM calls per sentence. |
| `2` | One re-run if the audit fires — **recommended default**. |
| `3` | Two possible re-runs — maximum recall, ~3× cost of round 1. |

Set it per-run:
```bash
python mqm_run.py --dataset wmt_ende --max_rounds 1   # speed mode
python mqm_run.py --dataset wmt_ende --max_rounds 2   # default
python mqm_run.py --dataset wmt_ende --max_rounds 3   # max recall
```

---

## Environment Setup

```bash
pip install -r requirements.txt
# echo "OPENAI_API_KEY=Your_API_Key" > .env
```

---

## Dataset 1: WMT en-de  (used by HiMATE 2025)

### Option A — HuggingFace (easiest, no labels download)
```bash
python mqm_run.py --dataset wmt_ende --split 2022 --max_samples 50
```

### Option B — Local TSV with error span annotations (HiMATE exact setup)
1. Download the TSV from the official Google repo:
   ```
   https://github.com/google/wmt-mqm-human-evaluation
   ```
   File needed: `mqm_general_MT2022_ende.tsv`

2. Place it in a `data/` folder:
   ```
   mkdir data
   mv mqm_general_MT2022_ende.tsv data/
   ```

3. Run:
   ```bash
   python mqm_run.py \
       --dataset wmt_ende_tsv \
       --tsv_path data/mqm_general_MT2022_ende.tsv \
       --max_samples 100
   ```

The TSV contains human-annotated error spans per segment (category + severity),
which lets you compute F1 score against the framework's predicted spans.

---

## Dataset 2: IndicMQM en-hi  (AI4Bharat)

### Option A — HuggingFace IN22-Gen (no MQM labels, but real en-hi sentences)
```bash
python mqm_run.py --dataset indic_hf --max_samples 50
```
This loads source + reference from IN22-Gen. Since no MT system output is
provided separately, the reference is used as MT. To evaluate a real MT system,
generate translations first (e.g. with IndicTrans2) and pass them as `mt`.

### Option B — Local IndicMT-Eval annotations (with MQM labels)
1. Clone the AI4Bharat repo:
   ```
   git clone https://github.com/AI4Bharat/IndicMT-Eval
   ```
2. Find the MQM-annotated Hindi file (typically in `data/` or `annotations/`).
3. Run:
   ```bash
   python mqm_run.py \
       --dataset indic_local \
       --annotations_path IndicMT-Eval/data/en_hi_mqm.csv \
       --max_samples 100
   ```

### Option C — Use IndicTrans2 to generate MT first
```python
# Generate MT with IndicTrans2, then run the framework
from mqm_datasets import load_indicmqm_hf
from mqm_run import run_batch

samples = load_indicmqm_hf(max_samples=50)

# Replace the 'mt' field with your MT system's output
for s in samples:
    s["mt"] = your_mt_system_translate(s["source"])   # your function

run_batch(samples, dataset_name="indic_en_hi_my_system")
```

---

## Error Span Output

Each result in the JSONL includes:
```json
{
  "predicted_error_spans": [
    {
      "start": 12,
      "end": 18,
      "span_text": "सौंदर्य",
      "error_type": "accuracy:mistranslation",
      "severity": "MINOR",
      "explanation": "The adjective form 'सौंदर्य' should be 'सौंदर्यात्मक' to match the adjectival usage in context."
    }
  ]
}
```

`mt[start:end]` always equals `span_text` — the framework validates this
at inference time and drops any span where the index doesn't match.

---

## Evaluating Span Detection (F1)

If you have human span annotations (e.g. from the WMT TSV):
```python
from mqm_run import extract_predicted_spans
import json

with open("results/wmt_ende_20250324_120000.jsonl") as f:
    results = [json.loads(l) for l in f]

for r in results:
    human_spans  = r["_meta"]["human_spans"]   # from TSV
    pred_spans   = r["predicted_error_spans"]  # from framework
    # compute overlap by category + severity
    # exact span match requires start/end alignment
```

---

## Reproducing HiMATE Comparison

The HiMATE paper (Zhang et al., EMNLP 2025) uses:
- Dataset: WMT 2022 en-de MQM (`mqm_general_MT2022_ende.tsv`)
- Metrics: segment-level Pearson/Spearman correlation + span F1

Run:
```bash
python mqm_run.py \
    --dataset wmt_ende_tsv \
    --tsv_path data/mqm_general_MT2022_ende.tsv \
    --max_samples 200 \
    --max_rounds 2
```
The summary JSON will report Pearson/Spearman against human MQM scores.
>>>>>>> Agentic_AI_MT_Eval
