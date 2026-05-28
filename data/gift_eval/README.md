# data/gift_eval

GIFT-Eval data for the Coverage-to-Utility study.

**Reference:** Aksu et al. (2024), "GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation." https://arxiv.org/abs/2410.10393

## Setup

```bash
pip install -U huggingface_hub
huggingface-cli login
```

## Download benchmark results -> `benchmark_results/`

```bash
tmp_dir="$(mktemp -d)"
hf download Salesforce/GIFT-Eval \
  --repo-type space \
  --include "results/**" \
  --local-dir data/gift_eval/benchmark_results

mkdir -p data/gift_eval/benchmark_results
rsync -a "$tmp_dir/results/" data/gift_eval/benchmark_results/
rm -rf "$tmp_dir"
```

## Download evaluation datasets -> `eval_data/`

```bash
tmp_dir="$(mktemp -d)"
hf download Salesforce/GiftEval \
  --repo-type dataset \
  --local-dir data/gift_eval/eval_data

mkdir -p data/gift_eval/eval_data
rsync -a "$tmp_dir"/ data/gift_eval/eval_data/
rm -rf "$tmp_dir"
```

## Download pretraining data -> `pretrain_data/`

Large (~230B data points). Use external storage and set `GIFTEVAL_PRETRAIN_DATA_DIR` in `.env`.

```bash
tmp_dir="$(mktemp -d)"
hf download Salesforce/GiftEval \
  --repo-type dataset \
  --include "pretrain/**" \
  --local-dir data/gift_eval/pretrain_data

mkdir -p data/gift_eval/pretrain_data
rsync -a "$tmp_dir/pretrain/" data/gift_eval/pretrain_data/
rm -rf "$tmp_dir"
```

## Committed metadata

| File | Source |
|---|---|
| `pretrain_metadata.csv` | Table 14, Aksu et al. (2024) |
| `eval_metadata.csv` | Table 13, Aksu et al. (2024) |
| `dataset_mapping_v1.csv` | Derived from the above two |
| `pretrain_data_info.csv` | Contains dataset names from HF index |
| `build_metadata.py` | Generates the CSVs above |
