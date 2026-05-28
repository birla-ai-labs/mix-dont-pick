# data/fev_bench

fev-bench data for the Coverage-to-Utility study.

**Reference:** Shchur et al. (2025), "fev: A Realistic Benchmark for Time Series Forecasting." https://arxiv.org/abs/2509.26468

## Setup

```bash
pip install -U huggingface_hub
huggingface-cli login
```

## Download benchmark CSV results -> `benchmark_results/`

```bash
git clone --depth 1 https://github.com/autogluon/fev /tmp/fev

mkdir -p data/fev_bench/benchmark_results
cp /tmp/fev/benchmarks/fev_bench/results/*.csv data/fev_bench/benchmark_results/

rm -rf /tmp/fev
```

## Download leaderboard tables -> `leaderboard_results/`

```bash
tmp_dir="$(mktemp -d)"
hf download autogluon/fev-bench \
  --repo-type space \
  --include "tables/**" \
  --local-dir data/fev_bench/leaderboard_results

mkdir -p data/fev_bench/leaderboard_results
rsync -a "$tmp_dir/tables/" data/fev_bench/leaderboard_results/tables/
rm -rf "$tmp_dir"
```

## Download evaluation datasets -> `eval_data/`

Large. Only needed for Stage 7 (fev-bench replication). Use external storage and set `FEV_EVAL_DATA_DIR` in `.env`.

```bash
tmp_dir="$(mktemp -d)"
huggingface-cli download autogluon/fev_datasets \
  --repo-type dataset \
  --local-dir "$tmp_dir" \
  --local-dir-use-symlinks False

mkdir -p data/fev_bench/eval_data
rsync -a "$tmp_dir"/ data/fev_bench/eval_data/
rm -rf "$tmp_dir"
```

## Update tasks list

Already committed as `tasks.yaml`. To pull latest:

```bash
curl -o data/fev_bench/tasks.yaml \
  https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml
```

Verify `dataset_fingerprint` matches benchmark CSVs when comparing.

## Committed metadata

| File | Source |
|---|---|
| `tasks.yaml` | fev GitHub repo |
| `fev_bench_leakage.csv` | Derived from fev-bench + model papers |
