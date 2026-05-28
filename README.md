# Mix, Don't Pick: Synthetic Corpus Composition for Time-Series Foundation Model Pretraining

Code and results for the paper **"Mix, Don't Pick: Why Synthetic Corpus Composition Matters
for Time Series Foundation Model Pretraining"**, presented at the Foundation Models for
Structured Data Workshop, ICML 2026.

> **TL;DR:** Under identical training budgets, the best and worst synthetic generators
> produce a 2× gap in forecasting error; but a simple equal-weight mixture of all
> generators matches or beats the best individual generator for both Chronos-T5-Mini and
> Moirai-Small. Synthetic pretraining is a corpus composition problem, not a generator
> selection problem.

## Results

Precomputed results for all conditions are in `results/`. The analysis notebook
`analysis.ipynb` reads from `results/tstr/` and reproduces all tables in the paper.

## Setup

```bash
git clone https://github.com/your-org/mix-dont-pick.git
cd mix-dont-pick
pip install -r requirements.txt
cp .env.example .env  # then fill in your paths
```

## Reproducing the Experiments

### 1. Generate synthetic corpora

```bash
python scripts/generate_corpora.py
```

This produces one Arrow corpus per generator (11 total) plus the Mixed11 mixture,
written to `$DATA_ROOT/arrow/`.

### 2. Build the real reference corpus

```bash
python scripts/build_pretrain_index.py
python scripts/build_real_reference.py
```

Requires the GIFT-Eval pretraining dataset. See `data/gift_eval/README.md` for setup.

### 3. Train

**Chronos-T5-Mini** (uses HuggingFace Trainer, configs in `configs/`):

```bash
bash run_chronos_sequential.sh
```

**Moirai-Small** (uses uni2ts CLI, patches in `uni2ts_patches/`):

```bash
bash run_moirai_queue.sh
```

Apply the patches in `uni2ts_patches/` to a fresh clone of
[uni2ts](https://github.com/SalesforceAIResearch/uni2ts) before running.
See `uni2ts_patches/CHANGES.md` for exactly what was changed.

### 4. Evaluate

```bash
python scripts/run_chronos_eval.py --gpu 0
python scripts/run_moirai_eval.py --gpu 0
```

Results are written to `results/tstr/`.

### 5. Reproduce paper tables

Open `analysis.ipynb` and run all cells. Outputs go to `results/fmsd_tables/`.

## Generators

| Name | Family |
|---|---|
| ARIMA | Linear statistical |
| ETS | Exponential smoothing |
| fBm | Stochastic process |
| SDE | Regime-switching OU process |
| GARCH | Volatility model |
| KernelSynth | Gaussian process |
| Chaotic | Lorenz / Mackey–Glass |
| StepFunction | Piecewise constant |
| Waveform | Non-smooth periodic |
| TimeSynth | Composite signal |
| TSI | Trend-seasonality-irregularity |

## Project Structure

```
generators/          # 11 synthetic generator implementations
scripts/             # one entry point per pipeline stage
configs/             # per-condition Chronos training configs
uni2ts_patches/      # modifications to uni2ts for Moirai training
utils/               # evaluation and preprocessing utilities
data/                # GIFT-Eval metadata and index files
results/
  tstr/              # raw per-seed evaluation results
  fmsd_tables/       # final tables as in the paper
analysis.ipynb       # reproduces all paper tables from results/
```

## Citation

```bibtex
@inproceedings{anonymous2026mix,
  title={Mix, Don{\textquoteright}t Pick: Why Synthetic Corpus Composition Matters for Time Series Foundation Model Pretraining},
  author={Anonymous},
  booktitle={2nd ICML Workshop on Foundation Models for Structured Data},
  year={2026},
  url={https://openreview.net/forum?id=ywumgXC5bh}
}
```
