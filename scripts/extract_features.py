#!/usr/bin/env python3
"""
scripts/extract_features.py — P2 Feature Space

Extract 33 features per series across all 12 corpora (11 synthetic + real).
Parallelized via multiprocessing fork + COW for 64-core / 120 GB machines.

Output
------
results/features/{corpus_name}.parquet   — one per corpus (N x 34: series_idx + 33 features)
results/features/nan_diagnostics.csv     — per-feature NaN/zero rates for appendix

Usage
-----
    python scripts/extract_features.py                            # all discovered corpora
    python scripts/extract_features.py --only sde arma real       # subset
    python scripts/extract_features.py --workers 48 --chunk 2000  # tune parallelism

Assumptions
-----------
- Synthetic corpora: data/synthetic_corpora/{name}.npy  shape (N, 1024)
- Real reference:    data/real_reference/{name}.npy      shape (N, 1024)
  Adjust CORPUS_DIRS below if your layout differs.
"""

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SYNTHETIC_CORPORA_DIR, FEATURES_DIR, REAL_REFERENCE_DIR
from utils.features import extract_features_single, FEATURE_NAMES, N_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


_CORPUS: np.ndarray | None = None

def _extract_batch(indices: np.ndarray) -> np.ndarray:
    """Worker: extract features for a batch of series indices."""
    out = np.empty((len(indices), N_FEATURES), dtype=np.float64)
    for i, idx in enumerate(indices):
        out[i] = extract_features_single(_CORPUS[idx])
    return out


def discover_corpora(seed: int = 42) -> dict[str, Path]:
    """Scan corpus dirs for series.npy files matching the target seed.

    Layout: {corpus_dir}/{generator}/seed_{seed}/series.npy

    Returns {name: path} sorted, e.g. {"arima": Path(...), "sde": Path(...)}
    """
    corpora = {}

    if SYNTHETIC_CORPORA_DIR.exists():
        for gen_dir in sorted(SYNTHETIC_CORPORA_DIR.iterdir()):
            if not gen_dir.is_dir():
                continue
            npy = gen_dir / f"seed_{seed}" / "series.npy"
            if npy.exists():
                corpora[gen_dir.name] = npy

    if REAL_REFERENCE_DIR.exists():
        npy = REAL_REFERENCE_DIR / f"seed_{seed}" / "series.npy"
        if npy.exists():
            corpora["real"] = npy
        else:
            for sub in sorted(REAL_REFERENCE_DIR.iterdir()):
                if not sub.is_dir():
                    continue
                npy = sub / f"seed_{seed}" / "series.npy"
                if npy.exists():
                    corpora[f"real_{sub.name}"] = npy

    return dict(sorted(corpora.items()))



def extract_corpus(
    name: str,
    path: Path,
    n_workers: int,
    chunk_size: int,
    subsample: int | None = None,
) -> pd.DataFrame:
    """Load one corpus, extract 33 features in parallel, return DataFrame."""
    global _CORPUS

    log.info("Loading %s from %s", name, path)
    t0 = time.perf_counter()
    _CORPUS = np.load(path, mmap_mode="r")
    n_series, seq_len = _CORPUS.shape

    if subsample and subsample < n_series:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n_series, size=subsample, replace=False))
        _CORPUS = _CORPUS[idx]  # triggers read from mmap into RAM — small now
        n_series = subsample
        log.info("  subsampled to %d series", n_series)

    mem_gb = _CORPUS.nbytes / 1e9
    log.info("  shape (%d, %d) | %.1f GB | loaded in %.1fs",
             n_series, seq_len, mem_gb, time.perf_counter() - t0)

    indices = np.arange(n_series)
    chunks = [indices[i : i + chunk_size] for i in range(0, n_series, chunk_size)]
    log.info("  %d chunks x ~%d series -> %d workers", len(chunks), chunk_size, n_workers)

    t1 = time.perf_counter()
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_extract_batch, chunks),
            total=len(chunks),
            desc=f"  {name}",
            unit="chunk",
            ncols=90,
        ))
    features = np.vstack(results)
    elapsed = time.perf_counter() - t1
    log.info("  done in %.1fs (%.0f series/s)", elapsed, n_series / elapsed)

    _CORPUS = None
    df = pd.DataFrame(features, columns=FEATURE_NAMES)
    df.insert(0, "series_idx", np.arange(n_series))
    return df


def nan_diagnostics(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-feature NaN/Inf/zero rates across all corpora."""
    rows = []
    for corpus_name, df in dfs.items():
        sub = df[FEATURE_NAMES]
        vals = sub.to_numpy()
        for j, feat in enumerate(FEATURE_NAMES):
            col = vals[:, j]
            rows.append({
                "corpus": corpus_name,
                "feature": feat,
                "nan_rate":  float(np.isnan(col).mean()),
                "inf_rate":  float(np.isinf(col).mean()),
                "zero_rate": float((col == 0.0).mean()),
                "mean":      float(np.nanmean(col)),
                "std":       float(np.nanstd(col)),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="P2: Extract 33 features from all corpora."
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Process only these corpus names (e.g. sde arma real_seed42)")
    parser.add_argument(
        "--workers", type=int, default=60,
        help="Parallel workers (default 60, leaving 4 cores for OS)")
    parser.add_argument(
        "--chunk", type=int, default=4000,
        help="Series per worker batch (default 4000 ≈ 250 chunks per 1M corpus)")
    parser.add_argument(
        "--outdir", type=str, default=str(FEATURES_DIR),
        help=f"Output directory (default {FEATURES_DIR})")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Which seed subdirectory to extract (default 42)")
    parser.add_argument(
        "--subsample", type=int, required=True,
        help="Randomly subsample N series per corpus before extraction")
    args = parser.parse_args()

    outdir = Path(args.outdir) / f"{args.seed}" / f"{args.subsample}"
    outdir.mkdir(parents=True, exist_ok=True)

    corpora = discover_corpora(seed=args.seed)
    if not corpora:
        log.error("No .npy files found")
        sys.exit(1)

    if args.only:
        corpora = {k: v for k, v in corpora.items() if k in args.only}
        if not corpora:
            log.error("--only filter matched nothing. Available: %s",
                      list(discover_corpora(seed=args.seed).keys()))
            sys.exit(1)

    log.info("=" * 60)
    log.info("P2 FEATURE EXTRACTION")
    log.info("=" * 60)
    log.info("Corpora (%d): %s", len(corpora), list(corpora.keys()))
    log.info("Workers: %d | Chunk: %d", args.workers, args.chunk)
    log.info("Output:  %s", outdir)
    log.info("-" * 60)

    dfs: dict[str, pd.DataFrame] = {}
    t_total = time.perf_counter()

    for name, path in corpora.items():
        out_path = outdir / f"{name}.parquet"

        if out_path.exists():
            log.info("SKIP %s — already exists (%s). Delete to re-extract.",
                     name, out_path)
            dfs[name] = pd.read_parquet(out_path)
            continue

        df = extract_corpus(name, path, args.workers, args.chunk, subsample=args.subsample)
        df.to_parquet(out_path, index=False, engine="pyarrow")
        log.info("  saved -> %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        dfs[name] = df

    # Diagnostics
    log.info("-" * 60)
    diag = nan_diagnostics(dfs)
    diag_path = outdir / "nan_diagnostics.csv"
    diag.to_csv(diag_path, index=False)
    log.info("NaN diagnostics -> %s", diag_path)

    # Summary table
    log.info("-" * 60)
    log.info("SUMMARY")
    for name, df in dfs.items():
        n = len(df)
        nan_count = df[FEATURE_NAMES].isna().sum().sum()
        log.info("  %-30s %8d series | %d NaN cells", name, n, nan_count)

    elapsed_total = time.perf_counter() - t_total
    log.info("=" * 60)
    log.info("DONE. %d corpora in %.1f min", len(dfs), elapsed_total / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()