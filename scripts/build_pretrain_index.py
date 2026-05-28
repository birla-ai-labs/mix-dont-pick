"""
scripts/build_pretrain_index.py

Scans all datasets in GIFTEVAL_PRETRAIN_DATA_DIR and records:
  - n_series          : number of rows in the dataset (= number of series)
  - n_variates        : number of target variates (from first row)
  - mean_series_len   : mean length of first variate across a probe sample
  - min_series_len    : min observed in probe
  - max_series_len    : max observed in probe
  - total_obs         : n_series * mean_series_len (estimated)
  - freq              : from existing on-disk index
  - domain            : from existing on-disk index

Output: outputs/pretrain_index_v2.csv
"""
import os
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import numpy as np
import datasets
import pandas as pd

from config import GIFTEVAL_PRETRAIN_DATA_DIR, GIFTEVAL_PRETRAIN_HF_INDEX

N_PROBE = 20          # series to sample for length stats
OUTPUT = Path("outputs/pretrain_index_v2.csv")

def probe_dataset(ds_path: Path, n_probe: int) -> dict:
    ds = datasets.load_from_disk(str(ds_path))
    n_series = len(ds)

    idx = np.linspace(0, n_series - 1, min(n_probe, n_series), dtype=int).tolist()
    
    lengths = []
    n_variates = None
    for i in idx:
        row = ds[i]
        t = np.array(row["target"], dtype=np.float32)
        if t.ndim == 1:
            n_variates = n_variates or 1
            lengths.append(len(t))
        elif t.ndim == 2:
            n_variates = n_variates or t.shape[0]
            lengths.append(t.shape[1])

    mean_len = float(np.mean(lengths)) if lengths else 0.0
    min_len  = int(np.min(lengths))    if lengths else 0
    max_len  = int(np.max(lengths))    if lengths else 0

    return {
        "n_series":        n_series,
        "n_variates":      n_variates or 0,
        "mean_series_len": round(mean_len, 1),
        "min_series_len":  min_len,
        "max_series_len":  max_len,
        "total_obs":       int(n_series * mean_len),
    }


def main():
    old_index = pd.read_csv(GIFTEVAL_PRETRAIN_HF_INDEX)
    freq_map   = dict(zip(old_index["name"], old_index["freq"]))
    domain_map = dict(zip(old_index["name"], old_index["domain"]))

    pretrain_dir = Path(GIFTEVAL_PRETRAIN_DATA_DIR)
    ds_names = sorted(p.name for p in pretrain_dir.iterdir() if p.is_dir())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name", "freq", "domain",
        "n_series", "n_variates",
        "mean_series_len", "min_series_len", "max_series_len",
        "total_obs",
    ]

    rows = []
    for i, ds_name in enumerate(ds_names):
        ds_path = pretrain_dir / ds_name
        freq   = freq_map.get(ds_name, "?")
        domain = domain_map.get(ds_name, "?")

        if freq == "?" or domain == "?":
            print(f"  [{i+1}/{len(ds_names)}] {ds_name} — NOT IN INDEX, skipping")
            continue

        try:
            stats = probe_dataset(ds_path, N_PROBE)
            row = {"name": ds_name, "freq": freq, "domain": domain, **stats}
            rows.append(row)
            print(f"  [{i+1}/{len(ds_names)}] {ds_name}: "
                  f"{stats['n_series']} series × {stats['n_variates']} var, "
                  f"len={stats['mean_series_len']:.0f} (mean), "
                  f"total_obs={stats['total_obs']:,}")
        except Exception as e:
            print(f"  [{i+1}/{len(ds_names)}] {ds_name} — ERROR: {e}")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} datasets -> {OUTPUT}")

    # quick sanity: flag anything missing from old index
    old_names = set(old_index["name"])
    new_names = {r["name"] for r in rows}
    missing = old_names - new_names
    if missing:
        print(f"WARNING: {len(missing)} old-index datasets not found on disk: {missing}")


if __name__ == "__main__":
    main()