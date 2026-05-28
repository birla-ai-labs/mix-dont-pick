"""
Build stratified real reference corpora from GIFT-Eval pretrain data.

Produces corpora (seeds 42-44), each 1M univariate windows of length 1024,
stratified by (freq, domain) with floors, per-dataset caps, and supply ceilings.
Multivariate datasets: first target variate only.
Long-series datasets: up to MAX_WINDOWS_PER_SERIES non-overlapping windows per source series.
"""
import gc
import os
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf_cache"

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count


from config import GIFTEVAL_PRETRAIN_DATA_DIR, GIFTEVAL_PRETRAIN_INDEX, REAL_REFERENCE_DIR
from utils.sampling import TSSampler
datasets.disable_caching()

TOTAL_SAMPLE       = 1_000_000
LENGTH             = 1024
SEEDS              = [42, 43, 44]
OUTPUT_DIR         = REAL_REFERENCE_DIR
FLOOR_PER_STRATUM  = 10_000
MAX_DATASET_SHARE  = 0.50
MAX_WINDOWS_PER_SERIES = 8
WINDOWED_THRESHOLD = LENGTH   # 8192
MIN_VALID          = 128
OVERSAMPLE_FACTOR  = 3
WORKERS           = min(25, cpu_count())
# will take 200GB RAM (at max) with 25 workers


def clean_series(series: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Trim trailing NaN, ffill/bfill interior NaN, return (padded, valid_len)."""
    valid_mask = ~np.isnan(series)
    if not valid_mask.any():
        return None, 0

    last_valid = int(np.where(valid_mask)[0][-1])
    portion = series[:last_valid + 1].copy()

    if np.isnan(portion).any():
        idx = np.where(~np.isnan(portion), np.arange(len(portion)), 0)
        np.maximum.accumulate(idx, out=idx)
        portion = portion[idx]
        if np.isnan(portion).any():
            idx = np.where(~np.isnan(portion),
                           np.arange(len(portion)), len(portion) - 1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            portion = portion[idx]

    valid_len = len(portion)
    if valid_len < MIN_VALID:
        return None, 0

    padded = np.full(LENGTH, np.nan, dtype=np.float32)
    padded[:valid_len] = portion.astype(np.float32)
    return padded, valid_len


def _dataset_supply(n_series: int, mean_len: float) -> tuple[bool, int, int]:
    """Return (use_windowed, max_windows, total_supply) for a dataset."""
    use_windowed = mean_len > WINDOWED_THRESHOLD
    max_windows  = min(max(1, int(mean_len // LENGTH)), MAX_WINDOWS_PER_SERIES)
    supply       = n_series * max_windows if use_windowed else n_series
    return use_windowed, max_windows, supply


def build_allocation(
    pretrain_info: pd.DataFrame,
    total: int,
) -> dict[str, dict]:
    """Return per-dataset allocation dicts summing to exactly `total`."""

    # 1. Drop datasets too short to yield a single window
    viable = pretrain_info[pretrain_info["mean_series_len"] >= LENGTH].copy()
    dropped = pretrain_info[pretrain_info["mean_series_len"] < LENGTH]
    if len(dropped):
        print(f"  Dropped {len(dropped)} datasets with mean_series_len < {LENGTH}:")
        for _, row in dropped.iterrows():
            print(f"    {row['name']}: mean_len={row['mean_series_len']}")

    # 2. Stratum-level allocation proportional to total_obs, with floor
    group_obs   = viable.groupby(["freq", "domain"])["total_obs"].sum()
    group_props = group_obs / group_obs.sum()
    raw_alloc   = (group_props * total).clip(lower=FLOOR_PER_STRATUM)
    scale       = total / raw_alloc.sum()
    group_alloc = (raw_alloc * scale).astype(int)

    # 3. Compute per-stratum supply ceiling and cap/redistribute
    stratum_supply = {}
    for (freq, domain) in group_alloc.index:
        sub = viable[(viable.freq == freq) & (viable.domain == domain)]
        supply = sum(
            _dataset_supply(int(r.n_series), float(r.mean_series_len))[2]
            for _, r in sub.iterrows()
        )
        stratum_supply[(freq, domain)] = supply

    for _ in range(10):
        surplus = 0
        for key in group_alloc.index:
            if group_alloc[key] > stratum_supply[key]:
                surplus += group_alloc[key] - stratum_supply[key]
                group_alloc[key] = stratum_supply[key]
        if surplus == 0:
            break
        uncapped = {k: group_alloc[k] for k in group_alloc.index
                    if group_alloc[k] < stratum_supply[k]}
        if not uncapped:
            break
        total_uncapped = sum(uncapped.values())
        for key, val in uncapped.items():
            bump = int(surplus * val / total_uncapped)
            group_alloc[key] = min(group_alloc[key] + bump, stratum_supply[key])

    # distribute rounding remainder by largest fractional part
    deficit = total - group_alloc.sum()
    if deficit > 0:
        fracs = (raw_alloc * scale) - group_alloc
        for key in fracs.nlargest(deficit).index:
            group_alloc[key] += 1

    # 4. Within-stratum allocation with share cap + supply cap
    sampling_dict: dict[str, dict] = {}

    for (freq, domain), n_group in group_alloc.items():
        if n_group == 0:
            continue

        sub = viable[(viable.freq == freq) & (viable.domain == domain)].copy()
        if sub.empty:
            continue

        sub = sub.set_index("name")
        props     = sub["total_obs"] / sub["total_obs"].sum()
        ds_names  = list(props.index)

        use_windowed = {}
        max_windows  = {}
        supply       = {}
        for ds in ds_names:
            uw, mw, sup = _dataset_supply(int(sub.loc[ds, "n_series"]),
                                          float(sub.loc[ds, "mean_series_len"]))
            use_windowed[ds] = uw
            max_windows[ds]  = mw
            supply[ds]       = sup

        share_cap = int(n_group * MAX_DATASET_SHARE)
        alloc     = (props.values * n_group).astype(float)

        for _ in range(10):
            capped  = np.array([min(alloc[i], share_cap, supply[ds_names[i]])
                                for i in range(len(alloc))], dtype=float)
            surplus = alloc.sum() - capped.sum()
            if surplus < 1.0:
                alloc = capped
                break
            # which datasets still have headroom?
            free = np.array([capped[i] < min(share_cap, supply[ds_names[i]])
                             for i in range(len(alloc))])
            if not free.any():
                alloc = capped
                break
            free_props = props.values * free
            free_props /= free_props.sum()
            alloc = capped + surplus * free_props

        floored = np.minimum(np.floor(alloc).astype(int),
                             [supply[ds] for ds in ds_names])

        # distribute remainder by fractional part, respecting supply
        n_rem = int(n_group - floored.sum())
        if n_rem > 0:
            order = np.argsort(-(alloc - floored))
            for i in order:
                if n_rem == 0:
                    break
                headroom = supply[ds_names[i]] - floored[i]
                if headroom <= 0:
                    continue
                add = min(n_rem, headroom)
                floored[i] += add
                n_rem -= add

        for i, ds in enumerate(ds_names):
            if floored[i] == 0:
                continue
            prop_frac   = float(props.values[i]) * (n_group / total)
            actual_frac = floored[i] / total
            sampling_dict[ds] = {
                "count":        int(floored[i]),
                "freq":         freq,
                "domain":       domain,
                "prop_share":   prop_frac,
                "actual_share": actual_frac,
                "weight":       prop_frac / actual_frac if actual_frac > 0 else 1.0,
                "use_windowed": use_windowed[ds],
                "max_windows":  max_windows[ds],
                "n_series":     int(sub.loc[ds, "n_series"]),
            }

    # 5. Final top-up across all datasets (rounding may leave small deficit)
    deficit = total - sum(v["count"] for v in sampling_dict.values())
    if deficit > 0:
        can_bump = [
            (ds, _dataset_supply(info["n_series"],
                                 float(pretrain_info.loc[
                                     pretrain_info["name"] == ds,
                                     "mean_series_len"].iloc[0]))[2] - info["count"])
            for ds, info in sampling_dict.items()
        ]
        can_bump = [(ds, h) for ds, h in can_bump if h > 0]
        can_bump.sort(key=lambda x: -x[1])
        for ds, headroom in can_bump:
            if deficit <= 0:
                break
            add = min(deficit, headroom)
            sampling_dict[ds]["count"] += add
            deficit -= add

    return sampling_dict


def _sample_dataset(args: tuple) -> tuple:
    """Sample one dataset and write directly to memmap slice."""
    ds_name, info, pretrain_dir, seed, mm_path, write_offset = args

    ds_path = Path(pretrain_dir) / ds_name
    if not ds_path.exists():
        return ds_name, [], info["count"], 0

    datasets.disable_caching()
    sampler  = TSSampler(seed=seed)
    ds       = datasets.load_from_disk(str(ds_path))
    count    = info["count"]

    # open memmap in r+ mode — file already exists, just write our slice
    series_mm = np.lib.format.open_memmap(mm_path, mode="r+")

    meta_out  = []
    rejected  = 0
    collected = 0

    supply_cap    = (info["n_series"] * info["max_windows"]) / info["count"]
    factor        = min(OVERSAMPLE_FACTOR, supply_cap)

    windows = sampler.sample_windows(
        ds,
        total_windows=int(count * factor),
        window_size=LENGTH,
        max_windows_per_series=info["max_windows"],
    )

    for w_idx, window in enumerate(windows):
        if collected >= count:
            break
        cleaned, valid_len = clean_series(window)
        if cleaned is None:
            rejected += 1
            continue
        series_mm[write_offset + collected] = cleaned
        collected += 1
        meta_out.append({
            "dataset":         ds_name,
            "freq":            info["freq"],
            "domain":          info["domain"],
            "valid_length":    valid_len,
            "window_index":    w_idx,
            "sampling_weight": info["weight"],
        })

    series_mm.flush()
    del series_mm, windows
    gc.collect()

    shortfall = max(0, count - collected)
    return ds_name, meta_out, shortfall, rejected


def sample_one_seed(
    seed: int,
    pretrain_info: pd.DataFrame,
    pretrain_dir: Path,
) -> None:
    rng        = np.random.default_rng(seed)
    allocation = build_allocation(pretrain_info, TOTAL_SAMPLE)

    print(f"\nSeed {seed}: {sum(v['count'] for v in allocation.values())} windows "
          f"across {len(allocation)} datasets "
          f"({sum(v['use_windowed'] for v in allocation.values())} windowed)")

    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    mm_path = str(seed_dir / "series_opt.npy")

    series_mm = np.lib.format.open_memmap(
        mm_path, mode="w+", dtype=np.float32, shape=(TOTAL_SAMPLE, LENGTH)
    )
    series_mm.flush()
    del series_mm

    sorted_datasets = sorted(allocation.items(), key=lambda x: -x[1]["actual_share"])
    offsets = {}
    pos = 0
    for ds, info in sorted_datasets:
        offsets[ds] = pos
        pos += info["count"]

    args_list = [
        (ds, info, str(pretrain_dir), seed * 100_000 + i, mm_path, offsets[ds])
        for i, (ds, info) in enumerate(sorted_datasets)
    ]

    all_meta: list[dict] = []
    stats: dict[str, dict] = {}

    with Pool(processes=WORKERS) as pool:
        for ds_name, meta_out, shortfall, rejected in pool.imap_unordered(
            _sample_dataset, args_list
        ):
            all_meta.extend(meta_out)
            stats[ds_name] = {
                "allocated":    allocation[ds_name]["count"],
                "collected":    len(meta_out),
                "shortfall":    shortfall,
                "rejected":     rejected,
                "backfill":     0,
                "use_windowed": allocation[ds_name]["use_windowed"],
            }
            print(f"  {ds_name}: {len(meta_out)}/{allocation[ds_name]['count']}"
                  + (" !" if shortfall else ""))

    total_shortfall = sum(s["shortfall"] for s in stats.values())
    if total_shortfall > 0:
        print(f"\n  Backfilling {total_shortfall} windows...")

        backfill_positions = []
        for ds_name, info in sorted_datasets:
            collected = stats[ds_name]["collected"]
            allocated = info["count"]
            if collected < allocated:
                start = offsets[ds_name] + collected
                end   = offsets[ds_name] + allocated
                backfill_positions.extend(range(start, end))

        series_mm = np.lib.format.open_memmap(mm_path, mode="r+")
        fill_idx  = 0  # pointer into backfill_positions
        filled    = 0

        # Pass 1: retry from the same dataset that had the shortfall
        for ds_name in [d for d in stats if stats[d]["shortfall"] > 0]:
            needed  = stats[ds_name]["shortfall"]
            info    = allocation[ds_name]
            ds_path = pretrain_dir / ds_name
            if not ds_path.exists():
                continue

            ds       = datasets.load_from_disk(str(ds_path))
            attempts = 0
            got      = 0
            while got < needed and attempts < needed * 20:
                attempts += 1
                try:
                    target = np.asarray(ds[int(rng.integers(0, len(ds)))]["target"],
                                        dtype=np.float32)
                    if target.ndim == 2:
                        target = target[0]
                    if len(target) < LENGTH:
                        continue
                    start = int(rng.integers(0, len(target) - LENGTH + 1))
                    raw   = target[start:start + LENGTH]
                except Exception:
                    continue
                cleaned, valid_len = clean_series(raw)
                if cleaned is None:
                    continue
                series_mm[backfill_positions[fill_idx]] = cleaned
                fill_idx += 1
                all_meta.append({
                    "dataset": ds_name, "freq": info["freq"],
                    "domain": info["domain"], "valid_length": valid_len,
                    "window_index": -1, "sampling_weight": info["weight"],
                })
                stats[ds_name]["backfill"]  += 1
                stats[ds_name]["shortfall"] -= 1
                got     += 1
                filled  += 1
            del ds
            gc.collect()

        # Pass 2: uniform draw across all datasets for remaining gaps
        remaining = total_shortfall - filled
        if remaining > 0:
            candidates = [(ds, info) for ds, info in allocation.items()
                          if (pretrain_dir / ds).exists()]
            probs = np.ones(len(candidates)) / len(candidates)
            draw_ds_idx = rng.choice(len(candidates), size=remaining * 5, p=probs)
            draws_by_ds: dict[str, list] = {}
            for idx in draw_ds_idx:
                draws_by_ds.setdefault(candidates[idx][0], []).append(idx)

            for ds_name, _ in draws_by_ds.items():
                if filled >= total_shortfall:
                    break
                info  = allocation[ds_name]
                ds    = datasets.load_from_disk(str(pretrain_dir / ds_name))
                need  = total_shortfall - filled
                rows  = np.unique(rng.integers(0, len(ds), size=need * 4))
                batch = ds[rows.tolist()]
                for raw_target in batch["target"]:
                    if filled >= total_shortfall:
                        break
                    try:
                        target = np.asarray(raw_target, dtype=np.float32)
                        if target.ndim == 2:
                            target = target[0]
                        if len(target) < LENGTH:
                            continue
                        start = int(rng.integers(0, len(target) - LENGTH + 1))
                        raw   = target[start:start + LENGTH]
                    except Exception:
                        continue
                    cleaned, valid_len = clean_series(raw)
                    if cleaned is None:
                        continue
                    series_mm[backfill_positions[fill_idx]] = cleaned
                    fill_idx += 1
                    all_meta.append({
                        "dataset": ds_name, "freq": info["freq"],
                        "domain": info["domain"], "valid_length": valid_len,
                        "window_index": -1, "sampling_weight": info["weight"],
                    })
                    stats[ds_name]["backfill"] += 1
                    filled += 1
                del ds
                gc.collect()

        series_mm.flush()
        del series_mm
        print(f"  Backfilled: {filled}/{total_shortfall}")

    else:
        pass

    # Sort all_meta by write position so metadata rows match memmap rows
    # Workers wrote in parallel so meta is unordered — sort by (dataset, window_index)
    all_meta_sorted = sorted(
        all_meta,
        key=lambda m: (offsets[m["dataset"]], m["window_index"] if m["window_index"] >= 0 else 999999)
    )

    with open(seed_dir / "metadata.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "freq", "domain", "valid_length",
            "window_index", "sampling_weight",
        ])
        writer.writeheader()
        writer.writerows(all_meta_sorted[:TOTAL_SAMPLE])

    # Summary
    print(f"\n{'Dataset':<45} {'Alloc':>7} {'Got':>7} {'Short':>7} "
          f"{'Reject':>7} {'Backfill':>9} {'Mode'}")
    print("-" * 97)
    totals = dict(allocated=0, collected=0, shortfall=0, rejected=0, backfill=0)
    for ds, s in sorted(stats.items()):
        for k in totals:
            totals[k] += s[k]
        mode = "windowed" if s["use_windowed"] else "single"
        flag = " !" if s["shortfall"] > 0 else ""
        print(f"  {ds:<43} {s['allocated']:>7,} {s['collected']:>7,} "
              f"{s['shortfall']:>7,} {s['rejected']:>7,} {s['backfill']:>9,} {mode}{flag}")
    print("-" * 97)
    print(f"  {'TOTAL':<43} {totals['allocated']:>7,} {totals['collected']:>7,} "
          f"{totals['shortfall']:>7,} {totals['rejected']:>7,} {totals['backfill']:>9,}")

    vl = np.array([m["valid_length"] for m in all_meta_sorted[:TOTAL_SAMPLE]])
    print(f"\n  Valid length: min={vl.min()}, median={int(np.median(vl))}, "
          f"full-1024={(vl == LENGTH).sum():,}, shorter={(vl < LENGTH).sum():,}")
    print(f"  Saved: ({TOTAL_SAMPLE}, {LENGTH}) -> {seed_dir}")
    if fill_idx != total_shortfall:
        print(f"  WARNING: {total_shortfall - fill_idx} slots remain zero")


def main() -> None:
    pretrain_info = pd.read_csv(GIFTEVAL_PRETRAIN_INDEX)
    for col in ("name", "freq", "domain", "n_series", "mean_series_len", "total_obs"):
        assert col in pretrain_info.columns, f"Missing column: {col}"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        sample_one_seed(seed, pretrain_info, GIFTEVAL_PRETRAIN_DATA_DIR)


if __name__ == "__main__":
    main()