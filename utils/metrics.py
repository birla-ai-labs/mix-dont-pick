"""
All functions except per-feature KS expect preprocessed arrays:
  log1p heavy-tails -> StandardScaler -> clip ±10.
Per-feature KS operates on raw 33-dim features (scale-invariant).
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from tqdm import tqdm

MAX_EVAL_N = 100_000


def _subsample(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(arr) <= n:
        return arr
    return arr[rng.choice(len(arr), size=n, replace=False)]


def _sym_sqrt(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, 0.0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def compute_per_feature_ks(
    real: np.ndarray,
    synth: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Two-sample KS per feature with Benjamini-Hochberg FDR correction.
    Runs on RAW features (scale-invariant), not preprocessed.
    """
    assert real.shape[1] == synth.shape[1] == len(feature_names)
    stats, pvals = [], []
    for j in range(real.shape[1]):
        r = real[:, j][np.isfinite(real[:, j])]
        s = synth[:, j][np.isfinite(synth[:, j])]
        if len(r) == 0 or len(s) == 0:
            stats.append(np.nan)
            pvals.append(1.0)
            continue
        res = ks_2samp(r, s)
        stats.append(float(res.statistic))
        pvals.append(float(res.pvalue))

    pvals_arr = np.array(pvals)
    reject, p_adj, _, _ = multipletests(pvals_arr, alpha=alpha, method="fdr_bh")
    return pd.DataFrame({
        "feature":    list(feature_names),
        "ks_stat":    stats,
        "pvalue_raw": pvals_arr,
        "pvalue_bh":  p_adj,
        "reject":     reject,
    })


def compute_all_coverage(
    real: np.ndarray,
    synth: np.ndarray,
    k: int = 5,
    seed: int = 42,
    batch_size: int = 2000,
) -> dict:
    """PRDC + α-β/authenticity/δ_centroid + FD — shared intermediates."""
    rng = np.random.default_rng(seed)
    real_s  = _subsample(real,  MAX_EVAL_N, rng)
    synth_s = _subsample(synth, MAX_EVAL_N, rng)
    n_real, n_synth = len(real_s), len(synth_s)

    nn_real  = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=50).fit(real_s)
    nn_synth = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=50).fit(synth_s)
    r_real  = nn_real.kneighbors(real_s)[0][:, -1]
    r_synth = nn_synth.kneighbors(synth_s)[0][:, -1]
    print(f"  Computed nearest neighbors (n={n_real} real, {n_synth} synth)")

    c_r = real_s.mean(0)
    c_s = synth_s.mean(0)

    prec_mask = np.zeros(n_synth, dtype=bool)
    density_count = np.zeros(n_synth, dtype=np.float64)

    for s in tqdm(range(0, n_synth, batch_size), desc="Precision/density batches"):
        e = min(s + batch_size, n_synth)
        d = cdist(real_s, synth_s[s:e])
        in_ball = d < r_real[:, None]
        prec_mask[s:e] = in_ball.any(axis=0)
        density_count[s:e] = in_ball.sum(axis=0)
        del d, in_ball

    precision = float(prec_mask.mean())
    density = float((density_count / k).mean())

    recall_mask = np.zeros(n_real, dtype=bool)
    for s in tqdm(range(0, n_real, batch_size), desc="Recall batches"):
        e = min(s + batch_size, n_real)
        d = cdist(real_s[s:e], synth_s)
        in_ball = d < r_synth[None, :]
        recall_mask[s:e] = in_ball.any(axis=1)
        del d, in_ball

    recall = float(recall_mask.mean())

    # coverage: reuse nn_synth
    d_rs = nn_synth.kneighbors(real_s, n_neighbors=1)[0]
    coverage = float((d_rs[:, 0] < r_real).mean())
    print(f"  Computed PRDC: precision={precision:.4f}, recall={recall:.4f}, "
          f"density={density:.4f}, coverage={coverage:.4f}")

    # ── α-β (reuse nn_real for authenticity) ────────────────────
    d_r_cr = np.linalg.norm(real_s  - c_r, axis=1)
    d_s_cs = np.linalg.norm(synth_s - c_s, axis=1)
    d_s_cr = np.linalg.norm(synth_s - c_r, axis=1)
    d_r_cs = np.linalg.norm(real_s  - c_s, axis=1)

    grid = np.linspace(0.0, 1.0, 31)[1:]
    r_quantiles = np.quantile(d_r_cr, grid)
    s_quantiles = np.quantile(d_s_cs, grid)

    alpha_hat = np.array([np.mean(d_s_cr <= rq) for rq in r_quantiles])
    beta_hat  = np.array([np.mean(d_r_cs <= sq) for sq in s_quantiles])

    alpha_precision = 1.0 - 2.0 * np.trapz(np.abs(grid - alpha_hat), grid)
    beta_recall     = 1.0 - 2.0 * np.trapz(np.abs(grid - beta_hat),  grid)

    # authenticity: reuse nn_real (already fit with k >= 2)
    real_nn_dist = nn_real.kneighbors(real_s, n_neighbors=2)[0][:, 1]
    d_sr, idx_sr = nn_real.kneighbors(synth_s, n_neighbors=1)
    authenticity = float(np.mean(d_sr[:, 0] > real_nn_dist[idx_sr[:, 0]]))

    delta_centroid = float(np.linalg.norm(c_r - c_s))
    print(f"  Computed α-β/authenticity/δ_centroid: "
          f"α_precision={alpha_precision:.4f}, beta_recall={beta_recall:.4f}, "
          f"authenticity={authenticity:.4f}, delta_centroid={delta_centroid:.4f}")

    # ── FD-stat (reuse centroids) ───────────────────────────────
    cov_r = np.cov(real_s,  rowvar=False)
    cov_s = np.cov(synth_s, rowvar=False)
    diff  = c_r - c_s
    cov_r_sqrt = _sym_sqrt(cov_r)
    inner = cov_r_sqrt @ cov_s @ cov_r_sqrt
    inner = (inner + inner.T) / 2.0
    fd_stat = float(diff @ diff + np.trace(cov_r + cov_s - 2.0 * _sym_sqrt(inner)))
    print(f"  Computed FD-stat: {fd_stat:.4f}")

    return {
        "precision": precision, "recall": recall,
        "density": density, "coverage": coverage,
        "alpha_precision": float(alpha_precision),
        "beta_recall": float(beta_recall),
        "authenticity": authenticity,
        "delta_centroid": delta_centroid,
        "fd_stat": fd_stat,
    }


def compute_null_baseline(
    real: np.ndarray,
    k: int = 5,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Real-vs-real: n random 50/50 splits -> mean ± std per metric."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_splits):
        idx  = rng.permutation(real.shape[0])
        half = len(idx) // 2
        a, b = real[idx[:half]], real[idx[half : 2 * half]]
        records.append(compute_all_coverage(a, b, k=k, seed=seed + i))

    keys = list(records[0].keys())
    out = {}
    for key in keys:
        vals = [r[key] for r in records]
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"]  = float(np.std(vals))
    return out