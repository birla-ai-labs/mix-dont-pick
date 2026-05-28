"""
utils/preprocessing.py — Feature preprocessing for coverage metrics.

Pipeline:
  1. Subset to 24 triaged features (from 33 extracted)
  2. log1p(|x|) * sign(x) on 6 heavy-tailed features
  3. StandardScaler fit on REAL reference only
  4. Clip ±10

Triage: iteratively removed the lower-KS member of the highest-correlated
remaining pair (|ρ| > 0.80) until no violations remained. DN_Mean excluded
separately (model-invariant under instance normalization).
Condition number: 490,359 (full 33) -> 87 (triaged 24).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


TRIAGED_FEATURES = [
    # catch22 subset (17 of 22)
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_trev_1_num",
    "MD_hrv_classic_pnn40",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_OutlierInclude_n_001_mdrmd",
    "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_centroid",
    # catch24 addition (0) — DN_Spread_Std/DN_Mean dropped
    # targeted features (7) — hurst_dfa dropped via triage
    "lumpiness",
    "spectral_entropy",
    "trend_strength",
    "seasonal_strength",
    "arch_lm",
    "e_acf1",
    "kpss",
]

HEAVY_TAIL_FEATURES = [
    "lumpiness",                              # var_real = 5.25e52
    "arch_lm",                                # var_real = 360K
    "PD_PeriodicityWang_th0_01",              # var_real = 10.6K
    "CO_FirstMin_ac",                         # var_real = 9.8K
    "SB_BinaryStats_diff_longstretch0",       # var_real = 59.6
    "CO_Embed2_Dist_tau_d_expfit_meandiff",   # var_real = 3.2, heavy across generators
]

CLIP_VALUE = 10.0

N_TRIAGED = len(TRIAGED_FEATURES)


class FeaturePreprocessor:
    """Fit once on real reference, transform any corpus. Stateful."""

    def __init__(self):
        self.triage_idx: np.ndarray | None = None
        self.heavy_idx: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self.fitted: bool = False

    def fit(self, real_features: np.ndarray, feature_names: list[str]) -> "FeaturePreprocessor":
        feature_names = list(feature_names)
        missing = [f for f in TRIAGED_FEATURES if f not in feature_names]
        if missing:
            raise ValueError(f"Missing triaged features: {missing}")

        self.triage_idx = np.array(
            [feature_names.index(f) for f in TRIAGED_FEATURES]
        )
        self.heavy_idx = np.array(
            [TRIAGED_FEATURES.index(f) for f in HEAVY_TAIL_FEATURES]
        )

        x = self._subset_and_log1p(real_features)
        self.scaler = StandardScaler().fit(x)
        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() before transform()")
        x = self._subset_and_log1p(features)
        x = self.scaler.transform(x)
        return np.clip(x, -CLIP_VALUE, CLIP_VALUE)

    def fit_transform(self, real_features: np.ndarray, feature_names: list[str]) -> np.ndarray:
        self.fit(real_features, feature_names)
        return self.transform(real_features)

    def _subset_and_log1p(self, features: np.ndarray) -> np.ndarray:
        x = features[:, self.triage_idx].astype(np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = x[:, self.heavy_idx]
        x[:, self.heavy_idx] = np.sign(h) * np.log1p(np.abs(h))
        return x