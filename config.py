"""Config file for the study.

Paths resolve in this order:
    1. Environment variable from .env (via python-dotenv)
    2. Default relative path under the repo
"""

import json
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Data directories

DATADIR = Path("data")

# GIFT-Eval
GIFTEVAL_DIR = DATADIR / "gift_eval"
GIFTEVAL_EVAL_DATA_DIR = Path(
    os.getenv("GIFTEVAL_EVAL_DATA_DIR", str(GIFTEVAL_DIR / "eval_data"))
)
GIFTEVAL_PRETRAIN_DATA_DIR = Path(
    os.getenv("GIFTEVAL_PRETRAIN_DATA_DIR", str(GIFTEVAL_DIR / "pretrain_data"))
)
with Path.open(GIFTEVAL_DIR / "eval_dataset_properties.json") as f:
    GIFTEVAL_DATASETS_PROPERTIES = json.load(f)

SHORT_GIFTEVAL_DATASETS = [
    "m4_yearly",
    "m4_quarterly",
    "m4_monthly",
    "m4_weekly",
    "m4_daily",
    "m4_hourly",
    "electricity/15T",
    "electricity/H",
    "electricity/D",
    "electricity/W",
    "solar/10T",
    "solar/H",
    "solar/D",
    "solar/W",
    "hospital",
    "covid_deaths",
    "us_births/D",
    "us_births/M",
    "us_births/W",
    "saugeenday/D",
    "saugeenday/M",
    "saugeenday/W",
    "temperature_rain_with_missing",
    "kdd_cup_2018_with_missing/H",
    "kdd_cup_2018_with_missing/D",
    "car_parts_with_missing",
    "restaurant",
    "hierarchical_sales/D",
    "hierarchical_sales/W",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
    "SZ_TAXI/15T",
    "SZ_TAXI/H",
    "M_DENSE/H",
    "M_DENSE/D",
    "ett1/15T",
    "ett1/H",
    "ett1/D",
    "ett1/W",
    "ett2/15T",
    "ett2/H",
    "ett2/D",
    "ett2/W",
    "jena_weather/10T",
    "jena_weather/H",
    "jena_weather/D",
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]

MEDIUM_LONG_GIFTEVAL_DATASETS = [
    "electricity/15T",
    "electricity/H",
    "solar/10T",
    "solar/H",
    "kdd_cup_2018_with_missing/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "SZ_TAXI/15T",
    "M_DENSE/H",
    "ett1/15T",
    "ett1/H",
    "ett2/15T",
    "ett2/H",
    "jena_weather/10T",
    "jena_weather/H",
    "bitbrains_fast_storage/5T",
    "bitbrains_rnd/5T",
    "bizitobs_application",
    "bizitobs_service",
    "bizitobs_l2c/5T",
    "bizitobs_l2c/H",
]

GIFTEVAL_DATASETS = list(set(SHORT_GIFTEVAL_DATASETS + MEDIUM_LONG_GIFTEVAL_DATASETS))



# fev-bench
FEV_DIR = DATADIR / "fev_bench"
FEV_EVAL_DATA_DIR = Path(
    os.getenv("FEV_EVAL_DATA_DIR", str(FEV_DIR / "eval_data"))
)

# synthetic corpora
SYNTHETIC_CORPORA_DIR = Path(
    os.getenv("SYNTHETIC_CORPORA_DIR", str(DATADIR / "synthetic_corpora"))
)

# real reference
REAL_REFERENCE_DIR = Path(
    os.getenv("REAL_REFERENCE_DIR", str(DATADIR / "real_reference"))
)

# metadata files (always in-repo, never overridden)

GIFTEVAL_PRETRAIN_METADATA = GIFTEVAL_DIR / "pretrain_metadata.csv"
GIFTEVAL_EVAL_METADATA = GIFTEVAL_DIR / "eval_metadata.csv"
GIFTEVAL_DATASET_MAPPING = GIFTEVAL_DIR / "dataset_mapping_v1.csv"
GIFTEVAL_EVAL_PROPERTIES = GIFTEVAL_DIR / "eval_dataset_properties.json"
GIFTEVAL_PRETRAIN_HF_INDEX = GIFTEVAL_DIR / "pretrain_data_info.csv"
GIFTEVAL_PRETRAIN_INDEX = Path("data/gift_eval/pretrain_index.csv")

FEV_TASKS_YAML = FEV_DIR / "tasks.yaml"
FEV_BENCHMARK_RESULTS_DIR = FEV_DIR / "benchmark_results"
FEV_LEAKAGE_CSV = FEV_DIR / "fev_bench_leakage.csv"


# results directories (outputs from this study)

RESULTS_DIR = Path("results")
FEATURES_DIR = RESULTS_DIR / "features"
COVERAGE_DIR = RESULTS_DIR / "coverage"
TSTR_DIR = RESULTS_DIR / "tstr"
TRAINED_MODELS_DIR = Path(
    os.getenv("TRAINED_MODELS_DIR", str(RESULTS_DIR / "trained_models"))
)