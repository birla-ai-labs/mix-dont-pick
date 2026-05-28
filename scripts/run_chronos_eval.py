import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",  type=int, required=True)
parser.add_argument("--conditions", nargs="+", default=None, help="List of conditions to run (default: all)")
parser.add_argument("--seed", nargs="+", default=None, help="List of seeds to run (default: all)")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.gift_eval import Evaluator
from utils.predictors import ChronosPredictor
from config import TRAINED_MODELS_DIR, TSTR_DIR

MAIN_CONDITIONS = ["waveform", "ets", "fbm", "mixed_all11", "real_reference", "arima", "chaotic", "garch", "kernelsynth", "sde", "stepfunction", "timesynth", "tsi"]
SPLIT_CONDITIONS = ["mixed_real_2575", "mixed_real_5050", "mixed_real_7525"]

CONDITIONS = MAIN_CONDITIONS + SPLIT_CONDITIONS

if args.conditions is not None:
    CONDITIONS = args.conditions

if args.seed is not None:
    seeds = args.seed
else:
    seeds = [42, 43, 44]

if __name__ == "__main__":
    for condition in CONDITIONS:
        for seed in seeds:
            model_path = TRAINED_MODELS_DIR / "chronos" / f"seed_{seed}" / condition
            if not model_path.exists():
                print(f"SKIP (not found): {model_path}")
                continue

            run_name = condition
            print(f"\n{'='*50}\nRunning: {run_name} seed={seed} gpu={args.gpu}\n{'='*50}")

            evaluator = Evaluator(
                model_name=run_name,
                output_dir=TSTR_DIR / "chronos" / f"seed_{seed}",
                predictor=ChronosPredictor,
                predictor_args={
                    "model_path": str(model_path),
                    "num_samples": 20,
                    "quantile_levels": [0.1, 0.5, 0.9],
                },
            )
            evaluator.iterate_datasets()
            evaluator.evaluate_datasets(save_forecasts=False)