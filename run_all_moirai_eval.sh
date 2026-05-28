#!/usr/bin/env bash
set -euo pipefail

seed=42

conditions=(arima chaotic garch kernelsynth sde stepfunction tsi timesynth)
gpus=(0 0 1 1 2 2 3 3)

for i in "${!conditions[@]}"; do
  condition="${conditions[$i]}"
  gpu="${gpus[$i]}"
  log="eval_gpu${gpu}_${condition}.log"

  echo "Launching $condition on GPU $gpu -> $log"

  nohup python -u scripts/run_moirai_eval.py \
    --seed "$seed" \
    --gpu "$gpu" \
    --conditions "$condition" \
    > "$log" 2>&1 &
done

echo "Launched all jobs:"
jobs -l
