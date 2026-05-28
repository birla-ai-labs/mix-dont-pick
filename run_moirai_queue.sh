#!/bin/bash
# run_moirai_queue.sh
# Invades any GPU with free space. No preferred GPU.

export WANDB_PROJECT="geometry-of-pretraining"
cd ${STORAGE_ROOT}/uni2ts

CONFIG_PATH="${STORAGE_ROOT}/uni2ts/cli/conf/pretrain"
LOG_DIR="${STORAGE_ROOT}/logs/moirai"
mkdir -p "$LOG_DIR"

MEMORY_THRESHOLD=9400
CHECK_INTERVAL=1200

declare -a QUEUE=(
    "waveform:42:moirai-center-waveform-s42"
    "ets:42:moirai-fringe-ets-s42"
    "fbm:42:moirai-baseline-fbm-s42"
    "real_reference:42:moirai-real-reference-s42"
    "mixed_all11:42:moirai-mixed-all11-s42"

    "waveform:43:moirai-center-waveform-s43"
    "ets:43:moirai-fringe-ets-s43"
    "fbm:43:moirai-baseline-fbm-s43"
    "real_reference:43:moirai-real-reference-s43"
    "mixed_all11:43:moirai-mixed-all11-s43"

    "waveform:44:moirai-center-waveform-s44"
    "ets:44:moirai-fringe-ets-s44"
    "fbm:44:moirai-baseline-fbm-s44"
    "real_reference:44:moirai-real-reference-s44"
    "mixed_all11:44:moirai-mixed-all11-s44"

    "mixed_real_7525:42:moirai-mixed-real-7525-s42"
    "mixed_real_5050:42:moirai-mixed-real-5050-s42"
    "mixed_real_2575:42:moirai-mixed-real-2575-s42"

    "mixed_real_7525:43:moirai-mixed-real-7525-s43"
    "mixed_real_5050:43:moirai-mixed-real-5050-s43"
    "mixed_real_2575:43:moirai-mixed-real-2575-s43"

    "mixed_real_7525:44:moirai-mixed-real-7525-s44"
    "mixed_real_5050:44:moirai-mixed-real-5050-s44"
    "mixed_real_2575:44:moirai-mixed-real-2575-s44"

    "arima:42:moirai-arima-s42"
    "chaotic:42:moirai-chaotic-s42"
    "garch:42:moirai-garch-s42"
    "kernelsynth:42:moirai-kernelsynth-s42"
    "sde:42:moirai-sde-s42"
    "stepfunction:42:moirai-stepfunction-s42"
    "timesynth:42:moirai-timesynth-s42"
    "tsi:42:moirai-tsi-s42"
)

get_free_memory() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1
}

find_free_gpu() {
    for gpu in 0 1 2 3; do
        free_mem=$(get_free_memory $gpu)
        if [ "$free_mem" -ge "$MEMORY_THRESHOLD" ]; then
            echo $gpu
            return 0
        fi
    done
    echo -1
    return 1
}

launch_job() {
    local gpu=$1
    local data=$2
    local seed=$3
    local run_name=$4
    local log_file="$LOG_DIR/${run_name}.log"

    echo "[LAUNCH] GPU $gpu | $run_name | seed $seed | $(date)" | tee -a "$log_file"

    CUDA_VISIBLE_DEVICES=$gpu python -m cli.train \
        --config-path "$CONFIG_PATH" \
        --config-name default \
        data="$data" \
        seed=$seed \
        run_name="$run_name" \
        >> "$log_file" 2>&1 &
}

queue_idx=0
total=${#QUEUE[@]}

echo "Starting Moirai queue: $total runs"
echo "Memory threshold: ${MEMORY_THRESHOLD}MiB free"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "$(date)"

declare -a ALL_PIDS
declare -a ALL_NAMES

while [ $queue_idx -lt $total ]; do
    item="${QUEUE[$queue_idx]}"
    IFS=':' read -r data seed run_name <<< "$item"

    gpu=$(find_free_gpu)
    if [ "$gpu" -ge 0 ]; then
        launch_job $gpu $data $seed $run_name
        ALL_PIDS+=("$!")
        ALL_NAMES+=("$run_name")
        queue_idx=$((queue_idx + 1))
        echo "[$queue_idx/$total] Launched PID $! on GPU $gpu. Waiting 120s..."
        sleep 120
    else
        echo "[WAIT] No GPU has ${MEMORY_THRESHOLD}MiB free. Checking in ${CHECK_INTERVAL}s... ($(date))"
        sleep $CHECK_INTERVAL
    fi
done

echo "All $total jobs launched. Waiting for completion..."

failed=0
for i in "${!ALL_PIDS[@]}"; do
    pid="${ALL_PIDS[$i]}"
    name="${ALL_NAMES[$i]}"
    if wait "$pid"; then
        echo "[DONE] $name (PID $pid)"
    else
        echo "[FAIL] $name (PID $pid) — exit code $?"
        ((failed++))
    fi
done

if [ "$failed" -gt 0 ]; then
    echo "[ALL DONE] $failed/$total runs FAILED. $(date)"
    exit 1
else
    echo "[ALL DONE] All $total runs succeeded. $(date)"
fi