#!/bin/bash
# run_chronos_queue.sh
# Checks every 10 minutes for a free GPU and launches next job on it.
# Skips completed runs.
# Does not relaunch runs already started in this script.
# Prevents duplicate configs from launching twice.

TRAIN_SCRIPT="${STORAGE_ROOT}/chronos-forecasting/scripts/training/train.py"
CONFIG_DIR="${REPO_DIR}/configs"
LOG_DIR="${REPO_DIR}/logs"
LOCK_DIR="$LOG_DIR/locks"

mkdir -p "$LOG_DIR"
mkdir -p "$LOCK_DIR"

export WANDB_PROJECT="geometry-of-pretraining"

USED_THRESHOLD=3000
CHECK_INTERVAL=600

CONFIGS=(
    "fbm_s42.yaml"
    "waveform_s42.yaml"
    "mixed_all11_s42.yaml"
    "real_reference_s42.yaml"
    "ets_s42.yaml"

    "fbm_s43.yaml"
    "waveform_s43.yaml"
    "mixed_all11_s43.yaml"
    "real_reference_s43.yaml"
    "ets_s43.yaml"

    "fbm_s44.yaml"
    "waveform_s44.yaml"
    "mixed_all11_s44.yaml"
    "real_reference_s44.yaml"
    "ets_s44.yaml"

    "real_mix_2575_s42.yaml"
    "real_mix_5050_s42.yaml"
    "real_mix_7525_s42.yaml"

    "real_mix_2575_s43.yaml"
    "real_mix_5050_s43.yaml"
    "real_mix_7525_s43.yaml"

    "real_mix_2575_s44.yaml"
    "real_mix_5050_s44.yaml"
    "real_mix_7525_s44.yaml"
)

declare -A LAUNCHED

get_used_memory() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1"
}

find_free_gpu() {
    for gpu in 0 1 2 3; do
        used_mem=$(get_used_memory "$gpu")
        if [ "$used_mem" -lt "$USED_THRESHOLD" ]; then
            echo "$gpu"
            return 0
        fi
    done
    echo -1
    return 1
}

get_output_dir() {
    local config="$1"
    grep '^output_dir:' "$CONFIG_DIR/$config" | awk '{print $2}' | tr -d '"'
}

get_max_steps() {
    local config="$1"
    grep '^max_steps:' "$CONFIG_DIR/$config" | awk '{print $2}'
}

is_complete() {
    local config="$1"
    local output_dir
    output_dir=$(get_output_dir "$config")

    if [ -f "$output_dir/trainer_state.json" ]; then
        local max_steps
        local last_step

        max_steps=$(get_max_steps "$config")

        last_step=$(python3 -c "
import json
try:
    with open('$output_dir/trainer_state.json') as f:
        state = json.load(f)
    print(state.get('global_step', 0))
except Exception:
    print(0)
")

        if [ "$last_step" -ge "$max_steps" ]; then
            return 0
        fi
    fi

    return 1
}

is_running() {
    local run_name="$1"
    local pid_file="$LOCK_DIR/${run_name}.pid"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")

        if kill -0 "$pid" 2>/dev/null; then
            return 0
        else
            echo "[STALE LOCK] Removing stale lock for $run_name"
            rm -f "$pid_file"
        fi
    fi

    return 1
}

launch_job() {
    local config="$1"
    local gpu="$2"
    local run_name="${config%.yaml}"
    local log_file="$LOG_DIR/${run_name}.log"
    local pid_file="$LOCK_DIR/${run_name}.pid"

    echo "[LAUNCH] GPU $gpu | $run_name | $(date)" | tee -a "$log_file"

    export WANDB_RUN_NAME="chronos-${run_name}"

    CUDA_VISIBLE_DEVICES="$gpu" python "$TRAIN_SCRIPT" \
        --config "$CONFIG_DIR/$config" \
        >> "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    LAUNCHED["$run_name"]=1

    echo "[PID] $pid | $run_name" | tee -a "$log_file"
}

all_done_or_launched() {
    for config in "${CONFIGS[@]}"; do
        local run_name="${config%.yaml}"

        if is_complete "$config"; then
            continue
        fi

        if is_running "$run_name"; then
            continue
        fi

        if [ "${LAUNCHED[$run_name]}" = "1" ]; then
            continue
        fi

        return 1
    done

    return 0
}

total=${#CONFIGS[@]}

echo "Starting Chronos queue: $total runs"
echo "Used memory threshold: ${USED_THRESHOLD}MiB"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "$(date)"

while true; do
    launched_this_round=0

    for CONFIG in "${CONFIGS[@]}"; do
        RUN_NAME="${CONFIG%.yaml}"

        if is_complete "$CONFIG"; then
            echo "[SKIP COMPLETE] $RUN_NAME"
            continue
        fi

        if is_running "$RUN_NAME"; then
            echo "[SKIP RUNNING] $RUN_NAME"
            continue
        fi

        if [ "${LAUNCHED[$RUN_NAME]}" = "1" ]; then
            echo "[SKIP LAUNCHED] $RUN_NAME already launched in this script."
            continue
        fi

        gpu=$(find_free_gpu)

        if [ "$gpu" -ge 0 ]; then
            launch_job "$CONFIG" "$gpu"
            launched_this_round=1
            echo "Waiting 120s for memory to allocate..."
            sleep 120
        else
            echo "[WAIT] No GPU free. Checking again in ${CHECK_INTERVAL}s... ($(date))"
            sleep "$CHECK_INTERVAL"
            break
        fi
    done

    if all_done_or_launched; then
        echo "All configs are either complete, running, or already launched once."
        break
    fi

    if [ "$launched_this_round" -eq 0 ]; then
        echo "[WAIT] No launch this round. Checking again in ${CHECK_INTERVAL}s... ($(date))"
        sleep "$CHECK_INTERVAL"
    fi
done

echo "Waiting for jobs launched by this script..."
wait

echo "[QUEUE DONE] ($(date))"
