#!/bin/bash

# Usage: ./summarize_rl_results_av_seed_const.sh /path/to/main_directory
trap "echo 'Stopping all child processes...'; kill 0; exit" SIGINT

MAIN_DIR="$1"
CHECKPOINT_NAME="checkpoint_best"

EVALUATE_SCRIPT="evaluate_control_rl.py"
NUM_TESTS=30
NUM_WORKERS=0
RANDOM_SEED=0
NUM_PROCESSES=10

declare -a SINGLE_AV_PERCENTS=()
declare -a MULTI_AV_PERCENTS=()

if [ -z "$MAIN_DIR" ]; then
    echo "Usage: $0 /path/to/main_directory"
    exit 1
fi

# Loop over all matching checkpoint directories
while read -r CHECKPOINT_DIR; do

    PARENT_DIR=$(dirname "$CHECKPOINT_DIR")
    PARENT_NAME=$(basename "$PARENT_DIR")

    # Extract av_<number>
    if [[ "$PARENT_NAME" =~ av_([0-9]+) ]]; then
        AV_PERCENT="${BASH_REMATCH[1]}"
    else
        echo "Warning: Could not find 'av_<percent>' in path: $PARENT_NAME"
        continue
    fi

    # --------------------------------------------
    # Loop over CONST_VAL in 0.0,0.1,...,1.0
    for CONST_VAL in $(seq 0 0.1 1.0); do

        CMD=(python "$EVALUATE_SCRIPT" "$CHECKPOINT_DIR" \
             --num_workers "$NUM_WORKERS" \
             --num_tests "$NUM_TESTS" \
             --random_seed "$RANDOM_SEED" \
             --auto_results_dir \
             --const_control \
             --const_control_val_norm "$CONST_VAL")

        if [[ "$PARENT_NAME" == *single_lane* ]]; then
            SINGLE_AV_PERCENTS+=("$AV_PERCENT")
        else
            MULTI_AV_PERCENTS+=("$AV_PERCENT")
        fi

        echo "Running (CONST_VAL=${CONST_VAL}): ${CMD[*]}"
        "${CMD[@]}" &  # Run in background

        # Throttle parallel processes
        background=( $(jobs -p) )
        if (( ${#background[@]} == NUM_PROCESSES )); then
            wait -n
        fi
    done
    # --------------------------------------------

done < <(find "$MAIN_DIR" -type d -name "$CHECKPOINT_NAME")

# Wait for all background jobs to finish
wait
