#!/bin/bash

# Usage: ./run_for_checkpoints.sh /path/to/main_directory

MAIN_DIR="$1"
CHECKPOINT_NAME="checkpoint_best"  # <--- Set this to your desired checkpoint directory name

RESULTS_DIR=$(basename "$MAIN_DIR")
EVALUATE_SCRIPT="evaluate_control_new.py"
ANALYSIS_SCRIPT="simulation_analysis.py"
NUM_TESTS=30
NUM_WORKERS=0
RANDOM_SEED=0
NUM_PROCESSES=4

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

    # Extract ENV_CLASS using pattern matching
    IFS='_' read -r ALGO ENV_CLASS _ <<< "$PARENT_NAME"

    # Extract av_<number>
    if [[ "$PARENT_NAME" =~ av_([0-9]+) ]]; then
        AV_PERCENT="${BASH_REMATCH[1]}"
    else
        echo "Warning: Could not find 'av_<percent>' in path: $PARENT_NAME"
        continue
    fi

    # Build the python command
    CMD=(python "$EVALUATE_SCRIPT" "$CHECKPOINT_DIR" --num_workers "$NUM_WORKERS" --num_tests "$NUM_TESTS" --av_percent "$AV_PERCENT" --results_dir "$RESULTS_DIR" --random_seed "$RANDOM_SEED" --env_class "$ENV_CLASS")

    # Add optional flag if 'single_lane' is in the path
    if [[ "$PARENT_NAME" == *single_lane* ]]; then
        SINGLE_AV_PERCENTS+=("$AV_PERCENT")
        CMD+=(--single_lane)
    else
        MULTI_AV_PERCENTS+=("$AV_PERCENT")
    fi

    echo "Running: ${CMD[*]}"
    "${CMD[@]}" &  # Run in background

    # Throttle parallel processes
    background=( $(jobs -p) )
    if (( ${#background[@]} == NUM_PROCESSES )); then
        wait -n
    fi

done < <(find "$MAIN_DIR" -type d -name "$CHECKPOINT_NAME")

# Wait for all background jobs to finish
wait

# Deduplicate each AV list
UNIQUE_SINGLE_AVS=($(printf "%s\n" "${SINGLE_AV_PERCENTS[@]}" | sort -n | uniq))
UNIQUE_MULTI_AVS=($(printf "%s\n" "${MULTI_AV_PERCENTS[@]}" | sort -n | uniq))

ANALYSIS_CMD=(python "$ANALYSIS_SCRIPT" --num_tests "$NUM_TESTS" --results_dir "$RESULTS_DIR")

# Run second script for single-lane AV percentages
if [ ${#UNIQUE_SINGLE_AVS[@]} -gt 0 ]; then
    echo "Running second script for single_lane AVs: ${UNIQUE_SINGLE_AVS[*]}"
    "${ANALYSIS_CMD[@]}" --av_percent "${UNIQUE_SINGLE_AVS[@]}" --single_lane &
fi

# Run second script for multi-lane AV percentages
if [ ${#UNIQUE_MULTI_AVS[@]} -gt 0 ]; then
    echo "Running second script for multi_lane AVs: ${UNIQUE_MULTI_AVS[*]}"
    "${ANALYSIS_CMD[@]}" --av_percent "${UNIQUE_MULTI_AVS[@]}" &
fi

wait
