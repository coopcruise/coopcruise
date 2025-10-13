#!/bin/bash

# Usage:
#   ./summarize_rl_results_av_seed.sh /path/to/main_directory [--exploit] [--results_dir_prefix PREFIX]
trap "echo 'Stopping all child processes...'; kill 0; exit" SIGINT

# -----------------------
# Parse required + optional args
# -----------------------
MAIN_DIR=""
EXPLOIT_FLAG=""
RESULTS_DIR_PREFIX=""

# First positional argument is MAIN_DIR
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/main_directory [--exploit] [--results_dir_prefix PREFIX]"
    exit 1
fi

MAIN_DIR="$1"
shift  # Remove MAIN_DIR from the list of arguments

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exploit)
            EXPLOIT_FLAG="--exploit"
            shift
            ;;
        --results_dir_prefix)
            if [[ -n "$2" ]]; then
                RESULTS_DIR_PREFIX="--results_dir_prefix $2"
                shift 2
            else
                echo "Error: --results_dir_prefix requires a string argument"
                exit 1
            fi
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 /path/to/main_directory [--exploit] [--results_dir_prefix PREFIX]"
            exit 1
            ;;
    esac
done

# -----------------------
# Script configuration
# -----------------------
CHECKPOINT_NAME="checkpoint_best"
EVALUATE_SCRIPT="evaluate_control_rl.py"
ANALYSIS_SCRIPT="simulation_analysis.py"
NUM_TESTS=30
NUM_WORKERS=0
RANDOM_SEED=0
NUM_PROCESSES=10

declare -a SINGLE_AV_PERCENTS=()
declare -a MULTI_AV_PERCENTS=()

# -----------------------
# Main loop
# -----------------------
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

    # Build the python command
    CMD=(python "$EVALUATE_SCRIPT" "$CHECKPOINT_DIR"
         --num_workers "$NUM_WORKERS"
         --num_tests "$NUM_TESTS"
         --random_seed "$RANDOM_SEED"
         --auto_results_dir)

    # Add optional flags if provided
    [[ -n "$EXPLOIT_FLAG" ]] && CMD+=($EXPLOIT_FLAG)
    [[ -n "$RESULTS_DIR_PREFIX" ]] && CMD+=($RESULTS_DIR_PREFIX)

    if [[ "$PARENT_NAME" == *single_lane* ]]; then
        SINGLE_AV_PERCENTS+=("$AV_PERCENT")
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
