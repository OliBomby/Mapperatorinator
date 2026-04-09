#!/bin/bash

# Multi-Train Script
# Trains a base model on all gamemodes, then fine-tunes on each gamemode separately.
#
# Usage: ./osuT5/multi_train.sh [--skip-base] <config_name> <finetune_config_name> <run_name> [extra_overrides...]
#
# Examples:
#   ./osuT5/multi_train.sh tiny64 tiny64_ft "tiny64 train"
#   ./osuT5/multi_train.sh tiny64 tiny64_ft "tiny64 train" optim.total_steps=50000
#   ./osuT5/multi_train.sh --skip-base tiny64 tiny64_ft "tiny64 train"

set -e  # Exit on error

# cd to the repository root (parent of the directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}======================================${NC}"
    echo
}

print_usage() {
    echo -e "${RED}Usage: $0 [--skip-base] <config_name> <finetune_config_name> <run_name> [extra_overrides...]${NC}"
    echo "  --skip-base:          Skip base training and use the latest checkpoint in ./logs/<run_name>/base"
    echo "  config_name:          Hydra config name for base training (e.g. tiny64)"
    echo "  finetune_config_name: Hydra config name for fine-tune runs (e.g. tiny64_ft)"
    echo "  run_name:             Base run name for wandb (e.g. 'tiny64 train')"
    echo "  extra_overrides:      Optional additional Hydra overrides"
}

# --- Argument parsing ---
SKIP_BASE=false

if [ "$1" = "--skip-base" ]; then
    SKIP_BASE=true
    shift
fi

if [ $# -lt 3 ]; then
    print_usage
    exit 1
fi

CONFIG_NAME="$1"
FINETUNE_CONFIG_NAME="$2"
RUN_NAME="$3"
shift 3
EXTRA_OVERRIDES=("$@")

# Sanitize run name for use as directory name (replace spaces with underscores)
RUN_DIR_NAME=$(echo "$RUN_NAME" | tr ' ' '_')

# Gamemode names for wandb run name suffixes
declare -A GAMEMODE_NAMES
GAMEMODE_NAMES[0]="standard"
GAMEMODE_NAMES[1]="taiko"
GAMEMODE_NAMES[2]="ctb"
GAMEMODE_NAMES[3]="mania"

GAMEMODES=(0 1 2 3)

# --- Helper: find the latest checkpoint directory ---
find_latest_checkpoint() {
    local run_dir="$1"
    local latest

    latest=$(find "$run_dir/checkpoints" -maxdepth 1 -type d -name 'checkpoint_*' 2>/dev/null \
        | sed 's/.*checkpoint_//' \
        | sort -n \
        | tail -1)
    if [ -n "$latest" ]; then
        echo "$run_dir/checkpoints/checkpoint_$latest"
        return
    fi

    latest=$(find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null \
        | sed 's/.*checkpoint-//' \
        | sort -n \
        | tail -1)
    if [ -n "$latest" ]; then
        echo "$run_dir/checkpoint-$latest"
        return
    fi

    echo ""
}

# ==============================
# Phase 1: Base model training
# ==============================
BASE_LOG_DIR="./logs/${RUN_DIR_NAME}/base"

if [ "$SKIP_BASE" = false ]; then
    print_header "Phase 1: Base Model Training"
    echo -e "${GREEN}Config:${NC}   $CONFIG_NAME"
    echo -e "${GREEN}Run Name:${NC} $RUN_NAME"
    echo -e "${GREEN}Log Dir:${NC}  $BASE_LOG_DIR"
    echo

    python osuT5/train.py \
        -cn "$CONFIG_NAME" \
        "logging.run_name=$RUN_NAME" \
        "data.gamemodes=[0,1,2,3]" \
        "hydra.run.dir=$BASE_LOG_DIR" \
        "${EXTRA_OVERRIDES[@]}"
else
    print_header "Phase 1: Base Model Training Skipped"
    echo -e "${GREEN}Using existing base log dir:${NC} $BASE_LOG_DIR"
    echo
fi

# Find the latest checkpoint from base training
LATEST_CHECKPOINT=$(find_latest_checkpoint "$BASE_LOG_DIR")

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo -e "${RED}Error: No checkpoint found in $BASE_LOG_DIR${NC}"
    exit 1
fi

# Convert to absolute path
LATEST_CHECKPOINT=$(realpath "$LATEST_CHECKPOINT")
echo -e "${GREEN}Latest base checkpoint:${NC} $LATEST_CHECKPOINT"

# ==============================
# Phase 2: Fine-tune per gamemode
# ==============================
for GM in "${GAMEMODES[@]}"; do
    GM_NAME="${GAMEMODE_NAMES[$GM]}"
    FT_RUN_NAME="$RUN_NAME $GM_NAME"
    FT_LOG_DIR="./logs/${RUN_DIR_NAME}/${GM_NAME}"

    print_header "Phase 2: Fine-Tune - $GM_NAME (gamemode $GM)"
    echo -e "${GREEN}Run Name:${NC}        $FT_RUN_NAME"
    echo -e "${GREEN}Pretrained Path:${NC} $LATEST_CHECKPOINT"
    echo -e "${GREEN}Log Dir:${NC}         $FT_LOG_DIR"
    echo

    python osuT5/train.py \
        -cn "$FINETUNE_CONFIG_NAME" \
        "logging.run_name=$FT_RUN_NAME" \
        "data.gamemodes=[$GM]" \
        "pretrained_path=$LATEST_CHECKPOINT" \
        "hydra.run.dir=$FT_LOG_DIR" \
        "${EXTRA_OVERRIDES[@]}"

    echo -e "${GREEN}Finished fine-tuning for $GM_NAME (gamemode $GM)${NC}"
done

print_header "All training runs complete!"
echo -e "${GREEN}Base model:${NC}     $BASE_LOG_DIR"
for GM in "${GAMEMODES[@]}"; do
    GM_NAME="${GAMEMODE_NAMES[$GM]}"
    echo -e "${GREEN}  $GM_NAME:${NC}  ./logs/${RUN_DIR_NAME}/${GM_NAME}"
done

