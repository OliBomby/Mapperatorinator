#!/bin/bash
# =============================================================================
# Full Training Pipeline for Mapperatorinator 2024-2025 Year Support
# =============================================================================
#
# This script automates the entire process of adding year 2024-2025 support:
#   1. Download ranked/loved beatmaps from 2024-2025
#   2. Build a training dataset from the downloaded .osz files
#   3. Run LoRA fine-tuning on the V31 model
#
# Prerequisites:
#   - osu! OAuth credentials (get from https://osu.ppy.sh/home/account/edit)
#   - NVIDIA GPU with CUDA (for training)
#   - Docker + nvidia-container-toolkit (for containerized training)
#   - OR: Local venv with all dependencies + flash-attn
#
# Usage:
#   # Set your osu! OAuth credentials
#   export OSU_CLIENT_ID="your_client_id"
#   export OSU_CLIENT_SECRET="your_client_secret"
#
#   # Option A: Run everything
#   ./training/run_pipeline.sh all
#
#   # Option B: Run individual steps
#   ./training/run_pipeline.sh download    # Step 1: Download beatmaps
#   ./training/run_pipeline.sh dataset     # Step 2: Build dataset
#   ./training/run_pipeline.sh train       # Step 3: Run LoRA fine-tune
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Config
OSZ_DIR="./training/osz_2024_2025"
DATASET_DIR="./training/dataset_2024_2025"
YEAR_START=2024
YEAR_END=2025

echo "============================================"
echo " Mapperatorinator 2024-2025 Training Pipeline"
echo "============================================"
echo ""
echo " Project: $PROJECT_DIR"
echo " OSZ Dir: $OSZ_DIR"
echo " Dataset: $DATASET_DIR"
echo ""

step_download() {
    echo "=== Step 1: Download Beatmaps ==="
    echo ""
    
    if [ -z "$OSU_CLIENT_ID" ] || [ -z "$OSU_CLIENT_SECRET" ]; then
        echo "ERROR: osu! OAuth credentials not set."
        echo "  export OSU_CLIENT_ID='your_id'"
        echo "  export OSU_CLIENT_SECRET='your_secret'"
        echo "  Get credentials at: https://osu.ppy.sh/home/account/edit"
        exit 1
    fi

    pip install requests tqdm 2>/dev/null || true

    python training/download_beatmaps.py \
        --client-id "$OSU_CLIENT_ID" \
        --client-secret "$OSU_CLIENT_SECRET" \
        --output "$OSZ_DIR" \
        --year-start $YEAR_START \
        --year-end $YEAR_END \
        --status ranked loved

    echo ""
    echo "Download complete. .osz files in: $OSZ_DIR"
    echo ""
}

step_dataset() {
    echo "=== Step 2: Build Dataset ==="
    echo ""

    pip install pandas pyarrow tqdm 2>/dev/null || true

    python training/build_dataset.py \
        --input "$OSZ_DIR" \
        --output "$DATASET_DIR" \
        --delete-osz

    echo ""
    echo "Dataset built in: $DATASET_DIR"
    
    # Calculate split values
    N_SETS=$(python -c "
import pandas as pd
from pathlib import Path
df = pd.read_parquet(Path('$DATASET_DIR') / 'metadata.parquet')
n = df.index.get_level_values(0).nunique()
print(n)
")
    TRAIN_END=$(python -c "print(int($N_SETS * 0.9))")
    
    echo ""
    echo "Dataset has $N_SETS beatmap sets."
    echo "  Train split: 0 - $TRAIN_END"
    echo "  Test split:  $TRAIN_END - $N_SETS"
    echo ""
    echo "Updating lora_2024_2025.yaml with split values..."
    
    # Update the config file with actual dataset sizes
    python -c "
import re
from pathlib import Path

config_path = Path('configs/train/lora_2024_2025.yaml')
text = config_path.read_text()

text = re.sub(r'train_dataset_end: \d+', 'train_dataset_end: $TRAIN_END', text)
text = re.sub(r'test_dataset_start: \d+', 'test_dataset_start: $TRAIN_END', text)
text = re.sub(r'test_dataset_end: \d+', 'test_dataset_end: $N_SETS', text)

config_path.write_text(text)
print('Config updated.')
"
}

step_train() {
    echo "=== Step 3: LoRA Fine-Tune ==="
    echo ""
    echo "Training options:"
    echo "  A) Docker (recommended for Linux/WSL)"
    echo "  B) Local venv (if you have flash-attn installed)"
    echo ""
    
    # Check if running in Docker
    if [ -f /.dockerenv ]; then
        echo "Running inside Docker container."
        DATASET_PATH="/workspace/datasets/MMRS_2024_2025"
        
        # Copy dataset to expected location if needed
        if [ ! -d "$DATASET_PATH" ] && [ -d "$DATASET_DIR" ]; then
            echo "Copying dataset to $DATASET_PATH..."
            cp -r "$DATASET_DIR" "$DATASET_PATH"
        fi
        
        python osuT5/train.py -cn lora_2024_2025 \
            data.train_dataset_path="$DATASET_PATH" \
            data.test_dataset_path="$DATASET_PATH"
    else
        echo "Running locally (outside Docker)."
        echo ""
        echo "If you have flash-attn installed, training will run locally."
        echo "Otherwise, use Docker:"
        echo "  docker compose up -d --force-recreate"
        echo "  docker attach mapperatorinator_space"
        echo "  cd Mapperatorinator"
        echo "  ./training/run_pipeline.sh train"
        echo ""
        
        DATASET_PATH="$(realpath "$DATASET_DIR")"
        
        python osuT5/train.py -cn lora_2024_2025 \
            data.train_dataset_path="$DATASET_PATH" \
            data.test_dataset_path="$DATASET_PATH"
    fi
    
    echo ""
    echo "=== Training Complete ==="
    echo ""
    echo "Your LoRA weights are saved in the checkpoint directory."
    echo "To use them during inference, set lora_path to the checkpoint path."
    echo ""
    echo "In the web UI: set the 'LoRA Path or ID' field to your checkpoint path."
    echo "In CLI: python inference.py lora_path=\"path/to/checkpoint\" year=2025"
}

# Main
case "${1:-all}" in
    download)
        step_download
        ;;
    dataset)
        step_dataset
        ;;
    train)
        step_train
        ;;
    all)
        step_download
        step_dataset
        step_train
        ;;
    *)
        echo "Usage: $0 {download|dataset|train|all}"
        exit 1
        ;;
esac
