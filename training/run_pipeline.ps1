# =============================================================================
# Full Training Pipeline for Mapperatorinator 2024-2025 Year Support (Windows)
# =============================================================================
#
# Prerequisites:
#   - osu! OAuth credentials (https://osu.ppy.sh/home/account/edit -> OAuth)
#   - NVIDIA GPU with CUDA
#   - Python 3.10/3.11 with venv activated
#
# Usage:
#   # Set credentials
#   $env:OSU_CLIENT_ID = "your_id"
#   $env:OSU_CLIENT_SECRET = "your_secret"
#
#   # Run all steps
#   .\training\run_pipeline.ps1 -Step all
#
#   # Or individual steps
#   .\training\run_pipeline.ps1 -Step download
#   .\training\run_pipeline.ps1 -Step dataset
#   .\training\run_pipeline.ps1 -Step train
#
# =============================================================================

param(
    [ValidateSet("download", "dataset", "train", "all")]
    [string]$Step = "all"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectDir

$OszDir = ".\training\osz_2024_2025"
$DatasetDir = ".\training\dataset_2024_2025"
$YearStart = 2024
$YearEnd = 2025

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Mapperatorinator 2024-2025 Training Pipeline" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host " Project:  $ProjectDir"
Write-Host " OSZ Dir:  $OszDir"
Write-Host " Dataset:  $DatasetDir"
Write-Host ""

function Step-Download {
    Write-Host "=== Step 1: Download Beatmaps ===" -ForegroundColor Green
    Write-Host ""

    if (-not $env:OSU_CLIENT_ID -or -not $env:OSU_CLIENT_SECRET) {
        Write-Host "ERROR: osu! OAuth credentials not set." -ForegroundColor Red
        Write-Host '  $env:OSU_CLIENT_ID = "your_id"'
        Write-Host '  $env:OSU_CLIENT_SECRET = "your_secret"'
        Write-Host "  Get them at: https://osu.ppy.sh/home/account/edit"
        exit 1
    }

    pip install requests tqdm 2>$null | Out-Null

    python training/download_beatmaps.py `
        --client-id $env:OSU_CLIENT_ID `
        --client-secret $env:OSU_CLIENT_SECRET `
        --output $OszDir `
        --year-start $YearStart `
        --year-end $YearEnd `
        --status ranked loved

    Write-Host ""
    Write-Host "Download complete. .osz files in: $OszDir" -ForegroundColor Green
}

function Step-Dataset {
    Write-Host "=== Step 2: Build Dataset ===" -ForegroundColor Green
    Write-Host ""

    pip install pandas pyarrow tqdm 2>$null | Out-Null

    python training/build_dataset.py `
        --input $OszDir `
        --output $DatasetDir `
        --delete-osz

    # Get split values and update config
    $splitInfo = python -c @"
import pandas as pd
from pathlib import Path
df = pd.read_parquet(Path(r'$DatasetDir') / 'metadata.parquet')
n = df.index.get_level_values(0).nunique()
train_end = int(n * 0.9)
print(f'{n},{train_end}')
"@

    $parts = $splitInfo.Split(',')
    $total = $parts[0]
    $trainEnd = $parts[1]

    Write-Host ""
    Write-Host "Dataset has $total beatmap sets." -ForegroundColor Yellow
    Write-Host "  Train: 0 - $trainEnd"
    Write-Host "  Test:  $trainEnd - $total"

    # Update config
    $configPath = "configs\train\lora_2024_2025.yaml"
    $content = Get-Content $configPath -Raw
    $content = $content -replace 'train_dataset_end: \d+', "train_dataset_end: $trainEnd"
    $content = $content -replace 'test_dataset_start: \d+', "test_dataset_start: $trainEnd"
    $content = $content -replace 'test_dataset_end: \d+', "test_dataset_end: $total"
    Set-Content $configPath $content

    Write-Host "Config updated: $configPath" -ForegroundColor Green
}

function Step-Train {
    Write-Host "=== Step 3: LoRA Fine-Tune ===" -ForegroundColor Green
    Write-Host ""

    $datasetPath = (Resolve-Path $DatasetDir).Path

    Write-Host "Training with dataset at: $datasetPath"
    Write-Host "This will take a while depending on your GPU..." -ForegroundColor Yellow
    Write-Host ""

    python osuT5/train.py -cn lora_2024_2025 `
        "data.train_dataset_path=`"$datasetPath`"" `
        "data.test_dataset_path=`"$datasetPath`""

    Write-Host ""
    Write-Host "=== Training Complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "LoRA weights saved in checkpoint directory."
    Write-Host "To use: set 'LoRA Path or ID' in the web UI to the checkpoint path."
    Write-Host "CLI:    python inference.py lora_path=`"path/to/checkpoint`" year=2025"
}

switch ($Step) {
    "download" { Step-Download }
    "dataset"  { Step-Dataset }
    "train"    { Step-Train }
    "all" {
        Step-Download
        Step-Dataset
        Step-Train
    }
}
