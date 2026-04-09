# Training Pipeline: 2024-2025 Year Support

Fine-tune Mapperatorinator on recent (2024-2025) beatmap data using LoRA.

## Prerequisites

- **osu! OAuth credentials** — get them at https://osu.ppy.sh/home/account/edit → OAuth → New Application
- **NVIDIA GPU with CUDA** (8GB+ VRAM recommended; 24GB for full batch sizes)
- **Python 3.10/3.11** with the project's venv activated
- **~50-100GB disk space** for .osz files + dataset

## Quick Start (Windows)

```powershell
# Set osu! API credentials
$env:OSU_CLIENT_ID = "your_client_id"
$env:OSU_CLIENT_SECRET = "your_client_secret"

# Run the full pipeline (download → build dataset → train)
.\training\run_pipeline.ps1 -Step all
```

## Quick Start (Linux/macOS)

```bash
export OSU_CLIENT_ID="your_client_id"
export OSU_CLIENT_SECRET="your_client_secret"

bash training/run_pipeline.sh all
```

## Individual Steps

### 1. Download Beatmaps

Downloads ranked and loved beatmaps from 2024-2025 as `.osz` files using the osu! API v2.

```powershell
.\training\run_pipeline.ps1 -Step download
# or manually:
python training/download_beatmaps.py --client-id YOUR_ID --client-secret YOUR_SECRET --output training/osz_2024_2025
```

### 2. Build Dataset

Converts `.osz` files into MMRS format (the training format Mapperatorinator expects).

```powershell
.\training\run_pipeline.ps1 -Step dataset
# or manually:
python training/build_dataset.py --input training/osz_2024_2025 --output training/dataset_2024_2025
```

### 3. Train (LoRA Fine-Tune)

Fine-tunes the V31 model using LoRA on the new dataset.

```powershell
.\training\run_pipeline.ps1 -Step train
# or manually:
python osuT5/train.py -cn lora_2024_2025 data.train_dataset_path="path/to/dataset" data.test_dataset_path="path/to/dataset"
```

## Using the Trained Model

After training completes, the LoRA checkpoint will be saved in the `logs/` directory.

**Web UI:** Set the "LoRA Path or ID" field to the checkpoint directory path.

**CLI:**
```bash
python inference.py lora_path="logs/<run>/checkpoints/<step>" year=2025
```

## Config

The training config is at `configs/train/lora_2024_2025.yaml`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `total_steps` | 2000 | Training iterations |
| `lora_r` | 64 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 128 | LoRA scaling factor |
| `learning_rate` | 0.0004 | Muon optimizer LR |
| `min_year` / `max_year` | 2024 / 2025 | Year range filter |

## Notes

- LoRA fine-tuning is much lighter than full retraining (~2000 steps vs 65000+)
- The base model (V31) already knows mapping patterns; LoRA adapts it to 2024-2025 data
- Training takes roughly 1-4 hours depending on GPU and dataset size
- Downloads use the catboy.best mirror for .osz files (faster than osu! direct)
