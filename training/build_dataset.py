import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_osu_metadata(osu_path: Path) -> dict:
    """Parse metadata from a .osu file header."""
    metadata = {}
    section = None

    try:
        with open(osu_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()

                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1]
                    continue

                if section in ("General", "Metadata", "Difficulty") and ":" in line:
                    key, _, value = line.partition(":")
                    metadata[key.strip()] = value.strip()

                # Stop after we have what we need
                if section == "HitObjects":
                    break
    except (UnicodeDecodeError, OSError):
        pass

    return metadata


def extract_osz(osz_path: Path, output_dir: Path) -> list[dict]:
    """Extract an .osz file and return metadata for each difficulty."""
    beatmapset_id = osz_path.stem
    set_dir = output_dir / "data" / str(beatmapset_id)

    if set_dir.exists():
        # Already extracted
        return []

    set_dir.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        with zipfile.ZipFile(osz_path, "r") as z:
            # Only extract .osu files and audio — skip images, videos, storyboards
            osu_files = [n for n in z.namelist() if n.endswith(".osu")]
            audio_files = [n for n in z.namelist()
                           if n.lower().endswith((".mp3", ".ogg", ".wav"))]

            # Extract audio (just the first one found)
            if audio_files:
                z.extract(audio_files[0], set_dir)

            # Extract .osu files
            for name in osu_files:
                z.extract(name, set_dir)

            # Parse each .osu file
            for name in osu_files:
                    osu_path = set_dir / name
                    meta = parse_osu_metadata(osu_path)

                    beatmap_id = meta.get("BeatmapID", "0")
                    mode = int(meta.get("Mode", "0"))
                    difficulty_name = meta.get("Version", "Unknown")

                    audio_filename = meta.get("AudioFilename", "")

                    results.append({
                        "BeatmapSetId": int(beatmapset_id),
                        "Id": int(beatmap_id),
                        "BeatmapSetFolder": str(beatmapset_id),
                        "BeatmapFile": name,
                        "AudioFile": audio_filename,
                        "ModeInt": mode,
                        "DifficultyName": difficulty_name,
                    })

    except (zipfile.BadZipFile, OSError) as e:
        print(f"  Failed to extract {osz_path}: {e}")
        if set_dir.exists():
            shutil.rmtree(set_dir, ignore_errors=True)
        return []

    return results


def enrich_with_api_metadata(records: list[dict], api_metadata_path: Path) -> list[dict]:
    """Merge extracted .osu metadata with API metadata (ranked date, star rating)."""
    if not api_metadata_path.exists():
        print("Warning: No API metadata file found. Star ratings and ranked dates will be estimated.")
        return records

    with open(api_metadata_path, "r", encoding="utf-8") as f:
        api_data = json.load(f)

    # Build lookup: beatmapset_id -> {beatmap_id -> beatmap_data}
    api_lookup = {}
    for bset in api_data:
        bset_id = bset["id"]
        api_lookup[bset_id] = {
            "ranked_date": bset.get("ranked_date") or bset.get("submitted_date"),
            "submitted_date": bset.get("submitted_date"),
            "user_id": bset.get("user_id", 0),
            "beatmaps": {}
        }
        for bm in bset.get("beatmaps", []):
            api_lookup[bset_id]["beatmaps"][bm["id"]] = {
                "difficulty_rating": bm.get("difficulty_rating", 0),
                "mode_int": bm.get("mode_int", 0),
            }

    enriched = []
    for rec in records:
        bset_id = rec["BeatmapSetId"]
        bm_id = rec["Id"]

        api_bset = api_lookup.get(bset_id, {})
        api_bm = api_bset.get("beatmaps", {}).get(bm_id, {})

        # Parse ranked date (as naive datetime to match MMRS format)
        ranked_str = api_bset.get("ranked_date")
        if ranked_str:
            try:
                ranked_date = datetime.fromisoformat(ranked_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, AttributeError):
                ranked_date = datetime(2024, 1, 1)
        else:
            ranked_date = datetime(2024, 1, 1)

        # Parse submitted date (as naive datetime to match MMRS format)
        submitted_str = api_bset.get("submitted_date")
        if submitted_str:
            try:
                submitted_date = datetime.fromisoformat(submitted_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, AttributeError):
                submitted_date = ranked_date
        else:
            submitted_date = ranked_date

        # StarRating must be a 7-element array for speeds [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        # We only have the 1.0x speed rating; estimate others with rough scaling
        base_sr = api_bm.get("difficulty_rating", 0.0)
        rec["StarRating"] = [
            base_sr * 0.6,   # 0.5x
            base_sr * 0.8,   # 0.75x
            base_sr,         # 1.0x (actual value)
            base_sr * 1.15,  # 1.25x
            base_sr * 1.3,   # 1.5x
            base_sr * 1.45,  # 1.75x
            base_sr * 1.6,   # 2.0x
        ]
        rec["DifficultyRating"] = base_sr
        rec["RankedDate"] = ranked_date
        rec["SubmittedDate"] = submitted_date
        rec["UserId"] = api_bset.get("user_id", 0)
        rec["ModeInt"] = api_bm.get("mode_int", rec.get("ModeInt", 0))
        enriched.append(rec)

    return enriched


def build_metadata_parquet(records: list[dict], output_dir: Path):
    """Build the metadata.parquet file in MMRS format."""
    if not records:
        print("No records to write!")
        return

    df = pd.DataFrame(records)

    # Add BeatmapIdx (sequential index)
    df["BeatmapIdx"] = range(len(df))

    # Ensure correct types
    df["BeatmapSetId"] = df["BeatmapSetId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["ModeInt"] = df["ModeInt"].astype(int)
    df["DifficultyRating"] = df["DifficultyRating"].astype(float)
    df["UserId"] = df["UserId"].astype(int)

    # Sort by BeatmapSetId
    df = df.sort_values("BeatmapSetId")

    # Do NOT set_index here — data_utils.load_mmrs_metadata() does that itself
    parquet_path = output_dir / "metadata.parquet"
    df.to_parquet(parquet_path, index=False)

    # Print stats
    n_sets = df["BeatmapSetId"].nunique()
    n_maps = len(df)
    print(f"\nDataset built:")
    print(f"  Beatmap sets: {n_sets}")
    print(f"  Total difficulties: {n_maps}")
    print(f"  Output: {parquet_path}")

    # Year distribution
    if "RankedDate" in df.columns:
        year_counts = df["RankedDate"].dt.year.value_counts().sort_index()
        print(f"\n  Year distribution:")
        for year, count in year_counts.items():
            print(f"    {year}: {count} beatmaps")

    # Gamemode distribution
    mode_names = {0: "Standard", 1: "Taiko", 2: "Catch", 3: "Mania"}
    mode_counts = df["ModeInt"].value_counts().sort_index()
    print(f"\n  Gamemode distribution:")
    for mode, count in mode_counts.items():
        print(f"    {mode_names.get(mode, f'Mode {mode}')}: {count}")

    return n_sets


def main():
    parser = argparse.ArgumentParser(description="Build MMRS dataset from .osz files")
    parser.add_argument("--input", required=True,
                        help="Directory containing .osz files (from download_beatmaps.py)")
    parser.add_argument("--output", required=True,
                        help="Output directory for MMRS dataset")
    parser.add_argument("--delete-osz", action="store_true",
                        help="Delete each .osz file after successful extraction to save disk space")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    osz_files = sorted(input_dir.glob("*.osz"))
    data_dir = output_dir / "data"
    if not osz_files and not data_dir.exists():
        print(f"No .osz files found in {input_dir}")
        sys.exit(1)

    if osz_files:
        print(f"Found {len(osz_files)} .osz files")
    else:
        print(f"No .osz files found, scanning already-extracted data in {data_dir}")

    # Extract all .osz files
    all_records = []
    for osz_path in tqdm(osz_files, desc="Extracting .osz files"):
        records = extract_osz(osz_path, output_dir)
        all_records.extend(records)
        # Delete .osz after successful extraction to free disk space
        if args.delete_osz and records:
            osz_path.unlink()

    # Also check already-extracted directories for records
    if not all_records:
        print("Checking already-extracted directories...")
        data_dir = output_dir / "data"
        if data_dir.exists():
            for set_dir in sorted(data_dir.iterdir()):
                if set_dir.is_dir():
                    for osu_file in set_dir.glob("*.osu"):
                        meta = parse_osu_metadata(osu_file)
                        all_records.append({
                            "BeatmapSetId": int(set_dir.name),
                            "Id": int(meta.get("BeatmapID", "0")),
                            "BeatmapSetFolder": set_dir.name,
                            "BeatmapFile": osu_file.name,
                            "AudioFile": meta.get("AudioFilename", ""),
                            "ModeInt": int(meta.get("Mode", "0")),
                            "DifficultyName": meta.get("Version", "Unknown"),
                        })

    print(f"\nTotal beatmap difficulties found: {len(all_records)}")

    # Enrich with API metadata
    api_metadata_path = input_dir / "beatmapset_metadata.json"
    all_records = enrich_with_api_metadata(all_records, api_metadata_path)

    # Build parquet
    n_sets = build_metadata_parquet(all_records, output_dir)

    if n_sets:
        train_end = int(n_sets * 0.9)
        print(f"\n--- Training Config Values ---")
        print(f"  train_dataset_path: \"{output_dir.resolve()}\"")
        print(f"  train_dataset_start: 0")
        print(f"  train_dataset_end: {train_end}")
        print(f"  test_dataset_start: {train_end}")
        print(f"  test_dataset_end: {n_sets}")
        print(f"\nUpdate configs/train/lora_2024_2025.yaml with these values.")


if __name__ == "__main__":
    main()
