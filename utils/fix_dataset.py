import pandas as pd
from pathlib import Path

from tqdm import tqdm

dataset_path = Path(r"/datasets/MMRS39389")
metadata_path = dataset_path / "metadata.parquet"


# For each row, find the corresponding file in the dataset and update the value in the BeatmapFile column

# To find the file, look through all .osu files in BeatmapSetFolder, read the BeatmapID from the .osu file, and compare it to the Id column
# If the BeatmapID matches, update the BeatmapFile column with the name of the .osu file
# Print each row that is changed

# Load metadata
df = pd.read_parquet(metadata_path)


def get_beatmap_id(osu_file: Path):
    with osu_file.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("BeatmapID:"):
                return int(line.split(":")[1].strip())
    return None

changed_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Updating BeatmapFile"):
    beatmapset_folder = dataset_path / "data" / str(row["BeatmapSetFolder"])
    if (beatmapset_folder / row["BeatmapFile"]).exists():
        continue
    found = False
    for osu_file in beatmapset_folder.glob("*.osu"):
        beatmap_id = get_beatmap_id(osu_file)
        if beatmap_id == row["Id"]:
            if row["BeatmapFile"] != osu_file.name:
                df.at[idx, "BeatmapFile"] = osu_file.name
                changed_rows.append((idx, row["Id"], osu_file.name))
            found = True
            break
    if not found:
        print(f"Warning: No matching .osu file found for Id {row['Id']} in {beatmapset_folder}")

# Print changed rows
for idx, beatmap_id, osu_name in changed_rows:
    print(f"Row {idx} (Id={beatmap_id}) updated to BeatmapFile={osu_name}")

# Save updated metadata
df.to_parquet(metadata_path)
