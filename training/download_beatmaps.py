import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tqdm import tqdm

DEFAULT_MAPPERS_PATH = "datasets/beatmap_users.json"

API_BASE = "https://osu.ppy.sh/api/v2"
TOKEN_URL = "https://osu.ppy.sh/oauth/token"
# catboy.best is a commonly used osu! beatmap mirror
MIRROR_URL = "https://catboy.best/d/{beatmapset_id}"


def get_oauth_token(client_id: str, client_secret: str) -> str:
    """Get an OAuth2 client credentials token from osu! API."""
    resp = requests.post(TOKEN_URL, json={
        "client_id": int(client_id),
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "public",
    })
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_beatmapsets(token: str, year_start: int, year_end: int, status: str = "ranked") -> list[dict]:
    """Search for all beatmap sets ranked/loved in the given year range."""
    headers = {"Authorization": f"Bearer {token}"}
    all_sets = []
    cursor_string = None

    # osu! API search uses created/ranked date filters
    # We search year by year for better pagination
    for year in range(year_start, year_end + 1):
        print(f"\nSearching for {status} beatmaps from {year}...")
        cursor_string = None
        year_count = 0

        while True:
            params = {
                "s": status,
                "sort": "ranked_asc",
                # Filter by ranked date range
                "q": f"ranked>={year}-01-01 ranked<{year + 1}-01-01",
            }
            if cursor_string:
                params["cursor_string"] = cursor_string

            resp = requests.get(
                f"{API_BASE}/beatmapsets/search",
                headers=headers,
                params=params,
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                print(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()

            beatmapsets = data.get("beatmapsets", [])
            if not beatmapsets:
                break

            all_sets.extend(beatmapsets)
            year_count += len(beatmapsets)

            cursor_string = data.get("cursor_string")
            if not cursor_string:
                break

            # Respect rate limits
            time.sleep(0.5)

        print(f"  Found {year_count} {status} beatmap sets from {year}")

    return all_sets


def load_mapper_user_ids(mappers_path: str) -> set[int]:
    """Load unique mapper user IDs from beatmap_users.json."""
    path = Path(mappers_path)
    if not path.exists():
        print(f"Error: Mappers file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    user_ids = {entry["user_id"] for entry in data}
    print(f"Loaded {len(user_ids)} unique mapper IDs from {path}")
    return user_ids


def filter_by_mappers(beatmapsets: list[dict], mapper_ids: set[int]) -> list[dict]:
    """Filter beatmap sets to only those created by known mappers."""
    filtered = [s for s in beatmapsets if s.get("user_id") in mapper_ids]
    print(f"  Mapper filter: {len(filtered)}/{len(beatmapsets)} sets match known mappers")
    return filtered


def download_osz(beatmapset_id: int, output_dir: Path, existing_ids: set) -> bool:
    """Download a single .osz file from the mirror."""
    if beatmapset_id in existing_ids:
        return False

    osz_path = output_dir / f"{beatmapset_id}.osz"
    if osz_path.exists():
        return False

    url = MIRROR_URL.format(beatmapset_id=beatmapset_id)
    try:
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()

        with open(osz_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True

    except Exception as e:
        # Clean up partial downloads
        if osz_path.exists():
            osz_path.unlink()
        print(f"  Failed to download {beatmapset_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download osu! beatmaps for training")
    parser.add_argument("--client-id", default=os.environ.get("OSU_CLIENT_ID"),
                        help="osu! OAuth client ID (or set OSU_CLIENT_ID env var)")
    parser.add_argument("--client-secret", default=os.environ.get("OSU_CLIENT_SECRET"),
                        help="osu! OAuth client secret (or set OSU_CLIENT_SECRET env var)")
    parser.add_argument("--output", default="./training/osz_2024_2025",
                        help="Output directory for .osz files")
    parser.add_argument("--year-start", type=int, default=2024,
                        help="Start year (inclusive)")
    parser.add_argument("--year-end", type=int, default=2025,
                        help="End year (inclusive)")
    parser.add_argument("--status", nargs="+", default=["ranked", "loved"],
                        help="Beatmap statuses to download (ranked, loved, qualified)")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Only fetch metadata, don't download .osz files")
    parser.add_argument("--mappers-path", default=DEFAULT_MAPPERS_PATH,
                        help="Path to beatmap_users.json for mapper filtering (default: datasets/beatmap_users.json)")
    parser.add_argument("--filter-mappers", action="store_true",
                        help="Only download maps from mappers in beatmap_users.json")
    args = parser.parse_args()

    if not args.client_id or not args.client_secret:
        print("Error: osu! OAuth credentials required.")
        print("Either pass --client-id and --client-secret, or set OSU_CLIENT_ID and OSU_CLIENT_SECRET env vars.")
        print("Get credentials at: https://osu.ppy.sh/home/account/edit (OAuth section)")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get OAuth token
    print("Authenticating with osu! API...")
    token = get_oauth_token(args.client_id, args.client_secret)
    print("Authenticated successfully.")

    # Search for beatmap sets
    all_sets = []
    for status in args.status:
        sets = search_beatmapsets(token, args.year_start, args.year_end, status)
        all_sets.extend(sets)

    # Deduplicate by ID
    seen = set()
    unique_sets = []
    for s in all_sets:
        if s["id"] not in seen:
            seen.add(s["id"])
            unique_sets.append(s)
    all_sets = unique_sets

    print(f"\nTotal unique beatmap sets found: {len(all_sets)}")

    # Optionally filter to known mappers only
    if args.filter_mappers:
        mapper_ids = load_mapper_user_ids(args.mappers_path)
        all_sets = filter_by_mappers(all_sets, mapper_ids)
        print(f"After mapper filter: {len(all_sets)} sets")

    # Save metadata
    metadata_path = output_dir / "beatmapset_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_sets, f, default=str)
    print(f"Saved metadata to {metadata_path}")

    if args.metadata_only:
        print("Metadata-only mode, skipping downloads.")
        return

    # Download .osz files
    existing_ids = {int(p.stem) for p in output_dir.glob("*.osz")}
    to_download = [s for s in all_sets if s["id"] not in existing_ids]
    print(f"Already downloaded: {len(existing_ids)}")
    print(f"To download: {len(to_download)}")

    downloaded = 0
    failed = 0
    for beatmapset in tqdm(to_download, desc="Downloading"):
        if download_osz(beatmapset["id"], output_dir, set()):
            downloaded += 1
        else:
            failed += 1
        # Rate limit downloads
        time.sleep(0.3)

    print(f"\nDownload complete: {downloaded} new, {failed} failed, {len(existing_ids)} already had")
    print(f"Total .osz files: {len(list(output_dir.glob('*.osz')))}")
    print(f"\nNext step: Create dataset with Mapperator:")
    print(f"  Mapperator.ConsoleApp.exe dataset2 -i \"{output_dir}\" -o \"./training/dataset_2024_2025\"")


if __name__ == "__main__":
    main()
