"""
filename_utils.py – osu! file naming utilities

Provides functions for building and managing osu! beatmap filenames
following the standard naming convention:
    "Artist - Title (Creator) [Difficulty].osu"
"""
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional


def _slugify(text: str, allow_unicode: bool = True) -> str:
    """
    Make a filename-safe slug while preserving spaces.
    
    Args:
        text: Input text to slugify
        allow_unicode: Whether to allow unicode characters
        
    Returns:
        Filename-safe string
    """
    text = text.strip()
    
    if allow_unicode:
        text = unicodedata.normalize("NFKC", text)
    else:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    
    # Remove characters forbidden in Windows/POSIX filenames
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", text)
    
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    
    # Limit length (Windows MAX_PATH consideration)
    return text[:180]


def build_osu_filename(
    artist: str,
    title: str,
    creator: str,
    difficulty: str,
    ext: str = ".osu"
) -> str:
    """
    Build an osu! filename following the standard convention.
    
    Args:
        artist: Song artist name
        title: Song title
        creator: Beatmap creator/mapper name
        difficulty: Difficulty name
        ext: File extension (default .osu)
        
    Returns:
        Filename like "Artist - Title (Creator) [Difficulty].osu"
    """
    # Sanitize each component
    artist = _slugify(artist) or "Unknown Artist"
    title = _slugify(title) or "Unknown Title"
    creator = _slugify(creator) or "Mapperatorinator"
    difficulty = _slugify(difficulty) or "Normal"
    
    filename = f"{artist} - {title} ({creator}) [{difficulty}]{ext}"
    return _slugify(filename, allow_unicode=True)


def rename_output(
    old_path: str,
    artist: str,
    title: str,
    creator: str,
    difficulty: str
) -> str:
    """
    Rename a beatmap file on disk to follow osu! naming convention.
    
    Args:
        old_path: Current file path
        artist: Song artist
        title: Song title
        creator: Beatmap creator
        difficulty: Difficulty name
        
    Returns:
        New absolute path after rename
    """
    directory = os.path.dirname(old_path)
    ext = Path(old_path).suffix
    
    new_name = build_osu_filename(artist, title, creator, difficulty, ext=ext)
    new_path = os.path.join(directory, new_name)
    
    # Handle filename collisions
    counter = 1
    base, suffix = os.path.splitext(new_path)
    while os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(old_path):
        new_path = f"{base} ({counter}){suffix}"
        counter += 1
    
    # Only rename if paths differ
    if os.path.abspath(old_path) != os.path.abspath(new_path):
        os.rename(old_path, new_path)
    
    return new_path


def get_default_difficulty_label(star_rating: float) -> str:
    """
    Get the standard osu! difficulty label based on star rating.
    
    Args:
        star_rating: The difficulty star rating
        
    Returns:
        Standard difficulty name (Easy, Normal, Hard, Insane, Expert)
    """
    if star_rating < 2.0:
        return "Easy"
    elif star_rating < 2.7:
        return "Normal"
    elif star_rating < 4.0:
        return "Hard"
    elif star_rating < 5.3:
        return "Insane"
    elif star_rating < 6.5:
        return "Expert"
    else:
        return "Expert+"


def compose_diff_name(
    form_data: dict,
    mapper_username: Optional[str] = None
) -> str:
    """
    Create a difficulty name string based on form data and mapper info.
    
    Priority:
    1. Manual difficulty_name from form → use as-is
    2. Mapper username provided → "mappername/cfg/temp/top_p/seed"
    3. No mapper → "Mapperatorinator V<model>"
    
    Args:
        form_data: Dict containing form field values
        mapper_username: Resolved mapper username (or None)
        
    Returns:
        Difficulty name string
    """
    # 1. Manual override takes priority
    custom_name = form_data.get("difficulty_name", "").strip()
    if custom_name:
        return custom_name
    
    # 2. If we have a mapper, create detailed name
    if mapper_username:
        cfg = form_data.get("cfg_scale") or "NAN"
        temp = form_data.get("temperature") or "NAN"
        top_p = form_data.get("top_p") or "NAN"
        seed = form_data.get("seed") or "NAN"
        
        # Format: mappername/cfg/temp/top_p/seed
        return f"{mapper_username}/{cfg}/{temp}/{top_p}/{seed}"
    
    # 3. Default to model version
    model = form_data.get("model", "v30")
    model_ver = model.lstrip("v").upper() if model else "30"
    return f"Mapperatorinator V{model_ver}"


def compose_mapper_difficulty_name(
    mapper_name: str,
    star_rating: float
) -> str:
    """
    Create a difficulty name like "browiec's Hard" based on mapper and star rating.
    
    Args:
        mapper_name: The mapper's username
        star_rating: The target star rating
        
    Returns:
        Difficulty name like "mappername's Hard"
    """
    base_diff = get_default_difficulty_label(star_rating)
    
    # Handle possessive apostrophe correctly
    if mapper_name.endswith('s') or mapper_name.endswith('S'):
        return f"{mapper_name}' {base_diff}"
    else:
        return f"{mapper_name}'s {base_diff}"
