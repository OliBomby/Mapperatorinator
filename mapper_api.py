"""
mapper_api.py – helper for osu! username lookup

Provides functionality to look up osu! mapper usernames from their user IDs
by scraping public profile pages. No API keys required.
"""
from __future__ import annotations
import re
from typing import Optional

import requests

# Simple in-memory cache for usernames
_USERNAME_CACHE: dict[str, str] = {}


def lookup_username(mapper_id: int | str) -> Optional[str]:
    """
    Return mapper's current username or None if not found.
    
    Scrapes the public osu! profile page to extract the username.
    No API keys required.
    
    Args:
        mapper_id: The osu! user ID to look up
        
    Returns:
        The username string or None if lookup fails
    """
    mapper_id = str(mapper_id).strip()
    if not mapper_id:
        return None
    
    # Check cache first
    if mapper_id in _USERNAME_CACHE:
        return _USERNAME_CACHE[mapper_id]
    
    url = f"https://osu.ppy.sh/users/{mapper_id}"
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            timeout=10
        )
        if r.status_code == 404:
            print(f"[mapper_api] User {mapper_id} not found")
            return None
        if r.status_code != 200:
            print(f"[mapper_api] Lookup failed: {r.status_code}")
            return None
        
        # Extract username from page title: "username · player info | osu!"
        match = re.search(r'<title>(.+?)\s*·\s*player', r.text)
        if match:
            username = match.group(1).strip()
            _USERNAME_CACHE[mapper_id] = username
            return username
        
        # Fallback: try to find in JSON data embedded in page
        match = re.search(r'"username"\s*:\s*"([^"]+)"', r.text)
        if match:
            username = match.group(1)
            _USERNAME_CACHE[mapper_id] = username
            return username
            
        print(f"[mapper_api] Could not parse username from page")
        return None
    except requests.RequestException as e:
        print(f"[mapper_api] Request error: {e}")
        return None


def lookup_user_info(mapper_id: int | str) -> Optional[dict]:
    """
    Return basic user info dict or None if not found.
    
    Since we're scraping instead of using the API, this returns
    limited information extracted from the profile page.
    """
    username = lookup_username(mapper_id)
    if not username:
        return None
    
    return {
        "id": int(mapper_id),
        "username": username
    }
