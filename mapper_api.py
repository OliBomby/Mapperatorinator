"""
mapper_api.py – helper for osu! API username lookup

Provides functionality to look up osu! mapper usernames from their user IDs
using the osu! API v2. Includes token caching and thread-safe token refresh.
"""
from __future__ import annotations
import os
import time
import threading
from typing import Optional

import requests

_OSU_TOKEN: str | None = None
_TOKEN_EXPIRES: float = 0.0
_LOCK = threading.Lock()


def _refresh_token() -> str:
    """
    Fetch a client-credentials token and cache it.
    
    Environment vars or app config:
        OSU_CLIENT_ID     – e.g. 42371
        OSU_CLIENT_SECRET – your secret
    """
    global _OSU_TOKEN, _TOKEN_EXPIRES
    cid = os.getenv("OSU_CLIENT_ID")
    sec = os.getenv("OSU_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("osu! API credentials not set (OSU_CLIENT_ID, OSU_CLIENT_SECRET)")
    
    try:
        r = requests.post(
            "https://osu.ppy.sh/oauth/token",
            json={
                "client_id": int(cid),
                "client_secret": sec,
                "grant_type": "client_credentials",
                "scope": "public",
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        _OSU_TOKEN = data["access_token"]
        _TOKEN_EXPIRES = time.time() + data["expires_in"] - 60  # refresh 1 min early
        return _OSU_TOKEN
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to refresh osu! API token: {e}")


def _get_token() -> str:
    """Get a valid API token, refreshing if necessary."""
    with _LOCK:
        if _OSU_TOKEN is None or time.time() >= _TOKEN_EXPIRES:
            return _refresh_token()
        return _OSU_TOKEN


def lookup_username(mapper_id: int | str) -> Optional[str]:
    """
    Return mapper's current username or None if not found.
    
    Args:
        mapper_id: The osu! user ID to look up
        
    Returns:
        The username string or None if lookup fails
    """
    try:
        token = _get_token()
    except Exception as e:
        print(f"[mapper_api] Token error: {e}")
        return None
    
    url = f"https://osu.ppy.sh/api/v2/users/{mapper_id}/osu"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        if r.status_code == 404:
            print(f"[mapper_api] User {mapper_id} not found")
            return None
        if r.status_code != 200:
            print(f"[mapper_api] Lookup failed: {r.status_code} - {r.text[:100]}")
            return None
        return r.json().get("username")
    except requests.RequestException as e:
        print(f"[mapper_api] Request error: {e}")
        return None


def lookup_user_info(mapper_id: int | str) -> Optional[dict]:
    """
    Return full user info dict or None if not found.
    Useful for getting additional info like avatar, country, etc.
    """
    try:
        token = _get_token()
    except Exception as e:
        print(f"[mapper_api] Token error: {e}")
        return None
    
    url = f"https://osu.ppy.sh/api/v2/users/{mapper_id}/osu"
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10
        )
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None
