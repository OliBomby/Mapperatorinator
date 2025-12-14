"""
audio_fingerprint.py – Audio recognition for song identification

Provides Shazam-style audio fingerprinting to identify songs from audio files.
Uses AcoustID (offline) as primary method with Shazam API fallback.
Automatically converts Japanese/Chinese characters to romaji.
"""
from __future__ import annotations
import asyncio
import os
import subprocess
import tempfile
from typing import Optional, Tuple

# AcoustID for offline fingerprinting
try:
    import acoustid
    import musicbrainzngs
    ACOUSTID_AVAILABLE = True
except ImportError:
    ACOUSTID_AVAILABLE = False

# Shazam for online fallback
try:
    from shazamio import Shazam
    SHAZAM_AVAILABLE = True
except ImportError:
    SHAZAM_AVAILABLE = False

# Japanese/Korean romanization
try:
    from pykakasi import kakasi
    _kks = kakasi()
    _kks.setMode("H", "a")  # Hiragana to ascii
    _kks.setMode("K", "a")  # Katakana to ascii
    _kks.setMode("J", "a")  # Japanese (Kanji) to ascii
    KAKASI_AVAILABLE = True
except ImportError:
    _kks = None
    KAKASI_AVAILABLE = False

# Configuration
ACOUSTID_KEY = os.getenv("ACOUSTID_KEY", "UT7pFXWpWG")  # Free key from acoustid.org

if ACOUSTID_AVAILABLE:
    musicbrainzngs.set_useragent("Mapperatorinator", "1.0", "https://github.com/OliBomby/Mapperatorinator")


def _to_romaji(text: str) -> str:
    """Convert Japanese/Korean text to romaji/romanization."""
    if not text or not KAKASI_AVAILABLE:
        return text
    
    # Check if text contains non-ASCII characters
    if all(ord(c) < 128 for c in text):
        return text
    
    try:
        result = _kks.convert(text)
        return " ".join([item['hepburn'] for item in result if item.get('hepburn')]) or text
    except Exception:
        return text


def _convert_to_wav(input_path: str) -> Optional[str]:
    """Convert audio file to WAV format for fingerprinting."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Use ffmpeg if available, fallback to pydub
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', input_path, '-ar', '44100', '-ac', '2', tmp_path],
                capture_output=True,
                timeout=60
            )
            if result.returncode == 0:
                return tmp_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback to pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(input_path)
            audio.export(tmp_path, format='wav')
            return tmp_path
        except Exception:
            pass
        
        return None
    except Exception:
        return None


async def _shazam_identify(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Identify song using Shazam API."""
    if not SHAZAM_AVAILABLE:
        return None, None
    
    try:
        shazam = Shazam()
        result = await shazam.recognize(path)
        track = result.get("track")
        if track:
            artist = track.get("subtitle")
            title = track.get("title")
            return artist, title
    except Exception as e:
        print(f"[audio_fingerprint] Shazam error: {e}")
    
    return None, None


def _acoustid_identify(path: str) -> Tuple[Optional[str], Optional[str]]:
    """Identify song using AcoustID + MusicBrainz (offline-capable)."""
    if not ACOUSTID_AVAILABLE or not ACOUSTID_KEY:
        return None, None
    
    try:
        # Generate fingerprint
        fp_result = acoustid.fingerprint_file(path)
        if not fp_result:
            return None, None
        
        fingerprint, duration = fp_result
        if isinstance(fingerprint, (bytes, bytearray)):
            fingerprint = fingerprint.decode("ascii", "ignore")
        
        # Look up on AcoustID
        result = acoustid.lookup(ACOUSTID_KEY, duration, fingerprint, meta="recordings")
        
        if not isinstance(result, dict) or result.get("status") != "ok":
            error = result.get("error", {}).get("message", "unknown") if isinstance(result, dict) else "invalid response"
            print(f"[audio_fingerprint] AcoustID lookup failed: {error}")
            return None, None
        
        # Find best match
        for r in result.get("results", []):
            score = r.get("score", 0)
            for recording in r.get("recordings", []):
                artists = recording.get("artists", [])
                artist = next((a["name"] for a in artists), None)
                title = recording.get("title")
                
                if score > 0.6 and artist and title:
                    return artist, title
        
        return None, None
        
    except acoustid.FingerprintGenerationError:
        # Try converting to WAV
        wav_path = _convert_to_wav(path)
        if wav_path and wav_path != path:
            try:
                result = _acoustid_identify(wav_path)
                return result
            finally:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
        return None, None
        
    except acoustid.AcoustidError as e:
        print(f"[audio_fingerprint] AcoustID error: {e}")
        return None, None
        
    except Exception as e:
        print(f"[audio_fingerprint] Unexpected error: {e}")
        return None, None


def identify_song(path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify a song from an audio file.
    
    Tries AcoustID (offline) first, then Shazam (online) as fallback.
    Automatically converts Japanese/Chinese artist/title to romaji.
    
    Args:
        path: Path to the audio file (.mp3, .ogg, .wav, etc.)
        
    Returns:
        Tuple of (artist, title) or (None, None) if not identified
    """
    if not os.path.isfile(path):
        print(f"[audio_fingerprint] File not found: {path}")
        return None, None
    
    artist, title = None, None
    
    # Try AcoustID first (offline/faster than Shazam)
    if ACOUSTID_AVAILABLE:
        artist, title = _acoustid_identify(path)
        if artist and title:
            print(f"[audio_fingerprint] AcoustID match: {artist} - {title}")
    
    # Fallback to Shazam if needed (slowest, network-dependent)
    if not (artist and title) and SHAZAM_AVAILABLE:
        print(f"[audio_fingerprint] Trying Shazam (this may take a moment)...")
        try:
            artist, title = asyncio.run(_shazam_identify(path))
            if artist and title:
                print(f"[audio_fingerprint] Shazam match: {artist} - {title}")
        except Exception as e:
            print(f"[audio_fingerprint] Shazam async error: {e}")
    
    # Convert to romaji if needed
    if artist and title:
        has_non_ascii = any(ord(c) > 127 for c in (artist + title))
        if has_non_ascii and KAKASI_AVAILABLE:
            artist = _to_romaji(artist)
            title = _to_romaji(title)
            print(f"[audio_fingerprint] Romanized: {artist} - {title}")
    
    return artist, title
    
    return artist, title


def read_artist_title(audio_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Alias for identify_song for backward compatibility.
    """
    return identify_song(audio_path)
