# Mapperatorinator Fork Changelog

**Author:** rosacry  
**Fork Repository:** [rosacry/Mapperatorinator](https://github.com/rosacry/Mapperatorinator)  
**Original Repository:** [OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)  
**Date:** December 2024

---

## Overview

This fork extends Mapperatorinator with a comprehensive queue-based workflow system, enabling batch beatmap generation with enhanced customization options. The total codebase increased by approximately **177%** (from ~2,742 lines to ~7,597 lines) with all changes being additive—no original functionality was removed.

---

## 🚀 New Features

### 1. Multi-Map Queue System
**Files:** `static/queue_manager.js` (new, 671 lines), `static/app.js`, `template/index.html`, `static/style.css`

A complete queue management system for batch beatmap generation:

- **Add to Queue**: Build a queue of generation tasks from the current form settings
- **Run Queue**: Process all queued tasks sequentially with progress tracking
- **Cancel/Skip**: Cancel entire queue or skip individual tasks mid-generation
- **Clear Queue**: Remove all queued tasks
- **Task Details**: Expandable view showing all parameters for each queued task:
  - Basic Settings (model, gamemode, difficulty, etc.)
  - Difficulty Metadata (HP, CS, OD, AR, etc.)
  - Advanced Settings (CFG scale, temperature, seed, etc.)
  - Generation Interval (start/end time)
  - Descriptors (positive/negative)

### 2. Mapper List & Style Sampling
**Files:** `static/queue_manager.js`, `static/app.js`, `template/index.html`, `mapper_api.py` (new)

Generate beatmaps in multiple mapper styles:

- **Mapper ID Input**: Add mapper IDs to a list
- **Mapper Name Lookup**: Automatically fetches and displays mapper usernames from osu! API v2
- **Queue Tasks From Mappers**: Generate one task per mapper with current settings
- **osu! API Integration**: Thread-safe token handling with automatic refresh

### 3. Audio Fingerprinting & Song Detection
**Files:** `audio_fingerprint.py` (rewritten, 227 lines), `static/app.js`, `web-ui.py`

Automatic song identification:

- **Primary Method**: AcoustID (offline fingerprinting) via MusicBrainz
- **Fallback Method**: Shazam API (online)
- **Character Conversion**: Automatic Japanese/Chinese to romaji conversion using pykakasi
- **Auto-Fill**: Detected artist/title automatically populate metadata fields
- **Cancellable**: Detection can be cancelled mid-process
- **Caching**: Results cached to avoid redundant API calls

### 4. Preview Time Picker
**Files:** `static/app.js`, `static/style.css`, `web-ui.py`, `template/index.html`

Interactive audio preview system:

- **Audio Player Modal**: Full-featured audio player with:
  - Play/Pause controls
  - Timeline slider with click-to-seek and drag support
  - Time display (current / total) in mm:ss format
  - Manual millisecond and second input fields
  - Volume control slider
  - Playback speed options (25%, 50%, 75%, 100%)
- **Test Preview**: Play 10 seconds from the selected point
- **Position Memory**: Reopening the picker remembers the last position
- **Apply**: Set the preview time which embeds into generated .osu files

### 5. Background Image Support
**Files:** `static/app.js`, `web-ui.py`, `template/index.html`, `static/style.css`

Custom background images for generated beatmaps:

- **Image Selection**: Browse for background image files
- **Preview Thumbnail**: Small preview of selected image
- **Auto-Copy**: Background image automatically copied to output directory
- **Embedding**: Background reference added to .osu [Events] section

### 6. Beatmap Set Compilation
**Files:** `web-ui.py`, `static/queue_manager.js`, `static/app.js`

Compile multiple difficulties into a single .osz:

- **Toggle Option**: "Compile as Beatmap Set" checkbox
- **Automatic Assembly**: After queue completes, all .osu files are packaged
- **Audio Inclusion**: Audio file included in the .osz package
- **Standard Structure**: Creates properly formatted osu! beatmap package

### 7. Smart File Renaming
**Files:** `filename_utils.py` (new, ~150 lines), `web-ui.py`

Automatic output file renaming:

- **Standard Format**: `Artist - Title (Creator) [Difficulty].osu`
- **Metadata Sources**: From detected song, mapper lookup, or form input
- **Difficulty Names**: Auto-generated based on star rating:
  - Easy (< 2★), Normal (< 2.7★), Hard (< 4★)
  - Insane (< 5.3★), Expert (< 6.5★), Expert+ (≥ 6.5★)
- **Safe Characters**: Invalid filename characters automatically sanitized

### 8. Enhanced Configuration Management
**Files:** `static/app.js`

Improved config export/import:

- **Version 1.1 Format**: New config version with expanded fields
- **Additional Fields**: Exports/imports:
  - Preview time
  - Background image path
  - Beatmap customization settings
- **Queue & Mapper Clearing**: Reset function now clears queue and mapper list
- **Custom Confirm Dialog**: Styled modal for reset confirmation (replaces native `confirm()`)

---

## 🐛 Bug Fixes

### 1. FileDialog Constant Fix
**File:** `web-ui.py`  
**Issue:** The original code used `OPEN_DIALOG` which doesn't exist in newer pywebview versions.  
**Fix:** Updated to use the correct `FileDialog.OPEN` constant for pywebview 6.0+ compatibility.

### 2. Missing torchaudio Dependency
**File:** `requirements.txt`  
**Issue:** `torchaudio` was not listed in requirements but is required for audio processing.  
**Fix:** Added `torchaudio` to requirements.txt.  
*(Note: This was later also fixed in upstream)*

### 3. Descriptor Dropdown Form Submission
**File:** `template/index.html`  
**Issue:** Clicking the descriptors dropdown button submitted the form.  
**Fix:** Added `type="button"` to prevent form submission.

### 4. Process Termination on Windows
**File:** `web-ui.py`  
**Issue:** Cancelling inference on Windows didn't properly kill child processes.  
**Fix:** Implemented proper process tree termination using `taskkill /F /T /PID`.

### 5. Global Variable Scope
**File:** `web-ui.py`  
**Issue:** Beatmapset variables modified inside generator function without `global` declaration.  
**Fix:** Added proper `global` declarations for `beatmapset_enabled`, `beatmapset_files`, `beatmapset_audio_path`, `beatmapset_output_dir`.

### 6. Element ID Mismatch
**File:** `static/queue_manager.js`  
**Issue:** JavaScript referenced `new_mapper_id` but HTML has `add-mapper-id`.  
**Fix:** Updated JavaScript to use correct element ID.

### 7. PyWebView Null Checks
**File:** `static/app.js`  
**Issue:** Browse buttons crashed when running in browser mode without pywebview.  
**Fix:** Added null checks for `window.pywebview?.api` with user-friendly error message.

### 8. Preview Slider Reset Bug
**File:** `static/app.js`  
**Issue:** Timeline slider would reset to start position when clicked or dragged.  
**Fix:** Rewrote slider handling with `preventDefault()`, manual position calculation, and document-level mouse tracking.

---

## 📁 New Files

| File | Description |
|------|-------------|
| `static/queue_manager.js` | Queue management classes (QueueManager, QueueUI, InferenceRunner) |
| `audio_fingerprint.py` | Audio fingerprinting with AcoustID/Shazam |
| `filename_utils.py` | Smart file naming utilities |
| `mapper_api.py` | osu! API v2 integration for mapper lookup |

---

## 📊 Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| `web-ui.py` | 655 lines | 1,151 lines | +76% |
| `static/app.js` | 979 lines | 2,733 lines | +179% |
| `template/index.html` | 347 lines | 509 lines | +47% |
| `static/style.css` | 739 lines | 1,876 lines | +154% |
| `requirements.txt` | 22 entries | 28 entries | +6 |
| **New Files** | 0 | ~1,300 lines | — |
| **Total** | ~2,742 lines | ~7,597 lines | **+177%** |

---

## 🔧 Dependencies Added

```
# Queue system & audio fingerprinting
requests>=2.32
shazamio
aiofiles
pyacoustid
musicbrainzngs
pykakasi
```

---

## 🎯 Architecture Changes

The fork transforms the single-inference workflow into a queue-based batch system:

```
Original Flow:
  Form → Submit → Single Inference → Output

Fork Flow:
  Form → Add to Queue → [Multiple Tasks] → Run Queue → Batch Processing
          ↓
  Mapper List → Generate Tasks from Mappers
          ↓
  Audio Fingerprint → Auto-fill Artist/Title
          ↓
  Preview Picker → Set Preview Time & Background
          ↓
  Beatmap Set → Compile .osz
```

---

## 🧪 Testing Notes

All features have been tested on Windows with:
- Python 3.10+
- PyWebView 6.0+
- Flask backend
- Various audio formats (MP3, WAV, OGG, FLAC, M4A)

---

## 📝 Compatibility

- Fully backward compatible with original Mapperatorinator
- All original features remain functional
- Configuration files from v1.0 can still be imported
- Merged with latest upstream changes (as of December 2024)
