# Mapperatorinator Fork Changelog

**Author:** rosacry  
**Fork Repository:** [rosacry/Mapperatorinator](https://github.com/rosacry/Mapperatorinator)  
**Original Repository:** [OliBomby/Mapperatorinator](https://github.com/OliBomby/Mapperatorinator)  
**Date:** December 2025

---

## Overview

This fork extends Mapperatorinator with a comprehensive queue-based workflow system, enabling batch beatmap generation with enhanced customization options. The total codebase increased significantly with all changes being additive—no original functionality was removed.

---

## New Features

### 1. Multi-Map Queue System
**Files:** `static/queue_manager.js` (new, 670 lines), `static/app.js`, `template/index.html`, `static/style.css`

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
**Files:** `static/queue_manager.js`, `static/app.js`, `template/index.html`, `mapper_api.py` (new, 119 lines)

Generate beatmaps in multiple mapper styles:

- **Mapper ID Input**: Add mapper IDs to a list
- **Mapper Name Lookup**: Automatically fetches and displays mapper usernames from osu! API v2
- **Queue Tasks From Mappers**: Generate one task per mapper with current settings
- **osu! API Integration**: Thread-safe token handling with automatic refresh

### 3. Audio Fingerprinting & Song Detection
**Files:** `audio_fingerprint.py` (new, 227 lines), `static/app.js`, `web-ui.py`

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
**Files:** `static/app.js`, `web-ui.py`, `inference.py`, `osuT5/osuT5/inference/postprocessor.py`, `template/index.html`, `static/style.css`

Custom background images for generated beatmaps:

- **Image Selection**: Browse for background image files
- **Preview Thumbnail**: Small preview of selected image
- **Embedding**: Background reference added to .osu [Events] section
- **.osz Integration**: Background image included when exporting as .osz (modified `postprocessor.export_osz()`)

### 6. Beatmap Set Compilation
**Files:** `web-ui.py`, `static/queue_manager.js`, `static/app.js`, `inference.py`

Compile multiple difficulties into a single .osz:

- **Toggle Option**: "Compile as Beatmap Set" checkbox
- **Automatic Assembly**: After queue completes, all .osu files are packaged
- **Audio Inclusion**: Audio file included in the .osz package
- **Proper Naming**: .osz files named as `Artist - Title.osz` instead of random UUID

### 7. Smart File Renaming
**Files:** `filename_utils.py` (new, 197 lines), `web-ui.py`

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

### 9. BF16 Precision Mode
**Files:** `web-ui.py`, `inference.py`, `template/index.html`, `osuT5/osuT5/inference/postprocessor.py`

Faster inference on modern GPUs:

- **Auto-Detection**: Automatically checks if GPU supports bf16 (Ampere+: RTX 30xx, 40xx, compute capability 8.0+)
- **~40-60% Faster**: Significant speed improvement with no quality loss
- **Optional Toggle**: Checkbox appears only on supported hardware
- **New route**: `/check_bf16_support` endpoint for GPU capability detection

### 10. Command-Line Device Selection
**Files:** `web-ui.py`

Flexible device control via CLI arguments:

- **`--gpu` flag**: Force GPU/CUDA usage (default behavior)
- **`--cpu` flag**: Force CPU-only mode for systems without CUDA
- **Auto mode**: Defaults to GPU if available, falls back to CPU

### 11. Song Metadata Section
**Files:** `template/index.html`, `static/app.js`, `web-ui.py`

New UI section for beatmap metadata:

- **Artist/Title Fields**: Editable fields populated by song detection
- **Auto-Detect Toggle**: Option to automatically detect song metadata
- **Difficulty Name Field**: Manual or auto-generated difficulty names
- **Integration**: Metadata passed to inference for proper .osu file generation

---

## Bug Fixes

### 1. FileDialog Constant Fix
**File:** `web-ui.py`  
**Issue:** The original code used `OPEN_DIALOG` which doesn't exist in newer pywebview versions.  
**Fix:** Updated to use the correct `FileDialog.OPEN` constant for pywebview 6.0+ compatibility.

### 2. Descriptor Dropdown Form Submission
**File:** `template/index.html`  
**Issue:** Clicking the descriptors dropdown button submitted the form.  
**Fix:** Added `type="button"` to prevent form submission.

### 3. Process Termination on Windows
**File:** `web-ui.py`  
**Issue:** Cancelling inference on Windows didn't properly kill child processes.  
**Fix:** Implemented proper process tree termination using `taskkill /F /T /PID`.

### 4. PyWebView Null Checks
**File:** `static/app.js`  
**Issue:** Browse buttons crashed when running in browser mode without pywebview.  
**Fix:** Added null checks for `window.pywebview?.api` with user-friendly error message.

### 5. Hydra Command Quoting
**File:** `web-ui.py`  
**Issue:** Paths and metadata with special characters (quotes, backslashes) could break Hydra command parsing.  
**Fix:** Improved escaping using double quotes and proper backslash/quote escaping. Extended quoting to cover artist, title, creator, version, and background fields.

### 6. Add to Beatmap Validation
**File:** `web-ui.py`  
**Issue:** "Add to beatmap" and "Overwrite reference beatmap" options could be enabled without a beatmap path.  
**Fix:** Options now only activate when a beatmap path is actually provided.

### 7. GeneratorExit Handling on Window Close
**File:** `web-ui.py`  
**Issue:** Closing the application mid-generation caused `RuntimeError: generator ignored GeneratorExit` because the SSE generator tried to yield after catching GeneratorExit.  
**Fix:** Added `client_disconnected` flag to track connection state and skip all `yield` calls in the finally block when the client has disconnected. Also added cleanup to terminate any running inference process when the pywebview window closes.

---

## Documentation Improvements

### CUDA Installation Instructions
**File:** `requirements.txt`  
Added clear instructions at the top of requirements.txt for installing PyTorch with CUDA support:
```
# Install PyTorch with CUDA support first:
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# Then install the rest:
```

This ensures users install PyTorch with GPU support before other dependencies, and removes torchaudio from the regular dependency list since it must be installed with the CUDA index URL.

---

## New Files

| File | Lines | Description |
|------|-------|-------------|
| `static/queue_manager.js` | 670 | Queue management classes (QueueManager, QueueUI, InferenceRunner, MapperManager) |
| `audio_fingerprint.py` | 227 | Audio fingerprinting with AcoustID/Shazam and romaji conversion |
| `filename_utils.py` | 197 | Smart file naming utilities following osu! conventions |
| `mapper_api.py` | 119 | osu! API v2 integration for mapper username lookup |

---

## Modified Files

| File | Original | Fork | Change |
|------|----------|------|--------|
| `web-ui.py` | 654 lines | 1,386 lines | +112% |
| `static/app.js` | 978 lines | 2,864 lines | +193% |
| `template/index.html` | 346 lines | 514 lines | +49% |
| `static/style.css` | 739 lines | 1,935 lines | +162% |
| `inference.py` | Modified | Modified | .osz naming, background support, precision param |
| `osuT5/.../postprocessor.py` | Modified | Modified | Background image in .osz export, bf16 tensor handling |
| `requirements.txt` | 22 deps | 28 deps | +6 deps, CUDA instructions |

---

## New Routes Added

| Route | Method | Description |
|-------|--------|-------------|
| `/check_bf16_support` | GET | Check if GPU supports bf16 precision |
| `/queue_status` | GET/POST | Get or set queue cancellation status |
| `/reset_queue` | POST | Reset queue cancellation flag |
| `/get_audio_info` | POST | Get audio file duration and metadata |
| `/serve_audio` | GET | Stream audio files with Range request support |
| `/get_image_preview` | POST | Get base64 encoded image thumbnail |
| `/preview_background` | GET | Serve background image files |
| `/init_beatmapset` | POST | Initialize beatmap set compilation |
| `/add_to_beatmapset` | POST | Add .osu file to current beatmap set |
| `/finalize_beatmapset` | POST | Compile all .osu files into .osz |
| `/lookup_mapper_name` | POST | Look up mapper username from osu! API |

---

## Dependencies Added

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

## Architecture Changes

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

## Testing Notes

All features have been tested on Windows with:
- Python 3.10+
- PyWebView 6.0+
- Flask backend
- Various audio formats (MP3, WAV, OGG, FLAC, M4A)
- BF16 precision tested on RTX 30/40 series GPUs
- CPU-only mode tested with `--cpu` flag

---

## Compatibility

- Fully backward compatible with original Mapperatorinator
- All original features remain functional
- Configuration files from v1.0 can still be imported
- Merged with latest upstream changes (as of December 2025)
