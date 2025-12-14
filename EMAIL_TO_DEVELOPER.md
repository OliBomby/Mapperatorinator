# Email to Olivier Schipper (OliBomby)

**To:** olivierschipper@gmail.com  
**Subject:** Pull Request: Comprehensive Queue System & Workflow Enhancements for Mapperatorinator

---

Hi Olivier,

I hope this email finds you well! I'm reaching out regarding a pull request I've prepared for Mapperatorinator, and I wanted to share some context about the work I've put into it.

First off, I just want to say thank you for creating Mapperatorinator. As someone who's deeply passionate about osu! mapping and the intersection of AI with creative tools, your project has been an absolute game-changer. The ability to generate beatmaps using machine learning is genuinely exciting, and I've spent countless hours exploring what's possible with it.

That passion is exactly what led me to create this fork. What started as me wanting to fix a few bugs turned into a much larger project—I found myself so invested in improving the tool that I ended up adding a complete queue-based workflow system with batch processing capabilities.

## What I've Added

The major additions include:

1. **Multi-Map Queue System** - Users can now queue up multiple generation tasks and process them all at once. This includes a full UI for managing the queue, progress tracking with real-time status updates, task skipping, and detailed parameter views for each queued item.

2. **Mapper List & Style Sampling** - Generate beatmaps in different mapper styles by adding mapper IDs to a persistent list. I integrated the osu! API v2 with thread-safe token handling to automatically fetch mapper usernames for display.

3. **Audio Fingerprinting** - Automatic song detection using AcoustID (offline) with Shazam fallback. This auto-fills the artist and title fields, which saves a lot of manual entry. Includes a full UI with progress indicators and status messages.

4. **Preview Time Picker** - A full audio player modal for selecting the preview point. Built using HTML5 Audio API with a proper timeline slider, volume control, playback speed options (0.5x-2.0x), and a "test preview" function that plays from your selected timestamp for 2 seconds.

5. **Background Image Support** - Users can select a background image that gets automatically embedded into the generated .osz files. Modified the postprocessor to properly include backgrounds in the archive.

6. **Beatmap Set Compilation** - After running a queue, all the generated difficulties can be compiled into a single .osz file with proper osu! naming convention (Artist - Title.osz).

7. **Smart File Renaming** - Output files are automatically renamed to follow the standard osu! naming convention based on detected/entered metadata (filename_utils.py handles all the sanitization).

8. **BF16 Precision Mode** - Optional faster inference (~40-60% speedup) on Ampere+ GPUs (RTX 30xx/40xx) with automatic hardware detection via `/check_bf16_support` endpoint. Adds a checkbox to the UI that auto-checks on supported hardware.

9. **CLI Device Selection** - Added `--cpu` and `--gpu` flags via argparse for flexible device control without editing config files.

10. **Difficulty Name Generator** - Generate themed difficulty names with prefixes, numbered sequences, or custom patterns.

11. **11 New API Routes** - Including `/serve_audio`, `/lookup_mapper_name`, `/check_bf16_support`, `/queue_status`, `/rename_output_file`, `/compile_osz`, and more.

## Bug Fixes

Along the way, I also fixed several issues:

- Updated to use `FileDialog.OPEN` for pywebview 6.0+ compatibility
- Fixed process termination on Windows (it wasn't properly killing child processes)
- Added missing `torchaudio` to requirements.txt (I noticed you also fixed this in a recent commit!)
- Fixed the descriptor dropdown button submitting the form
- Fixed some global variable scope issues in the SSE stream generator
- Added null checks for pywebview API calls so it degrades gracefully in browser mode
- Improved Hydra command quoting for paths and metadata with special characters
- Added preview slider position memory (remembers last position even after stop/cancel)
- Improved confirm dialog to show only after parameters are saved

## Documentation & Setup

- Added comprehensive CUDA installation instructions at the top of requirements.txt for Windows users
- Moved torchaudio to CUDA install comment (since it requires matching CUDA version)

## Code Quality

I've done my best to maintain consistency with your existing code style. Here's the growth breakdown:

| File | Original | Fork | Growth |
|------|----------|------|--------|
| web-ui.py | 654 | 1,386 | +112% |
| static/app.js | 978 | 2,864 | +193% |
| static/style.css | 739 | 1,935 | +162% |
| template/index.html | 346 | 514 | +49% |
| **New Files** | - | 1,766 | - |

All changes are additive—I didn't remove or break any existing functionality. I also made sure to merge the latest upstream changes before submitting.

I've included a detailed CHANGELOG.md in the fork that documents everything comprehensively, complete with file references, code comparisons, and statistics.

## The Pull Request

You can find the fork here: https://github.com/rosacry/Mapperatorinator

I'd be honored if you'd consider reviewing and potentially merging these changes. I understand this is a significant addition, so if you'd prefer to cherry-pick certain features or have any concerns about the implementation, I'm more than happy to discuss and make adjustments.

Either way, I just wanted to express my appreciation for the project. Working on this has been genuinely fun, and I hope these additions can be useful to the broader community.

Thanks for your time, and keep up the amazing work!

Best regards,  
rosacry

---

**P.S.** - If you have any questions about specific implementations or want me to walk through any of the code, just let me know. I'm happy to chat on Discord if that's easier.
Discord: chrig.
