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

1. **Multi-Map Queue System** - Users can now queue up multiple generation tasks and process them all at once. This includes progress tracking, task skipping, and detailed parameter views for each queued item.

2. **Mapper List & Style Sampling** - Generate beatmaps in different mapper styles by adding mapper IDs to a list. I integrated the osu! API v2 to automatically fetch mapper usernames for display.

3. **Audio Fingerprinting** - Automatic song detection using AcoustID (offline) with Shazam fallback. This auto-fills the artist and title fields, which saves a lot of manual entry.

4. **Preview Time Picker** - A full audio player modal for selecting the preview point. It has a proper timeline slider, volume control, playback speed options, and a "test preview" function.

5. **Background Image Support** - Users can select a background image that gets automatically embedded into the generated .osu files.

6. **Beatmap Set Compilation** - After running a queue, all the generated difficulties can be compiled into a single .osz file.

7. **Smart File Renaming** - Output files are automatically renamed to follow the standard osu! naming convention based on detected/entered metadata.

8. **BF16 Precision Mode** - Optional faster inference (~40-60% speedup) on Ampere+ GPUs with automatic hardware detection.

9. **CLI Device Selection** - Added `--cpu` and `--gpu` flags for flexible device control.

## Bug Fixes

Along the way, I also fixed several issues:

- Updated to use `FileDialog.OPEN` for pywebview 6.0+ compatibility
- Fixed process termination on Windows (it wasn't properly killing child processes)
- Added missing `torchaudio` to requirements.txt (I noticed you also fixed this in a recent commit!)
- Fixed the descriptor dropdown button submitting the form
- Fixed some global variable scope issues in the SSE stream generator
- Added null checks for pywebview API calls so it degrades gracefully in browser mode
- Improved Hydra command quoting for paths and metadata with special characters

## Code Quality

I've done my best to maintain consistency with your existing code style. The total codebase grew from about 2,700 lines to 7,600 lines, but all changes are additive—I didn't remove or break any existing functionality. I also made sure to merge the latest upstream changes before submitting.

I've included a detailed CHANGELOG.md in the fork that documents everything comprehensively, complete with file references, code comparisons, and statistics.

## The Pull Request

You can find the fork here: https://github.com/rosacry/Mapperatorinator

I'd be honored if you'd consider reviewing and potentially merging these changes. I understand this is a significant addition, so if you'd prefer to cherry-pick certain features or have any concerns about the implementation, I'm more than happy to discuss and make adjustments.

Either way, I just wanted to express my appreciation for the project. Working on this has been genuinely fun, and I hope these additions can be useful to the broader community.

Thanks for your time, and keep up the amazing work!

Best regards,  
rosacry

---

**P.S.** - If you have any questions about specific implementations or want me to walk through any of the code, just let me know. I'm happy to hop on a call or chat on Discord if that's easier.
