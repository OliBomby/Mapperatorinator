import excepthook  # noqa
import functools
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
import datetime
from typing import Callable, Any, Tuple, Dict, Optional

import webview
import werkzeug.serving
from flask import Flask, render_template, request, Response, jsonify

from config import InferenceConfig
from inference import autofill_paths

# Queue system imports
try:
    from mapper_api import lookup_username
    from audio_fingerprint import identify_song
    from filename_utils import rename_output, compose_diff_name
    QUEUE_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Queue features not available: {e}")
    QUEUE_FEATURES_AVAILABLE = False
    def lookup_username(mapper_id): return None
    def identify_song(path): return None, None
    def rename_output(*args): return args[0]
    def compose_diff_name(*args): return "Mapperatorinator"

script_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(script_dir, 'template')
static_folder = os.path.join(script_dir, 'static')

if not os.path.isdir(static_folder):
    print(f"Warning: Static folder not found at {static_folder}. Ensure it exists and contains your CSS/images.")


# Set Flask environment to production before initializing Flask app to silence warning
# os.environ['FLASK_ENV'] = 'production' # Removed, using cli patch instead

# --- Werkzeug Warning Suppressor Patch ---
def _ansi_style_supressor(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        # Check if the first argument is the specific warning string
        if args:
            first_arg = args[0]
            if isinstance(first_arg, str) and first_arg.startswith('WARNING: This is a development server.'):
                return ''  # Return empty string to suppress
        # Otherwise, call the original function
        return func(*args, **kwargs)

    return wrapper


# Apply the patch before Flask initialization
# noinspection PyProtectedMember
werkzeug.serving._ansi_style = _ansi_style_supressor(werkzeug.serving._ansi_style)
# --- End Patch ---

if hasattr(webview, "FileDialog"):
    OPEN_DIALOG = webview.FileDialog.OPEN
    FOLDER_DIALOG = webview.FileDialog.FOLDER
    SAVE_DIALOG = webview.FileDialog.SAVE
else:
    OPEN_DIALOG = webview.OPEN_DIALOG
    FOLDER_DIALOG = webview.FOLDER_DIALOG
    SAVE_DIALOG = webview.SAVE_DIALOG


def parse_file_dialog_result(result):
    if not result:
        return None
    return result[0] if isinstance(result, (list, tuple)) else result

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.secret_key = os.urandom(24)  # Set a secret key for Flask

# ── osu! API credentials (for mapper lookup) ────────────────────────
# You can override these via environment variables
OSU_CLIENT_ID = os.getenv("OSU_CLIENT_ID", "42371")
OSU_CLIENT_SECRET = os.getenv("OSU_CLIENT_SECRET", "wNDRUcvRGjzk39LpT9zR5LNKlA8fFRREoUo3eh8T")

app.config["OSU_CLIENT_ID"] = OSU_CLIENT_ID
app.config["OSU_CLIENT_SECRET"] = OSU_CLIENT_SECRET

# Mirror to environment for helper modules
os.environ.setdefault("OSU_CLIENT_ID", OSU_CLIENT_ID)
os.environ.setdefault("OSU_CLIENT_SECRET", OSU_CLIENT_SECRET)

# ── Shared state for inference process and queue ────────────────────
last_form_data: dict = {}  # Remember form data for file renaming
queue_cancelled: bool = False  # Flag to cancel queue processing
song_detection_cache: dict = {}  # Cache for song detection results {audio_path: (artist, title)}


def apply_beatmap_customizations(osu_path, preview_time=None, background_path=None):
    """Apply preview time and background image to a .osu file."""
    if not os.path.exists(osu_path):
        print(f"Warning: Cannot customize .osu file - not found: {osu_path}")
        return
    
    try:
        with open(osu_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        new_lines = []
        in_general = False
        in_events = False
        preview_written = False
        bg_written = False
        bg_name = None
        
        # If we have a background, copy it to the output directory
        if background_path and os.path.exists(background_path):
            import shutil
            output_dir = os.path.dirname(osu_path)
            bg_name = os.path.basename(background_path)
            dest_bg = os.path.join(output_dir, bg_name)
            if not os.path.exists(dest_bg):
                shutil.copy2(background_path, dest_bg)
                print(f"Copied background to: {dest_bg}")
        
        for line in lines:
            stripped = line.strip()
            
            # Track sections
            if stripped == '[General]':
                in_general = True
                in_events = False
                new_lines.append(line)
                continue
            elif stripped == '[Events]':
                in_events = True
                in_general = False
                new_lines.append(line)
                continue
            elif stripped.startswith('[') and (in_general or in_events):
                # Leaving section
                if in_general and preview_time and not preview_written:
                    new_lines.append(f"PreviewTime: {preview_time}\n")
                if in_events and bg_name and not bg_written:
                    new_lines.append(f'0,0,"{bg_name}",0,0\n')
                in_general = False
                in_events = False
            
            # Modify preview time in [General]
            if in_general and stripped.startswith('PreviewTime:'):
                if preview_time:
                    new_lines.append(f"PreviewTime: {preview_time}\n")
                    preview_written = True
                else:
                    new_lines.append(line)
                continue
            
            # Modify/add background in [Events]
            if in_events and (stripped.startswith('0,0,"') or stripped.startswith("0,0,'")):
                if bg_name:
                    new_lines.append(f'0,0,"{bg_name}",0,0\n')
                    bg_written = True
                else:
                    new_lines.append(line)
                continue
            
            new_lines.append(line)
        
        # Write back
        with open(osu_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"Applied customizations to: {osu_path}")
        
    except Exception as e:
        print(f"Warning: Failed to apply customizations to {osu_path}: {e}")

# --- pywebview API Class ---
class Api:
    # No __init__ needed as we get the window dynamically
    def save_file(self, filename):
        """Opens a save file dialog and returns the selected file path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(SAVE_DIALOG, save_filename=filename)
        print(f"File dialog result: {result}")  # Debugging
        return parse_file_dialog_result(result)

    def browse_file(self, file_types=None):
        """Opens a file dialog and returns the selected file path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None

        current_window = webview.windows[0]

        # File type filter
        try:
            if file_types and isinstance(file_types, list):
                file_types = tuple(file_types)

            result = current_window.create_file_dialog(
                OPEN_DIALOG,
                file_types=file_types
            )
        except Exception:
            result = current_window.create_file_dialog(OPEN_DIALOG)

        return parse_file_dialog_result(result)

    def browse_folder(self):
        """Opens a folder dialog and returns the selected folder path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(FOLDER_DIALOG)
        print(f"Folder dialog result: {result}")  # Debugging
        # FOLDER_DIALOG also returns a tuple containing the path
        return parse_file_dialog_result(result)


# --- Shared State for Inference Process ---
current_process: subprocess.Popen | None = None
process_lock = threading.Lock()  # Lock for accessing current_process safely


# --- Helper Function (same as original Flask) ---
def dq_quote(s):
    """Wrap the string in double quotes and escape inner double quotes."""
    # Basic check if it looks quoted
    if isinstance(s, str) and s.startswith('"') and s.endswith('"'):
        return s
    return '"' + str(s).replace('"', '\\"') + '"'


# Helper function for double-single quotes
def dsq_quote(s):
    """
    Prepares a path string for Hydra command-line override.
    Wraps the path in single quotes, escaping internal single quotes (' -> \\').
    Then wraps the result in double quotes for shell safety.
    Example: "C:/My's Folder" becomes "\"'C:/My\\'s Folder'\""
    """
    path_str = str(s)

    # 1. Escape internal single quotes within the path string itself
    escaped_path = path_str.replace("'", "\\'")  # Replace ' with \'

    # 2. Wrap the escaped path string in single quotes
    inner_quoted = "'" + escaped_path + "'"

    # 3. Wrap the single-quoted string in double quotes for the shell command line
    return '"' + inner_quoted + '"'


def format_list_arg(items):
    """Formats a list of strings for the command line argument."""
    return "[" + ",".join("'" + str(d) + "'" for d in items) + "]"


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Jinja rendering is now handled by Flask's render_template
    return render_template('index.html')


@app.route('/start_inference', methods=['POST'])
def start_inference():
    """Starts the inference process based on form data."""
    global current_process, last_form_data
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({"status": "error", "message": "Process already running"}), 409  # Conflict

        # Save form data for file renaming after completion
        last_form_data = request.form.to_dict(flat=True)

        # --- Construct Command List (shell=False) ---
        python_executable = sys.executable  # Get path to current Python interpreter
        cmd = [python_executable, "inference.py", "-cn"]

        # Get the model name from the form
        model_name = request.form.get('model')
        config_name = model_name
        cmd.append(config_name)  # Add the config name to the command

        # Helper to quote values for Hydra's command-line parser
        def hydra_quote(value):
            """Quotes a value for Hydra (single quotes, escapes internal)."""
            value_str = str(value)
            # Escape internal single quotes: ' -> '\''
            escaped_value = value_str.replace("'", r"\'")
            return f"'{escaped_value}'"

        # Set of keys known to be paths needing quoting for Hydra
        path_keys = {"audio_path", "output_path", "beatmap_path", "lora_path"}

        # Helper to add argument if value exists
        def add_arg(key, value):
            if value is not None and value != '':  # Ensure value is not empty
                if key in path_keys:
                    # Quote path values for Hydra
                    cmd.append(f"{key}={hydra_quote(value)}")
                else:
                    # Other values usually don't need explicit Hydra quoting when passed via list
                    cmd.append(f"{key}={value}")

        # Helper for list arguments (Hydra format: key=['item1','item2',...])
        def add_list_arg(key, items):
            if items:
                # Wrap each item in single quotes and join with comma
                quoted_items = [f"'{str(item)}'" for item in items]
                items_str = ",".join(quoted_items)
                cmd.append(f"{key}=[{items_str}]")

        # Required Paths
        add_arg("audio_path", request.form.get('audio_path'))
        add_arg("output_path", request.form.get('output_path'))
        # Beatmap path
        beatmap_path = request.form.get('beatmap_path')
        add_arg("beatmap_path", beatmap_path)

        # Optional LoRA path
        if 'lora_path' in request.form:
            add_arg("lora_path", request.form.get('lora_path'))

        # Basic settings
        if 'gamemode' in request.form:
            add_arg("gamemode", request.form.get('gamemode'))
        else:
            # Default to 0 if not provided
            add_arg("gamemode", 0)
        add_arg("difficulty", request.form.get('difficulty'))
        add_arg("year", request.form.get('year'))

        # Numeric settings
        for param in ['hp_drain_rate', 'circle_size', 'overall_difficulty', 'approach_rate', 'slider_multiplier',
                      'slider_tick_rate', 'keycount', 'hold_note_ratio', 'scroll_speed_ratio',
                      'cfg_scale', 'temperature', 'top_p', 'seed']:
            add_arg(param, request.form.get(param))
        # mapper_id
        add_arg("mapper_id", request.form.get('mapper_id'))

        # Timing and segmentation
        for param in ['start_time', 'end_time']:
            add_arg(param, request.form.get(param))

        # Checkboxes
        if 'export_osz' in request.form:
            cmd.append("export_osz=true")
        else :
            cmd.append("export_osz=false")
        if 'add_to_beatmap' in request.form:
            cmd.append("add_to_beatmap=true")
        else:
            cmd.append("add_to_beatmap=false")
        if 'overwrite_reference_beatmap' in request.form:
            cmd.append("overwrite_reference_beatmap=true")
        else:
            cmd.append("overwrite_reference_beatmap=false")
        if 'hitsounded' in request.form:
            cmd.append("hitsounded=true")
        else:
            cmd.append("hitsounded=false")
        if 'super_timing' in request.form:
            cmd.append("super_timing=true")
        else:
            cmd.append("super_timing=false")

        # Descriptors
        descriptors = request.form.getlist('descriptors')
        add_list_arg("descriptors", descriptors)

        # Negative Descriptors
        negative_descriptors = request.form.getlist('negative_descriptors')
        add_list_arg("negative_descriptors", negative_descriptors)

        # In-Context Options
        in_context_options = request.form.getlist('in_context_options')
        if in_context_options and beatmap_path:  # Only add if not empty
            add_list_arg("in_context", in_context_options)
        # --- End Command List Construction ---

        print("Executing Command List (shell=False):", cmd)

        try:
            # Start the inference process without shell=True
            current_process = subprocess.Popen(
                cmd,  # Pass the list directly
                shell=False,  # Explicitly False (default)
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stdout and stderr
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            print(f"Started process with PID: {current_process.pid}")
            # Return success to the AJAX call
            return jsonify({"status": "success", "message": "Inference started"}), 202  # Accepted

        except Exception as e:
            print(f"Error starting subprocess: {e}")
            current_process = None
            return jsonify({"status": "error", "message": f"Failed to start process: {e}"}), 500


@app.route('/stream_output')
def stream_output():
    """Streams the output of the running inference process using SSE."""

    def generate():
        global current_process, beatmapset_enabled, beatmapset_files, beatmapset_audio_path, beatmapset_output_dir
        process_to_stream = None

        # Short lock to safely get the process object
        with process_lock:
            if current_process and current_process.poll() is None:
                process_to_stream = current_process
                print(f"Attempting to stream output for PID: {process_to_stream.pid}")
            else:
                # Handle case where process is already finished or never started
                print("Stream requested but no active process found or process already finished.")
                yield "event: end\ndata: No active process or process already finished\n\n"
                return

        # If we got a process, proceed with streaming
        if process_to_stream:
            print(f"Streaming output for PID: {process_to_stream.pid}")
            full_output_lines = []
            error_occurred = False
            log_filepath = None

            try:
                # Stream lines from stdout
                for line in iter(process_to_stream.stdout.readline, ""):
                    full_output_lines.append(line)
                    yield f"data: {line.rstrip()}\n\n"
                    sys.stdout.flush()  # Ensure data is sent

                # --- Process finished, check status ---
                process_to_stream.stdout.close()  # Close the pipe
                return_code = process_to_stream.wait()  # Wait for process to terminate fully
                print(f"Process {process_to_stream.pid} finished streaming with exit code: {return_code}")

                if return_code != 0:
                    error_occurred = True
                    print(f"Non-zero exit code ({return_code}) detected for PID {process_to_stream.pid}. Marking as error.")
                else:
                    # --- Rename output file after successful generation ---
                    if QUEUE_FEATURES_AVAILABLE and last_form_data:
                        try:
                            output_path = last_form_data.get('output_path', '')
                            audio_path = last_form_data.get('audio_path', '')
                            mapper_name = last_form_data.get('mapper_name', '')
                            mapper_id = last_form_data.get('mapper_id', '')
                            diff_name = last_form_data.get('diff_name', '')
                            
                            if output_path and os.path.isfile(output_path):
                                new_path = rename_output(
                                    output_path=output_path,
                                    audio_path=audio_path,
                                    mapper_name=mapper_name,
                                    mapper_id=mapper_id,
                                    diff_name=diff_name
                                )
                                if new_path != output_path:
                                    print(f"Renamed output file to: {new_path}")
                                    yield f"event: renamed\ndata: {new_path.replace(os.sep, '/')}\n\n"
                                
                                # Apply preview time and background if set
                                final_path = new_path if new_path != output_path else output_path
                                preview_time = last_form_data.get('preview_time', '')
                                background_path = last_form_data.get('background_path', '')
                                
                                if preview_time or background_path:
                                    apply_beatmap_customizations(final_path, preview_time, background_path)
                                
                                # Add to beatmap set if enabled
                                if beatmapset_enabled:
                                    final_path = new_path if new_path != output_path else output_path
                                    beatmapset_files.append(final_path)
                                    if not beatmapset_audio_path and audio_path:
                                        beatmapset_audio_path = audio_path
                                    if not beatmapset_output_dir:
                                        beatmapset_output_dir = os.path.dirname(final_path)
                                    print(f"Added to beatmap set: {final_path} (total: {len(beatmapset_files)})")
                        except Exception as rename_e:
                            print(f"Warning: Could not rename output file: {rename_e}")

            except Exception as e:
                print(f"Error during streaming for PID {process_to_stream.pid}: {e}")
                error_occurred = True
                full_output_lines.append(f"\n--- STREAMING ERROR ---\n{e}\n")
            finally:
                # --- Log Saving Logic (if error occurred) ---
                if error_occurred:
                    try:
                        log_dir = os.path.join(script_dir, 'logs')
                        os.makedirs(log_dir, exist_ok=True)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_filename = f"error_{process_to_stream.pid}_{timestamp}.log"
                        log_filepath = os.path.join(log_dir, log_filename)
                        error_content = "".join(full_output_lines)

                        with open(log_filepath, 'w', encoding='utf-8') as f:
                            f.write(error_content)
                        print(f"Error log saved for PID {process_to_stream.pid} to: {log_filepath}")
                        yield f"event: error_log\ndata: {log_filepath.replace(os.sep, '/')}\n\n"

                    except Exception as log_e:
                        print(f"FATAL: Could not write error log for PID {process_to_stream.pid}: {log_e}")

                # --- Standard End Event ---
                completion_message = "Process completed"
                if error_occurred:
                    completion_message += " with errors"
                yield f"event: end\ndata: {completion_message}\n\n"
                print(f"Finished streaming for PID: {process_to_stream.pid}. Sent 'end' event.")

                # --- Cleanup global process reference ---
                with process_lock:
                    if current_process == process_to_stream:
                        current_process = None
                        print("Cleared global current_process reference.")
                    else:
                        print(f"Stale process {process_to_stream.pid} finished streaming, global reference was already updated/cleared.")

    return Response(generate(), mimetype='text/event-stream')


@app.route('/cancel_inference', methods=['POST'])
def cancel_inference():
    """Attempts to terminate the currently running inference process."""
    global current_process, queue_cancelled
    message = ""
    success = False
    status_code = 500
    
    # Check if we should also cancel the queue
    clear_queue = request.json.get('clear_queue', False) if request.is_json else False
    if clear_queue:
        queue_cancelled = True

    with process_lock:
        if current_process and current_process.poll() is None:
            try:
                pid = current_process.pid
                print(f"Attempting to terminate process PID: {pid}...")
                
                # On Windows, use taskkill to kill the process tree
                if sys.platform == 'win32':
                    try:
                        import subprocess as sp
                        sp.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                               capture_output=True, check=False)
                        print(f"Process tree PID: {pid} killed with taskkill.")
                        message = "Cancel request sent, process tree terminated."
                    except Exception as taskkill_error:
                        print(f"taskkill failed: {taskkill_error}, falling back to terminate()")
                        current_process.terminate()
                        message = "Cancel request sent. Process termination might take a moment."
                else:
                    # On Unix, use process group kill
                    try:
                        import os as os_mod
                        os_mod.killpg(os_mod.getpgid(pid), signal.SIGTERM)
                        print(f"Process group PID: {pid} terminated with SIGTERM.")
                        message = "Cancel request sent, process group terminated."
                    except Exception:
                        current_process.terminate()
                        message = "Cancel request sent. Process termination might take a moment."
                
                # Wait briefly to see if it terminates
                try:
                    current_process.wait(timeout=1)
                    print(f"Process PID: {pid} terminated successfully after request.")
                except subprocess.TimeoutExpired:
                    print(f"Process PID: {pid} did not terminate immediately.")

                success = True
                status_code = 200
                # DO NOT set current_process = None here. Let the stream generator handle it.
            except Exception as e:
                print(f"Error terminating process: {e}")
                message = f"Error occurred during cancellation: {e}"
                success = False
                status_code = 500
        elif current_process:
            message = "Process already finished."
            success = True if clear_queue else False
            status_code = 200 if clear_queue else 409
        else:
            message = "No process is currently running."
            if clear_queue:
                message = "Queue cancelled. No process was running."
                success = True
                status_code = 200
            else:
                success = False
                status_code = 404

    response_data = {"status": "success" if success else "error", "message": message}
    if clear_queue:
        response_data["queue_cleared"] = True
    return jsonify(response_data), status_code


@app.route('/queue_status', methods=['GET', 'POST'])
def queue_status():
    """Get or set the queue cancellation status."""
    global queue_cancelled
    
    if request.method == 'POST':
        data = request.json or {}
        if 'cancelled' in data:
            queue_cancelled = data['cancelled']
            return jsonify({"status": "success", "cancelled": queue_cancelled})
        return jsonify({"status": "error", "message": "Missing 'cancelled' field"}), 400
    
    # GET - return current status
    return jsonify({
        "cancelled": queue_cancelled,
        "process_running": current_process is not None and current_process.poll() is None
    })


@app.route('/reset_queue', methods=['POST'])
def reset_queue():
    """Reset the queue cancellation flag (called when starting a new queue)."""
    global queue_cancelled
    queue_cancelled = False
    return jsonify({"status": "success", "message": "Queue reset"})


@app.route('/get_audio_info', methods=['POST'])
def get_audio_info():
    """Get audio file info and return a URL for playback."""
    data = request.get_json() or {}
    audio_path = data.get('path', '')
    
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"success": False, "message": "Audio file not found"})
    
    try:
        # Return a URL that serves the audio file
        # We'll create a route to serve audio files
        return jsonify({
            "success": True,
            "url": f"/serve_audio?path={audio_path}",
            "duration": None  # Could add duration detection if needed
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@app.route('/serve_audio', methods=['GET'])
def serve_audio():
    """Serve an audio file for preview playback with Range request support for seeking."""
    audio_path = request.args.get('path', '')
    
    if not audio_path or not os.path.exists(audio_path):
        return "Audio file not found", 404
    
    # Determine content type
    ext = os.path.splitext(audio_path)[1].lower()
    content_types = {
        '.mp3': 'audio/mpeg',
        '.ogg': 'audio/ogg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.m4a': 'audio/mp4'
    }
    content_type = content_types.get(ext, 'audio/mpeg')
    
    # Get file size
    file_size = os.path.getsize(audio_path)
    
    # Check for Range header (needed for seeking)
    range_header = request.headers.get('Range', None)
    
    if range_header:
        # Parse Range header (e.g., "bytes=0-" or "bytes=1000-2000")
        byte_start = 0
        byte_end = file_size - 1
        
        range_match = range_header.replace('bytes=', '').split('-')
        if range_match[0]:
            byte_start = int(range_match[0])
        if len(range_match) > 1 and range_match[1]:
            byte_end = int(range_match[1])
        
        # Ensure valid range
        byte_end = min(byte_end, file_size - 1)
        content_length = byte_end - byte_start + 1
        
        def generate_range():
            with open(audio_path, 'rb') as f:
                f.seek(byte_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(8192, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        # Return 206 Partial Content
        response = Response(
            generate_range(),
            status=206,
            mimetype=content_type,
            direct_passthrough=True
        )
        response.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = content_length
        return response
    else:
        # No Range header - return full file
        def generate():
            with open(audio_path, 'rb') as f:
                while chunk := f.read(8192):
                    yield chunk
        
        response = Response(generate(), mimetype=content_type)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Content-Length'] = file_size
        return response


@app.route('/get_image_preview', methods=['POST'])
def get_image_preview():
    """Get base64 encoded image preview for background."""
    import base64
    
    data = request.get_json() or {}
    image_path = data.get('path', '')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({"success": False, "message": "Image file not found"})
    
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            return jsonify({"success": False, "message": "Unsupported image format"})
        
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        type_map = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png', '.gif': 'gif', '.bmp': 'bmp'}
        image_type = type_map.get(ext, 'jpeg')
        
        return jsonify({
            "success": True,
            "data": image_data,
            "type": image_type
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# Beatmap set compilation state
beatmapset_enabled = False
beatmapset_files = []
beatmapset_audio_path = None
beatmapset_output_dir = None


@app.route('/init_beatmapset', methods=['POST'])
def init_beatmapset():
    """Initialize beatmap set compilation for a queue run."""
    global beatmapset_enabled, beatmapset_files, beatmapset_audio_path, beatmapset_output_dir
    data = request.get_json() or {}
    beatmapset_enabled = data.get('enabled', False)
    beatmapset_files = []
    beatmapset_audio_path = None
    beatmapset_output_dir = None
    return jsonify({"status": "success", "enabled": beatmapset_enabled})


@app.route('/add_to_beatmapset', methods=['POST'])
def add_to_beatmapset():
    """Add a generated .osu file to the beatmap set."""
    global beatmapset_files, beatmapset_audio_path, beatmapset_output_dir
    if not beatmapset_enabled:
        return jsonify({"status": "skipped", "message": "Beatmap set compilation not enabled"})
    
    data = request.get_json() or {}
    osu_file = data.get('osu_file')
    audio_path = data.get('audio_path')
    output_dir = data.get('output_dir')
    
    if osu_file and os.path.exists(osu_file):
        beatmapset_files.append(osu_file)
        if not beatmapset_audio_path and audio_path:
            beatmapset_audio_path = audio_path
        if not beatmapset_output_dir and output_dir:
            beatmapset_output_dir = output_dir
    
    return jsonify({"status": "success", "files_count": len(beatmapset_files)})


@app.route('/finalize_beatmapset', methods=['POST'])
def finalize_beatmapset():
    """Compile all collected .osu files into a single .osz beatmap set."""
    global beatmapset_enabled, beatmapset_files, beatmapset_audio_path, beatmapset_output_dir
    
    if not beatmapset_enabled or len(beatmapset_files) == 0:
        return jsonify({"success": False, "message": "No files to compile"})
    
    try:
        import zipfile
        import shutil
        
        # Determine output directory
        output_dir = beatmapset_output_dir or os.path.dirname(beatmapset_files[0])
        
        # Get metadata from first .osu file to name the osz
        first_osu = beatmapset_files[0]
        artist = "Unknown"
        title = "Unknown"
        
        try:
            with open(first_osu, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('Artist:'):
                        artist = line.split(':', 1)[1].strip()
                    elif line.startswith('Title:'):
                        title = line.split(':', 1)[1].strip()
                    if artist != "Unknown" and title != "Unknown":
                        break
        except Exception as e:
            print(f"Error reading metadata from .osu file: {e}")
        
        # Create osz filename
        safe_artist = "".join(c for c in artist if c.isalnum() or c in " -_").strip()
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()
        osz_name = f"{safe_artist} - {safe_title}.osz"
        osz_path = os.path.join(output_dir, osz_name)
        
        # Create the .osz (which is just a .zip)
        with zipfile.ZipFile(osz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all .osu files
            for osu_file in beatmapset_files:
                zf.write(osu_file, os.path.basename(osu_file))
            
            # Add audio file if exists
            if beatmapset_audio_path and os.path.exists(beatmapset_audio_path):
                zf.write(beatmapset_audio_path, os.path.basename(beatmapset_audio_path))
        
        # Reset state
        beatmapset_enabled = False
        beatmapset_files = []
        beatmapset_audio_path = None
        beatmapset_output_dir = None
        
        return jsonify({"success": True, "filename": osz_name, "path": osz_path})
        
    except Exception as e:
        print(f"Error creating beatmap set: {e}")
        return jsonify({"success": False, "message": str(e)})


@app.route('/open_folder', methods=['GET'])
def open_folder():
    """Opens a folder in the file explorer."""
    folder_path = request.args.get('folder')
    print(f"Request received to open folder: {folder_path}")
    if not folder_path:
        return jsonify({"status": "error", "message": "No folder path specified"}), 400

    # Resolve to absolute path for checks
    abs_folder_path = os.path.abspath(folder_path)

    # Security check: Basic check if it's within the project directory.
    # Adjust this check based on your security needs and where output is expected.
    workspace_root = os.path.abspath(script_dir)
    # Example: Only allow opening if it's inside the workspace root
    # if not abs_folder_path.startswith(workspace_root):
    #     print(f"Security Warning: Attempt to open potentially restricted folder: {abs_folder_path}")
    #     return jsonify({"status": "error", "message": "Access denied to specified folder path."}), 403

    if not os.path.isdir(abs_folder_path):
        print(f"Invalid folder path provided or folder does not exist: {abs_folder_path}")
        return jsonify({"status": "error", "message": "Invalid or non-existent folder path specified"}), 400

    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(os.path.normpath(abs_folder_path))
        elif system == 'Darwin':
            subprocess.Popen(['open', abs_folder_path])
        else:
            subprocess.Popen(['xdg-open', abs_folder_path])
        print(f"Successfully requested to open folder: {abs_folder_path}")
        return jsonify({"status": "success", "message": "Folder open request sent."}), 200
    except Exception as e:
        print(f"Error opening folder '{abs_folder_path}': {e}")
        return jsonify({"status": "error", "message": f"Could not open folder: {e}"}), 500


@app.route('/open_log_file', methods=['GET'])
def open_log_file():
    """Opens a specific log file."""
    log_path = request.args.get('path')
    print(f"Request received to open log file: {log_path}")
    if not log_path:
        return jsonify({"status": "error", "message": "No log file path specified"}), 400

    # Security Check: Ensure the file is within the 'logs' directory
    log_dir = os.path.abspath(os.path.join(script_dir, 'logs'))
    # Normalize the input path and resolve symlinks etc.
    abs_log_path = os.path.abspath(os.path.normpath(log_path))

    # IMPORTANT SECURITY CHECK:
    if not abs_log_path.startswith(log_dir + os.sep):
        print(f"Security Alert: Attempt to open file outside of logs directory: {abs_log_path} (Log dir: {log_dir})")
        return jsonify({"status": "error", "message": "Access denied: File is outside the designated logs directory."}), 403

    if not os.path.isfile(abs_log_path):
        print(f"Log file not found at: {abs_log_path}")
        return jsonify({"status": "error", "message": "Log file not found."}), 404

    try:
        system = platform.system()
        if system == 'Windows':
            os.startfile(abs_log_path) # normpath already applied
        elif system == 'Darwin':
            subprocess.Popen(['open', abs_log_path])
        else:
            subprocess.Popen(['xdg-open', abs_log_path])
        print(f"Successfully requested to open log file: {abs_log_path}")
        return jsonify({"status": "success", "message": "Log file open request sent."}), 200
    except Exception as e:
        print(f"Error opening log file '{abs_log_path}': {e}")
        return jsonify({"status": "error", "message": f"Could not open log file: {e}"}), 500


@app.route('/save_config', methods=['POST'])
def save_config():
    try:
        file_path = request.form.get('file_path')
        config_data = request.form.get('config_data')

        if not file_path or not config_data:
            return jsonify({'success': False, 'error': 'Missing required parameters'})

        # Write the configuration file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(config_data)

        return jsonify({
            'success': True,
            'file_path': file_path,
            'message': 'Configuration saved successfully'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to save configuration: {str(e)}'
        })


# ═══════════════════════════════════════════════════════════════════════
#  QUEUE SYSTEM ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.route("/lookup_mapper_name", methods=["POST"])
def api_lookup_mapper():
    """Look up mapper username from osu! API."""
    data = request.get_json() or {}
    mapper_id = data.get("mapper_id")
    print(f"[lookup_mapper] Received request for mapper_id: {mapper_id}")
    
    if not mapper_id:
        return jsonify({"error": "mapper_id required"}), 400
    
    if not QUEUE_FEATURES_AVAILABLE:
        print("[lookup_mapper] Queue features not available")
        return jsonify({"error": "Queue features not available"}), 503
    
    try:
        name = lookup_username(mapper_id)
        print(f"[lookup_mapper] Result for {mapper_id}: {name}")
        if not name:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"username": name})
    except Exception as e:
        print(f"Mapper lookup error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/validate_paths', methods=['POST'])
def validate_paths():
    """Validates paths, autofills folders, and attempts song detection."""
    try:
        # Get paths
        audio_path = request.form.get('audio_path', '').strip()
        beatmap_path = request.form.get('beatmap_path', '').strip()
        output_path = request.form.get('output_path', '').strip()

        inference_args = InferenceConfig()
        inference_args.audio_path = audio_path
        inference_args.beatmap_path = beatmap_path
        inference_args.output_path = output_path

        result = autofill_paths(inference_args)

        # Attempt song detection only if explicitly requested (with caching)
        detected_artist = None
        detected_title = None
        detect_song = request.form.get('detect_song', 'false').lower() == 'true'
        
        if detect_song and QUEUE_FEATURES_AVAILABLE and inference_args.audio_path:
            actual_audio = inference_args.audio_path
            if os.path.isfile(actual_audio):
                # Check cache first
                if actual_audio in song_detection_cache:
                    detected_artist, detected_title = song_detection_cache[actual_audio]
                else:
                    try:
                        detected_artist, detected_title = identify_song(actual_audio)
                        # Cache the result (even if None)
                        song_detection_cache[actual_audio] = (detected_artist, detected_title)
                    except Exception as e:
                        print(f"Song detection error: {e}")
                        song_detection_cache[actual_audio] = (None, None)

        # Build response
        response_data = {
            'success': result['success'],
            'autofilled_audio_path': inference_args.audio_path,
            'autofilled_output_path': inference_args.output_path,
            'errors': result['errors']
        }
        
        if detected_artist:
            response_data['detected_artist'] = detected_artist
        if detected_title:
            response_data['detected_title'] = detected_title

        return jsonify(response_data), 200

    except Exception as e:
        error_msg = f"Error during path validation: {str(e)}"
        print(f"Path validation error: {error_msg}")
        return jsonify({
            'success': False,
            'errors': [error_msg],
            'autofilled_audio_path': None,
            'autofilled_output_path': None
        }), 500


# --- Function to Run Flask in a Thread ---
def run_flask(port):
    """Runs the Flask app."""

    # Use threaded=True for better concurrency within Flask
    # Avoid debug=True as it interferes with threading and pywebview
    print(f"Starting Flask server on http://127.0.0.1:{port}")
    try:
        # Explicitly set debug=False, in addition to FLASK_ENV=production
        app.run(host='127.0.0.1', port=port, threaded=True, debug=False)
    except OSError as e:
        print(f"Flask server could not start on port {port}: {e}")
        # Optionally: try another port or exit


# --- Function to Find Available Port ---
def find_available_port(start_port=5000, max_tries=100):
    """Finds an available TCP port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                print(f"Found available port: {port}")
                return port
            except OSError:
                continue  # Port already in use
    raise IOError("Could not find an available port.")


# --- Main Execution ---
if __name__ == '__main__':
    # Find an available port for Flask
    flask_port = find_available_port()

    # Start Flask server in a daemon thread
    flask_thread = threading.Thread(target=run_flask, args=(flask_port,), daemon=True)
    flask_thread.start()

    # Give Flask a moment to start up
    time.sleep(1)

    # --- Calculate Responsive Window Size ---
    try:
        primary_screen = webview.screens[0]
        screen_width = primary_screen.width
        screen_height = primary_screen.height
        # Calculate window size (e.g., 45% width, 95% height of primary screen)
        window_width = int(screen_width * 0.45)
        window_height = int(screen_height * 0.95)
        print(f"Screen: {screen_width}x{screen_height}, Window: {window_width}x{window_height}")
    except Exception as e:
        print(f"Could not get screen dimensions, using default: {e}")
        # Fallback to default size if screen info is unavailable
        window_width = 900
        window_height = 1000
    # --- End Calculate Responsive Window Size ---

    # Create the pywebview window pointing to the Flask server
    window_title = 'Mapperatorinator'
    flask_url = f'http://127.0.0.1:{flask_port}/'

    print(f"Creating pywebview window loading URL: {flask_url}")

    # Instantiate the API class (doesn't need window object anymore)
    api = Api()

    # Pass api instance directly to create_window via js_api
    window = webview.create_window(
        window_title,
        url=flask_url,
        width=window_width,  # Use calculated width
        height=window_height,  # Use calculated height
        resizable=True,
        js_api=api  # Expose Python API class here
    )

    # Start the pywebview event loop
    webview.start()

    print("Pywebview window closed. Exiting application.")
    # Flask thread will exit automatically as it's a daemon
