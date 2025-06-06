import functools
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import datetime
from typing import Callable, Any, Tuple, Dict

import webview
import werkzeug.serving
from flask import Flask, render_template, request, Response, jsonify

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

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
app.secret_key = os.urandom(24)  # Set a secret key for Flask


# --- pywebview API Class ---
class Api:
    # No __init__ needed as we get the window dynamically

    def browse_file(self):
        """Opens a file dialog and returns the selected file path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(webview.OPEN_DIALOG)
        print(f"File dialog result: {result}")  # Debugging
        # pywebview returns a tuple, even for single file selection
        return result[0] if result else None

    def browse_folder(self):
        """Opens a folder dialog and returns the selected folder path."""
        # Get the window dynamically from the global list
        if not webview.windows:
            print("Error: No pywebview window found.")
            return None
        current_window = webview.windows[0]
        result = current_window.create_file_dialog(webview.FOLDER_DIALOG)
        print(f"Folder dialog result: {result}")  # Debugging
        # FOLDER_DIALOG also returns a tuple containing the path
        return result[0] if result else None


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
    global current_process
    with process_lock:
        if current_process and current_process.poll() is None:
            return jsonify({"status": "error", "message": "Process already running"}), 409  # Conflict

        # --- Construct Command List (shell=False) ---
        python_executable = sys.executable  # Get path to current Python interpreter
        cmd = [python_executable, "inference.py", "-cn"]

        # Get the model name from the form
        model_name = request.form.get('model')
        config_name = "inference_" + model_name
        cmd.append(config_name)  # Add the config name to the command

        # Helper to quote values for Hydra's command-line parser
        def hydra_quote(value):
            """Quotes a value for Hydra (single quotes, escapes internal)."""
            value_str = str(value)
            # Escape internal single quotes: ' -> '\''
            escaped_value = value_str.replace("'", r"\'")
            return f"'{escaped_value}'"

        # Set of keys known to be paths needing quoting for Hydra
        path_keys = {"audio_path", "output_path", "beatmap_path"}

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
        global current_process
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
    global current_process

    with process_lock:
        if current_process and current_process.poll() is None:
            try:
                pid = current_process.pid
                print(f"Attempting to terminate process PID: {pid}...")
                current_process.terminate()  # Send SIGTERM
                # Optional: Add a short wait to see if it terminates quickly
                try:
                    current_process.wait(timeout=1)
                    print(f"Process PID: {pid} terminated successfully after request.")
                    message = "Cancel request sent, process terminated."
                except subprocess.TimeoutExpired:
                    print(f"Process PID: {pid} did not terminate immediately after SIGTERM.")
                    message = "Cancel request sent. Process termination might take a moment."
                    # You could consider current_process.kill() here if terminate isn't enough

                success = True
                # DO NOT set current_process = None here. Let the stream generator handle it.
            except Exception as e:
                print(f"Error terminating process: {e}")
                message = f"Error occurred during cancellation: {e}"
                success = False
        elif current_process:
            message = "Process already finished."
            success = False  # Or True if you consider it 'cancelled' as it's done
        else:
            message = "No process is currently running."
            success = False

    if success:
        return jsonify({"status": "success", "message": message}), 200
    else:
        # Use 409 Conflict if already finished, 404 if never started, 500 for error
        status_code = 500 if "Error occurred" in message else (409 if "already finished" in message else 404)
        return jsonify({"status": "error", "message": message}), status_code


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
        folder_path = request.form.get('folder_path')
        filename = request.form.get('filename')
        config_data = request.form.get('config_data')

        if not folder_path or not filename or not config_data:
            return jsonify({'success': False, 'error': 'Missing required parameters'})

        # Ensure the folder path exists
        if not os.path.exists(folder_path):
            return jsonify({'success': False, 'error': 'Selected folder does not exist'})

        # Create full file path
        file_path = os.path.join(folder_path, filename)

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

    # Start the pywebview event loop (no args needed here now)
    webview.start()

    print("Pywebview window closed. Exiting application.")
    # Flask thread will exit automatically as it's a daemon
