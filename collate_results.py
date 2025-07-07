import os
import re


def get_color_for_value(value, min_val, max_val, lower_is_better=False):
    """
    Generates an HSL color string from red to green based on a value's
    position between a min and max.

    Args:
        value (float): The current value.
        min_val (float): The minimum value in the dataset for this metric.
        max_val (float): The maximum value in the dataset for this metric.
        lower_is_better (bool): If True, lower values get greener colors.

    Returns:
        str: An HSL color string for use in CSS.
    """
    # Avoid division by zero if all values are the same
    if min_val == max_val:
        return "hsl(120, 70%, 60%)"  # Default to green

    # Normalize the value to a 0-1 range
    normalized = (value - min_val) / (max_val - min_val)

    if lower_is_better:
        # Invert the scale: 1 (best) -> 0 (worst)
        hue = (1 - normalized) * 120
    else:
        # Standard scale: 0 (worst) -> 1 (best)
        hue = normalized * 120

    # Return HSL color: hue from 0 (red) to 120 (green), with fixed saturation and lightness
    return f"hsl({hue:.0f}, 70%, 60%)"


def parse_log_files(root_dir):
    """
    Parses log files in subdirectories to extract metrics and format them
    into an HTML table with colored cells.

    Args:
        root_dir (str): The path to the main folder containing the model subfolders.

    Returns:
        str: A string containing the formatted HTML table.
    """
    results = []
    dir_pattern = re.compile(r"inference=(inference_)?([a-zA-Z0-9_-]+)")
    metric_patterns = {
        'FID': re.compile(r"FID: ([\d.]+)"),
        'AR Pr.': re.compile(r"Active Rhythm Precision: ([\d.]+)"),
        'AR Re.': re.compile(r"Active Rhythm Recall: ([\d.]+)"),
        'AR F1': re.compile(r"Active Rhythm F1: ([\d.]+)"),
        'PR Pr.': re.compile(r"Passive Rhythm Precision: ([\d.]+)"),
        'PR Re.': re.compile(r"Passive Rhythm Recall: ([\d.]+)"),
        'PR F1': re.compile(r"Passive Rhythm F1: ([\d.]+)")
    }

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            for dirname in dirnames:
                dir_match = dir_pattern.match(dirname)
                if not dir_match:
                    continue
                model_name = dir_match.group(2)
                log_file_path = os.path.join(dirpath, dirname, 'calc_fid.log')

                if not os.path.exists(log_file_path):
                    print(f"Warning: 'calc_fid.log' not found in {dirname}")
                    continue

                latest_metrics = {}
                try:
                    with open(log_file_path, 'r') as f:
                        for line in f:
                            for key, pattern in metric_patterns.items():
                                match = pattern.search(line)
                                if match:
                                    latest_metrics[key] = float(match.group(1))
                except Exception as e:
                    print(f"Error reading {log_file_path}: {e}")
                    continue

                if latest_metrics:
                    latest_metrics['Model name'] = model_name
                    results.append(latest_metrics)
            dirnames[:] = []

    if not results:
        return "<p>No results found. Check if <code>root_dir</code> is correct and log files exist.</p>"

    # --- Pre-calculate Min/Max for coloring ---
    headers = ["Model name", "FID", "AR Pr.", "AR Re.", "AR F1", "PR Pr.", "PR Re.", "PR F1"]
    min_max_vals = {}
    for header in headers:
        if header == "Model name":
            continue
        # Get all valid values for the current header
        values = [res.get(header) for res in results if res.get(header) is not None]
        if values:
            min_max_vals[header] = {'min': min(values), 'max': max(values)}

    # --- Generate HTML Table ---
    html = ["<table>"]
    # Header row
    html.append("  <thead>")
    html.append("    <tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>")
    html.append("  </thead>")

    # Data rows
    html.append("  <tbody>")
    for res in sorted(results, key=lambda x: x.get('Model name', '')):
        row_html = "    <tr>"
        for header in headers:
            value = res.get(header)

            if header == 'Model name':
                row_html += f"<td>{res.get('Model name', 'N/A')}</td>"
                continue

            if value is None:
                row_html += "<td>N/A</td>"
                continue

            # Formatting
            if header == 'FID':
                formatted_value = f"{value:.2f}"
                lower_is_better = True
            else:
                formatted_value = f"{value:.3f}"
                lower_is_better = False

            # Get color and apply style
            color = get_color_for_value(value, min_max_vals[header]['min'], min_max_vals[header]['max'],
                                        lower_is_better)
            # Added a light text shadow for better readability on bright colors
            style = f"background-color: {color}; color: black; text-shadow: 0 0 5px white;"
            row_html += f'<td style="{style}">{formatted_value}</td>'

        row_html += "</tr>"
        html.append(row_html)

    html.append("  </tbody>")
    html.append("</table>")

    return "\n".join(html)


if __name__ == '__main__':
    # --- IMPORTANT ---
    # Change this to the path of your main results folder.
    # You can use "." if the script is in the same parent folder as the "inference=..." folders.
    logs_directory = './logs_fid/sweeps/test_1'

    markdown_table = parse_log_files(logs_directory)
    print(markdown_table)

