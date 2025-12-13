import sys

# Store the original excepthook
original_excepthook = sys.excepthook


def custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception hook to modify ModuleNotFoundError messages.
    """
    if issubclass(exc_type, ModuleNotFoundError):
        # Still print the original traceback for debugging purposes
        original_excepthook(exc_type, exc_value, exc_traceback)

        missing_module = str(exc_value).split("'")[-2]  # Extract the module name
        print(f"\nError: The module '{missing_module}' was not found.", file=sys.stderr)
        print("To fix this, please ensure all required packages are installed by running:", file=sys.stderr)
        print("`pip install -r requirements.txt`", file=sys.stderr)
    else:
        # For other exceptions, call the original excepthook
        original_excepthook(exc_type, exc_value, exc_traceback)

# Set the custom excepthook
sys.excepthook = custom_excepthook
