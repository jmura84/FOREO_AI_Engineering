import os
import sys

# Add the root directory to the path (good practice, though not always necessary if run from root)
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    # Import the app creation function from our UI module
    from ui.gradio_ui import create_app
except ImportError as e:
    print(f"Error: Could not import the application.")
    print(f"Make sure __init__.py files are in place and the structure is correct.")
    print(f"Detailed error: {e}")
    sys.exit(1)


def main():
    """
    Main entry point for launching the application.
    """

    # Ensure the 'data' directory exists before starting
    data_dir = os.path.join(root_dir, 'data')
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
            print(f"'data' directory created at: {data_dir}")
        except Exception as e:
            print(f"Error creating 'data' directory: {e}")
            sys.exit(1)

    print("Starting Gradio application...")

    # Create the app instance
    app = create_app()

    # Launch the app
    # Launch the app - explicit binding for container/GKE
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()