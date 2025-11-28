import os
import sys
import gradio as gr

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the corrected interface (no THEME)
from ui.interface import create_gradio_interface, CUSTOM_CSS

# -----------------------------
# Manual lifecycle hooks (simulate FastAPI lifespan)
# -----------------------------
def on_startup():
    print("Starting Translation App HF...")

def on_shutdown():
    print("Shutting down...")

# -----------------------------
# Create Gradio interface
# -----------------------------
io = create_gradio_interface()

# -----------------------------
# Run locally (Gradio)
# -----------------------------
if __name__ == "__main__":
    try:
        on_startup()

        # Local run
        io.launch(
            server_name="127.0.0.1",  # Localhost
            server_port=8000,
            css=CUSTOM_CSS
        )

        # For HuggingFace Spaces:
        # io.launch(server_name="0.0.0.0", server_port=7860, css=CUSTOM_CSS)

    except KeyboardInterrupt:
        print("Server stopped by user (KeyboardInterrupt). Exiting gracefully.")
        sys.exit(0)

    finally:
        on_shutdown()
