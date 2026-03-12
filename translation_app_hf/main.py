import os
import sys
from fastapi import FastAPI
import gradio as gr
from contextlib import asynccontextmanager

# -----------------------------
# Add current directory to Python path
# -----------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Import the corrected interface (no THEME)
# -----------------------------
from ui.interface import create_gradio_interface, CUSTOM_CSS

# -----------------------------
# FastAPI lifespan hooks
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Translation App HF...")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# -----------------------------
# Create Gradio interface
# -----------------------------
io = create_gradio_interface()

# -----------------------------
# Mount Gradio app into FastAPI
# -----------------------------
# Note: theme removed because HF Spaces may use Gradio < 3.36
app = gr.mount_gradio_app(app, io, path="/", css=CUSTOM_CSS)

# -----------------------------
# Run locally with uvicorn
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    try:
        # HF Spaces: run with host="0.0.0.0", port=7860
        uvicorn.run(app, host="0.0.0.0", port=7860)

        # Local run with custom IP (uncomment to use)
        # uvicorn.run(app, host="127.0.0.1", port=8000)

    except KeyboardInterrupt:
        print("Server stopped by user (KeyboardInterrupt). Exiting gracefully.")
        sys.exit(0)
