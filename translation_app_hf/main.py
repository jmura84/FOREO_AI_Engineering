import os
import sys
from fastapi import FastAPI
import gradio as gr
from contextlib import asynccontextmanager

# Add the current directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.interface import create_gradio_interface, THEME, CUSTOM_CSS

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic if needed
    print("Starting Translation App HF...")
    yield
    # Shutdown logic if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Create Gradio interface
io = create_gradio_interface()

# Mount Gradio app
# path="/" means the Gradio app will be at the root URL
app = gr.mount_gradio_app(app, io, path="/", theme=THEME, css=CUSTOM_CSS)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
