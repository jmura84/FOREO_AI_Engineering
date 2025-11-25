import gradio as gr
import inspect

print(f"Gradio version: {gr.__version__}")
try:
    sig = inspect.signature(gr.mount_gradio_app)
    for name, param in sig.parameters.items():
        print(f"{name}: {param.kind}")
except Exception as e:
    print(f"Error getting signature: {e}")
