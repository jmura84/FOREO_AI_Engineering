import gradio as gr

def check_upload(file_obj):
    print(f"DEBUG: file_obj type: {type(file_obj)}")
    print(f"DEBUG: file_obj: {file_obj}")
    if file_obj is None:
        return "No file received"
    if hasattr(file_obj, 'name'):
        return f"File name: {file_obj.name}"
    return f"File object: {file_obj}"

with gr.Blocks() as demo:
    btn = gr.UploadButton("Upload", file_types=["audio"])
    out = gr.Textbox()
    btn.upload(check_upload, inputs=btn, outputs=out)

if __name__ == "__main__":
    demo.launch(show_error=True)
