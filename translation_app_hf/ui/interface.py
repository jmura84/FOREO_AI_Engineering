import gradio as gr
import os
import pandas as pd
from datetime import datetime
from core.llm_engine import get_llm_engine
from core.vision_engine import get_vision_engine
from core.audio_engine import get_audio_engine
from core.corrector import review_and_correct

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CSV_FILE_PATH = os.path.join(DATA_DIR, 'user_mods_tm.csv')

LANGUAGES = ["English", "Spanish", "Polish", "Turkish"]

import time
import threading

def run_with_timer(func, args, update_interval=0.1):
    result_container = {}
    def target():
        try:
            result_container['result'] = func(*args)
        except Exception as e:
            result_container['error'] = e

    t = threading.Thread(target=target)
    t.start()
    
    start_time = time.time()
    while t.is_alive():
        elapsed = time.time() - start_time
        yield elapsed
        t.join(timeout=update_interval)
    
    if 'error' in result_container:
        raise result_container['error']
    return result_container['result']

def translate_wrapper(text, model_name, temperature, source_lang, target_lang):
    if not text:
        yield "", "", gr.Button(visible=False), gr.Label(visible=False)
        return

    print(f"Translating with {model_name}...")
    
    def _translate_task():
        # Get engine and translate
        engine = get_llm_engine(model_name)
        raw_translation = engine.translate(text, source_lang, target_lang, temperature)

        # Apply corrections
        final_translation = review_and_correct(
            raw_translation, text, CSV_FILE_PATH, source_lang, target_lang
        )
        return final_translation

    try:
        for elapsed in run_with_timer(_translate_task, ()):
            if isinstance(elapsed, float):
                yield "", "", gr.Button(visible=False), gr.Label(value=f"Translating... {elapsed:.1f}s", visible=True)
            else:
                # This part is technically unreachable with current run_with_timer logic but good for safety
                pass
        
        # When generator finishes, run_with_timer returns the result, but we need to catch it
        # Actually run_with_timer is a generator that yields elapsed time, and returns result at the end?
        # No, generators in Python 3.3+ can return a value, but iterating over them doesn't give the return value easily.
        # Let's adjust run_with_timer to yield the result as the last item wrapped in a tuple or similar.
        pass
    except Exception as e:
        yield "", "", gr.Button(visible=False), gr.Label(value=f"Error: {e}", visible=True)
        return

    # Let's redefine run_with_timer slightly to be easier to use in a loop
    # Or just inline the logic here for simplicity and robustness
    
    result_container = {}
    def target():
        try:
            result_container['result'] = _translate_task()
        except Exception as e:
            result_container['error'] = e

    t = threading.Thread(target=target)
    t.start()
    
    start_time = time.time()
    while t.is_alive():
        elapsed = time.time() - start_time
        yield "", "", gr.Button(visible=False), gr.Label(value=f"Translating... {elapsed:.1f}s", visible=True)
        t.join(timeout=0.1)
    
    if 'error' in result_container:
        yield "", "", gr.Button(visible=False), gr.Label(value=f"Error: {result_container['error']}", visible=True)
    else:
        final_translation = result_container['result']
        elapsed = time.time() - start_time
        yield final_translation, final_translation, gr.Button(visible=False), gr.Label(value=f"Translation complete in {elapsed:.1f}s", visible=True)

def save_modification(source_text, modified_target_text, original_translation, source_lang, target_lang):
    language_pair = f"{source_lang} -> {target_lang}"
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    source_lines = (source_text or "").split('\n')
    original_target_lines = (original_translation or "").split('\n')
    modified_target_lines = (modified_target_text or "").split('\n')

    new_data = []
    for i in range(len(modified_target_lines)):
        line_changed = False
        if i >= len(original_target_lines):
            line_changed = True
        elif modified_target_lines[i] != original_target_lines[i]:
            line_changed = True

        if line_changed and i < len(source_lines) and source_lines[i].strip() != "":
            new_data.append({
                'source': source_lines[i],
                'target': modified_target_lines[i],
                'language_pairs': language_pair,
                'datetime': current_time
            })

    if not new_data:
        return gr.Label(value="No new changes detected.", visible=True), gr.Button(visible=False)

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        df_new = pd.DataFrame(new_data)
        
        if os.path.exists(CSV_FILE_PATH):
            df_old = pd.read_csv(CSV_FILE_PATH)
            df_combined = pd.concat([df_old, df_new])
        else:
            df_combined = df_new

        df_combined['source_normalized'] = df_combined['source'].astype(str).str.lower().str.strip().str.rstrip('.,!;')
        df_combined.drop_duplicates(subset=['source_normalized', 'language_pairs'], keep='last', inplace=True)
        df_combined.drop(columns=['source_normalized'], inplace=True)
        df_combined.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')

        return gr.Label(value="Modification(s) saved successfully!", visible=True), gr.Button(visible=False)
    except Exception as e:
        return gr.Label(value=f"Error: {e}", visible=True), gr.Button(visible=True)

def transcribe_audio(file_path, model_name):
    if file_path is None:
        yield gr.update(), gr.Label(value="No file uploaded.", visible=True)
        return
    
    result_container = {}
    def target():
        try:
            engine = get_audio_engine(model_name)
            result_container['result'] = engine.transcribe(file_path)
        except Exception as e:
            result_container['error'] = e

    t = threading.Thread(target=target)
    t.start()
    
    start_time = time.time()
    while t.is_alive():
        elapsed = time.time() - start_time
        yield gr.update(), gr.Label(value=f"Transcribing... {elapsed:.1f}s", visible=True)
        t.join(timeout=0.1)
        
    if 'error' in result_container:
        yield gr.update(), gr.Label(value=f"Error: {result_container['error']}", visible=True)
    else:
        elapsed = time.time() - start_time
        yield result_container['result'], gr.Label(value=f"Audio transcription successful! ({elapsed:.1f}s)", visible=True)

def transcribe_image(file_path, source_lang):
    if file_path is None:
        yield gr.update(), gr.Label(value="No image uploaded.", visible=True)
        return
    
    result_container = {}
    def target():
        try:
            # Hardcoded model for now as per previous app logic, but using our new engine
            model_name = "Qwen/Qwen2-VL-2B-Instruct" 
            engine = get_vision_engine(model_name)
            result_container['result'] = engine.transcribe_image(file_path, source_lang)
        except Exception as e:
            result_container['error'] = e

    t = threading.Thread(target=target)
    t.start()
    
    start_time = time.time()
    while t.is_alive():
        elapsed = time.time() - start_time
        yield gr.update(), gr.Label(value=f"Transcribing Image... {elapsed:.1f}s", visible=True)
        t.join(timeout=0.1)
        
    if 'error' in result_container:
        yield gr.update(), gr.Label(value=f"Error: {result_container['error']}", visible=True)
    else:
        elapsed = time.time() - start_time
        yield result_container['result'], gr.Label(value=f"Image transcription successful! ({elapsed:.1f}s)", visible=True)

def update_target_languages(source_lang):
    if source_lang == "English":
        return gr.Dropdown(choices=[l for l in LANGUAGES if l != "English"], value="Spanish", interactive=True)
    else:
        return gr.Dropdown(choices=["English"], value="English", interactive=False)

def show_save_button():
    return gr.Button(visible=True), gr.Label(visible=False)

THEME = gr.themes.Soft()
CUSTOM_CSS = """
#translate_button { background: #4CAF50 !important; color: white !important; }
#translate_button:hover { background: #45a049 !important; }
"""

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 FOREO HF Translator")
        gr.Markdown("Powered by Hugging Face Transformers & FastAPI")

        original_translation_state = gr.State()

        with gr.Row():
            model_name = gr.Dropdown(
                label="Text Model",
                value="google/gemma-2-2b-it",
                choices=[
                    "google/gemma-2-2b-it", 
                    "microsoft/Phi-3-mini-4k-instruct",
                    "google/gemma-3-4b-it",
                    "google/gemma-3-12b-it",
                    "google/gemma-2-9b-it"
                ]
            )
            temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.3)

        gr.Markdown("## Text Translation")
        with gr.Row():
            source_lang_dd = gr.Dropdown(label="Source Language", choices=LANGUAGES, value="English")
            target_lang_dd = gr.Dropdown(label="Target Language", choices=["Spanish", "Polish", "Turkish"], value="Spanish")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                source_text = gr.Textbox(lines=15, label="Source Text", placeholder="Type text or upload media...")
            with gr.Column(scale=1, min_width=100):
                translate_button = gr.Button("Translate ➡️", variant="primary", elem_id="translate_button")
            with gr.Column(scale=5):
                target_text = gr.Textbox(lines=15, label="Translated Text", interactive=True)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    transcribe_button = gr.UploadButton("Transcribe Audio/Video 🎵", file_types=["audio", "video"], type="filepath")
                    transcribe_image_button = gr.UploadButton("Transcribe Image (OCR) 🖼️", file_types=["image"], type="filepath")
                
                with gr.Row():
                    whisper_model_dd = gr.Dropdown(label="Whisper Model", value="base", choices=["tiny", "base", "small", "medium", "large"])
                    gr.Markdown("### OCR Model: Qwen/Qwen2-VL-2B-Instruct")
                
                transcription_status = gr.Label(visible=False, show_label=False)

            with gr.Column(scale=1): pass

            with gr.Column(scale=5):
                with gr.Row():
                    feedback_label = gr.Label(visible=False, scale=3)
                    save_button = gr.Button("Save Modification", variant="stop", visible=False, scale=1)

        # Event Wiring
        source_lang_dd.change(update_target_languages, inputs=source_lang_dd, outputs=target_lang_dd)
        
        translate_button.click(
            translate_wrapper,
            inputs=[source_text, model_name, temp, source_lang_dd, target_lang_dd],
            outputs=[target_text, original_translation_state, save_button, feedback_label]
        )

        target_text.input(show_save_button, outputs=[save_button, feedback_label])

        save_button.click(
            save_modification,
            inputs=[source_text, target_text, original_translation_state, source_lang_dd, target_lang_dd],
            outputs=[feedback_label, save_button]
        )

        transcribe_button.upload(
            transcribe_audio,
            inputs=[transcribe_button, whisper_model_dd],
            outputs=[source_text, transcription_status]
        )

        transcribe_image_button.upload(
            transcribe_image,
            inputs=[transcribe_image_button, source_lang_dd],
            outputs=[source_text, transcription_status]
        )

    return demo
