import gradio as gr
import os
import pandas as pd
from datetime import datetime
from core.llm_engine import get_llm_engine
from core.vision_engine import get_vision_engine
from core.audio_engine import get_audio_engine
from core.rag_engine import get_rag_engine
from core.corrector import review_and_correct
import time
import threading

# -----------------------------
# Paths and config
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CSV_FILE_PATH = os.path.join(DATA_DIR, 'user_mods_tm.csv')

LANGUAGES = ["English", "Spanish", "Polish", "Turkish"]

CUSTOM_CSS = """
#translate_button { background: #4CAF50 !important; color: white !important; }
#translate_button:hover { background: #45a049 !important; }
"""

# -----------------------------
# Language model choices
# -----------------------------
MODEL_CHOICES = [
    {"label": "gemini-2.0-flash-lite (Lowest latency - Highest rate limit)", "value": "gemini-2.0-flash-lite"},
    {"label": "gemini-2.0-flash (Good quality - High rate limit)", "value": "gemini-2.0-flash"},
    {"label": "gemini-2.5-flash-lite (Lowest latency - Highest rate limit)", "value": "gemini-2.5-flash-lite"},
    {"label": "gemini-2.5-flash (Good quality - High rate limit)", "value": "gemini-2.5-flash"},
    {"label": "gemini-2.5-pro (Best quality - Lower rate limit)", "value": "gemini-2.5-pro"},
    {"label": "google/gemma-2-2b-it (Open-source - No rate limits)", "value": "google/gemma-2-2b-it"}
]

MODEL_LABELS = [m["label"] for m in MODEL_CHOICES]
LABEL_TO_VALUE = {m["label"]: m["value"] for m in MODEL_CHOICES}

# -----------------------------
# Thread helper
# -----------------------------
def run_with_thread(func, args=(), update_interval=0.1):
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

# -----------------------------
# Translation wrapper
# -----------------------------
def translate_wrapper(text, model_name_label, temperature, source_lang, target_lang, use_rag):
    if not text:
        yield "", "", gr.Button(visible=False), gr.Label(visible=False)
        return

    model_name = LABEL_TO_VALUE[model_name_label]

    def _translate_task():
        engine = get_llm_engine(model_name)
        
        rag_context = None
        rag_context = None
        # Enable RAG for Eng->Spa AND Spa->Eng
        is_rag_pair = (source_lang == "English" and target_lang == "Spanish") or (source_lang == "Spanish" and target_lang == "English")
        
        if use_rag and is_rag_pair:
            try:
                rag_engine = get_rag_engine()
                # Retrieve context (bidirectional logic in engine)
                best_target, context_pairs = rag_engine.retrieve_context(text, source_lang=source_lang)
                
                # If perfect match found, return it
                if best_target:
                    return best_target
                
                # Otherwise prepare prompt
                rag_context = rag_engine.format_rag_prompt(text, context_pairs, source_lang=source_lang, target_lang=target_lang)
                
            except Exception as e:
                print(f"RAG Error: {e}")
                # Fallback to standard translation if RAG fails
                rag_context = None

        # Standard Translation (with potential RAG context injected)
        raw_translation = engine.translate(text, source_lang, target_lang, temperature, rag_context=rag_context)
        
        # If we used RAG, we might skip the corrector or keep it. 
        # The notebook didn't use a corrector, but the app does. 
        # Let's keep the corrector for consistency unless it interferes.
        return review_and_correct(raw_translation, text, CSV_FILE_PATH, source_lang, target_lang)

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

# -----------------------------
# Save user modifications
# -----------------------------
def save_modification(source_text, modified_target_text, original_translation, source_lang, target_lang):
    language_pair = f"{source_lang} -> {target_lang}"
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    source_lines = (source_text or "").split('\n')
    original_target_lines = (original_translation or "").split('\n')
    modified_target_lines = (modified_target_text or "").split('\n')

    new_data = []
    for i in range(len(modified_target_lines)):
        line_changed = False
        if i >= len(original_target_lines) or modified_target_lines[i] != original_target_lines[i]:
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

# -----------------------------
# Audio transcription (Whisper)
# -----------------------------
def transcribe_audio(file_path, whisper_model):
    if file_path is None:
        yield gr.update(), gr.Label(value="No file uploaded.", visible=True)
        return

    result_container = {}
    def target():
        try:
            engine = get_audio_engine(whisper_model)
            engine.load_model()
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

# -----------------------------
# Image transcription (Qwen OR Gemini)
# -----------------------------
def transcribe_image(file_path, source_lang, ocr_model_label):
    if file_path is None:
        yield gr.update(), gr.Label(value="No image uploaded.", visible=True)
        return

    # Detect selected OCR model
    if "gemini" in ocr_model_label.lower():
        selected_model = "gemini"
    else:
        selected_model = "qwen"

    result_container = {}

    def target():
        try:
            if selected_model == "qwen":
                # Local Qwen OCR
                engine = get_vision_engine("Qwen/Qwen2-VL-2B-Instruct")
                engine.load_model()
                result_container['result'] = engine.transcribe_image(file_path, source_lang)

            else:
                # Gemini OCR (via Google API)
                import google.generativeai as genai
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is required for Gemini OCR")

                genai.configure(api_key=api_key)

                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    generation_config={
                        "temperature": 0.0,
                        "top_p": 0.0,
                        "top_k": 1,
                        "max_output_tokens": 2048,
                        "response_mime_type": "text/plain",
                    }
                )

                prompt = (
                    "Extract all text EXACTLY as it appears in the image. "
                    "Do NOT add, remove, rewrite, correct, or guess anything. "
                    "Return ONLY the text present in the image."
                )

                with open(file_path, "rb") as f:
                    img_bytes = f.read()

                response = model.generate_content(
                    [prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
                )

                result_container["result"] = response.text.strip()

        except Exception as e:
            result_container['error'] = e

    # Threading wrapper
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

# -----------------------------
# UI helpers
# -----------------------------
def update_other_dropdown(selected_val, other_val):
    if selected_val != "English":
        return gr.Dropdown(value="English", choices=["English"], interactive=False)
    else:
        new_val = other_val
        if new_val == "English":
            new_val = "Spanish"
        return gr.Dropdown(value=new_val, choices=LANGUAGES, interactive=True)

def show_save_button():
    return gr.Button(visible=True), gr.Label(visible=False)

def update_rag_visibility(source_lang, target_lang):
    if (source_lang == "English" and target_lang == "Spanish") or \
       (source_lang == "Spanish" and target_lang == "English"):
        return gr.Checkbox(interactive=True, label="RAG (English <-> Spanish only)")
    else:
        return gr.Checkbox(value=False, interactive=False, label="RAG (English <-> Spanish only)")

# -----------------------------
# Gradio Interface
# -----------------------------
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 🤖 FOREO HF Translator")
        gr.Markdown("Powered by Hugging Face, Google, OpenAI & Qwen/Vision Models")

        original_translation_state = gr.State()

        # Language model + temperature
        with gr.Row():
            model_name_label = gr.Dropdown(
                label="Language Model",
                value="gemini-2.5-flash (Good quality - High rate limit)",
                choices=MODEL_LABELS
            )
            temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.3)


        # Text Translation
        gr.Markdown("## Text Translation")
        with gr.Row():
            source_lang_dd = gr.Dropdown(label="Source Language", choices=LANGUAGES, value="English")
            target_lang_dd = gr.Dropdown(label="Target Language", choices=LANGUAGES, value="Spanish")
        
        rag_checkbox = gr.Checkbox(label="RAG (English <-> Spanish only)", value=False)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                source_text = gr.Textbox(lines=15, label="Source Text", placeholder="Type text or upload media...")
            with gr.Column(scale=1, min_width=100):
                translate_button = gr.Button("Translate ➡️", variant="primary", elem_id="translate_button")
            with gr.Column(scale=5):
                target_text = gr.Textbox(lines=15, label="Translated Text", interactive=True)

        # Audio + OCR
        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row():
                    transcribe_button = gr.UploadButton(
                        "Transcribe Audio/Video (SRT) 🎵",
                        file_types=["audio", "video"], type="filepath"
                    )
                    transcribe_image_button = gr.UploadButton(
                        "Transcribe Image (OCR) 🖼️",
                        file_types=["image"], type="filepath"
                    )
                with gr.Row():
                    whisper_model_dd = gr.Dropdown(
                        label="Whisper Model",
                        value="medium",
                        choices=["tiny","base","small","medium","large"]
                    )
                    ocr_model_dd = gr.Dropdown(
                        label="OCR Model",
                        value="gemini-2.5-flash (Fast - Rate limited)",
                        choices=[
                            "Qwen/Qwen2-VL-2B-Instruct (Slow - No rate limits)",
                            "gemini-2.5-flash (Fast - Rate limited)"
                        ],
                    )

                transcription_status = gr.Label(visible=False, show_label=False)

            with gr.Column(scale=1): 
                pass

            with gr.Column(scale=5):
                with gr.Row():
                    feedback_label = gr.Label(visible=False, scale=3)
                    save_button = gr.Button("Save Modification", variant="stop", visible=False, scale=1)

        # Event wiring
        source_lang_dd.change(update_other_dropdown, inputs=[source_lang_dd, target_lang_dd], outputs=target_lang_dd)
        target_lang_dd.change(update_other_dropdown, inputs=[target_lang_dd, source_lang_dd], outputs=source_lang_dd)
        
        # Update RAG checkbox availability when languages change
        source_lang_dd.change(update_rag_visibility, inputs=[source_lang_dd, target_lang_dd], outputs=rag_checkbox)
        target_lang_dd.change(update_rag_visibility, inputs=[source_lang_dd, target_lang_dd], outputs=rag_checkbox)

        translate_button.click(
            translate_wrapper,
            inputs=[source_text, model_name_label, temp, source_lang_dd, target_lang_dd, rag_checkbox],
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
            inputs=[transcribe_image_button, source_lang_dd, ocr_model_dd],
            outputs=[source_text, transcription_status]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
