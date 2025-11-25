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

def translate_wrapper(text, model_name, temperature, source_lang, target_lang):
    if not text:
        return "", "", gr.Button(visible=False), gr.Label(visible=False)

    print(f"Translating with {model_name}...")
    
    # Get engine and translate
    engine = get_llm_engine(model_name)
    raw_translation = engine.translate(text, source_lang, target_lang, temperature)

    # Apply corrections
    final_translation = review_and_correct(
        raw_translation, text, CSV_FILE_PATH, source_lang, target_lang
    )

    return final_translation, final_translation, gr.Button(visible=False), gr.Label(visible=False)

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

def transcribe_audio(file_obj, model_name):
    if file_obj is None:
        return None, gr.Label(value="No file uploaded.", visible=True)
    
    engine = get_audio_engine(model_name)
    srt = engine.transcribe(file_obj.name)
    return srt, gr.Label(value="Audio transcription successful!", visible=True)

def transcribe_image(file_obj, source_lang):
    if file_obj is None:
        return None, gr.Label(value="No image uploaded.", visible=True)
    
    # Hardcoded model for now as per previous app logic, but using our new engine
    model_name = "Qwen/Qwen2-VL-2B-Instruct" 
    engine = get_vision_engine(model_name)
    text = engine.transcribe_image(file_obj.name, source_lang)
    return text, gr.Label(value="Image transcription successful!", visible=True)

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
                choices=["google/gemma-2-2b-it", "microsoft/Phi-3-mini-4k-instruct"]
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
                    transcribe_button = gr.UploadButton("Transcribe Audio/Video 🎵", file_types=["audio", "video"])
                    transcribe_image_button = gr.UploadButton("Transcribe Image (OCR) 🖼️", file_types=["image"])
                
                with gr.Row():
                    whisper_model_dd = gr.Dropdown(label="Whisper Model", value="base", choices=["tiny", "base", "small"])
                    gr.Markdown("OCR Model: Qwen2-VL-2B-Instruct")
                
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

        transcribe_button.upload(lambda: gr.Label(value="Transcribing...", visible=True), outputs=transcription_status) \
            .then(transcribe_audio, inputs=[transcribe_button, whisper_model_dd], outputs=[source_text, transcription_status])

        transcribe_image_button.upload(lambda: gr.Label(value="Transcribing Image...", visible=True), outputs=transcription_status) \
            .then(transcribe_image, inputs=[transcribe_image_button, source_lang_dd], outputs=[source_text, transcription_status])

    return demo
