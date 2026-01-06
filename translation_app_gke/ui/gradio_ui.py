import gradio as gr
import time
import os
import pandas as pd
from datetime import datetime

try:
    # --- Imports ---
    from modules.llm_call import llm_translation
    from modules.user_mods_corrector import review_and_correct
    from modules.audio2text import generate_srt_from_file
    from modules.img_transcriptor import image_ocr_llm_langchain
    from modules.rag_manager import get_rag_context_string
except ImportError:
    print("Error: ui.gradio_ui could not import modules. Run main.py from root.")
    raise

# --- Config ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
CSV_FILE_PATH = os.path.join(root_path, 'data', 'user_mods_tm.csv')

LANGUAGES = ["English", "Spanish", "Polish", "Turkish"]

# --- Model Choices (Matching HF App) ---
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

# --- Logic ---

def translate_and_clean(text, model_label_or_value, temperature, source_lang, target_lang, use_rag):
    if not text:
        return "", "", gr.Button(visible=False), gr.Label(visible=False)

    # Map label to value if needed
    model_name = LABEL_TO_VALUE.get(model_label_or_value, model_label_or_value)
    
    context_text = None
    
    # RAG Logic (Only for En -> Es as per requirements)
    if use_rag and source_lang == "English" and target_lang == "Spanish":
        print("Orchestrator: RAG Enabled. Retrieving context...")
        rag_context, best_match = get_rag_context_string(text)
        
        if best_match:
            print("Orchestrator: Strong RAG match found. Using it directly.")
            context_text = rag_context
        elif rag_context:
            print(f"Orchestrator: Context retrieved ({len(rag_context)} chars).")
            context_text = rag_context
        else:
            print("Orchestrator: No relevant context found.")

    # Call Gemini (Agent 1)
    print(f"Orchestrator: Calling Agent 1 ({model_name})...")
    raw_translation = llm_translation(model_name, text, temperature, source_lang, target_lang, context_text=context_text)

    # Call TM Processor (Post-Editing)
    print("Orchestrator: Calling TM Processor...")
    final_translation = review_and_correct(
        raw_translation=raw_translation,
        source_text=text,
        csv_path=CSV_FILE_PATH,
        model_name=model_name,
        temp=temperature,
        source_lang=source_lang,
        target_lang=target_lang
    )

    clean_trans = final_translation
    return clean_trans, clean_trans, gr.Button(visible=False), gr.Label(visible=False)


def save_modification(source_text, modified_target_text, original_translation, source_lang, target_lang):
    """Saves user modifications to CSV."""
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
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        if os.path.exists(CSV_FILE_PATH):
            df_old = pd.read_csv(CSV_FILE_PATH)
            df_combined = pd.concat([df_old, pd.DataFrame(new_data)])
        else:
            df_combined = pd.DataFrame(new_data)

        # Deduplicate
        df_combined['source_normalized'] = df_combined['source'].astype(str).str.lower().str.strip().str.rstrip('.,!;')
        df_combined.drop_duplicates(subset=['source_normalized', 'language_pairs'], keep='last', inplace=True)
        df_combined.drop(columns=['source_normalized'], inplace=True)

        df_combined.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')
        return gr.Label(value="Modification(s) saved successfully!", visible=True), gr.Button(visible=False)

    except Exception as e:
        print(f"Error saving CSV: {e}")
        return gr.Label(value=f"Error: {e}", visible=True), gr.Button(visible=True)


def show_save_button():
    return gr.Button(visible=True), gr.Label(visible=False)


def update_target_languages(source_lang):
    if source_lang == "English":
        return gr.Dropdown(choices=[l for l in LANGUAGES if l != "English"], value="Spanish", interactive=True), gr.Checkbox(visible=True)
    else:
        return gr.Dropdown(choices=["English"], value="English", interactive=False), gr.Checkbox(visible=False, value=False)


def handle_transcription_complete(file_temp_obj, whisper_model):
    if file_temp_obj is None:
        return None, gr.Label(value="File upload cancelled.", visible=True)
    
    print(f"Transcribing audio: {file_temp_obj.name}")
    srt_output = generate_srt_from_file(file_temp_obj.name, whisper_model)
    
    if srt_output.startswith("Error:"):
        return None, gr.Label(value=srt_output, visible=True)
        
    return srt_output, gr.Label(value="Audio Transcription Successful!", visible=True)


def handle_image_transcription_complete(file_temp_obj, source_lang, ocr_model):
    if file_temp_obj is None:
        return None, gr.Label(value="Image upload cancelled.", visible=True)

    print(f"Transcribing image: {file_temp_obj.name} with {ocr_model}")
    # Use the refactored Gemini Vision function
    transcription = image_ocr_llm_langchain(
        model_name=ocr_model, 
        input_img_path=file_temp_obj.name,
        source_lang=source_lang
    )

    if transcription.startswith("Error:"):
        return None, gr.Label(value=transcription, visible=True)

    return transcription, gr.Label(value="Image OCR Successful!", visible=True)


def create_app():
    custom_css = """
    #translate_button { background: #4CAF50 !important; color: white !important; }
    #translate_button:hover { background: #45a049 !important; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# 🤖 FOREO AI Translator (Gemini Edition)")
        gr.Markdown("Multimodal translator powered by Google Gemini.")

        original_translation_state = gr.State()

        with gr.Row():
            # Dropdown for Models
            model_dd = gr.Dropdown(
                label="Translation Model",
                choices=MODEL_LABELS,
                value=MODEL_LABELS[2] # Default to gemini-2.5-flash-lite or similar middle ground
            )
            temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1, value=0.3)

        gr.Markdown("## Text Translation")
        with gr.Row():
            source_lang_dd = gr.Dropdown(label="Source", choices=LANGUAGES, value="English")
            target_lang_dd = gr.Dropdown(label="Target", choices=LANGUAGES, value="Spanish")
            
        with gr.Row():
             use_rag_cb = gr.Checkbox(label="Use RAG (English -> Spanish only)", value=True, visible=True)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                source_text = gr.Textbox(lines=15, label="Source Text", placeholder="Text, SRT, or transcribe below...")
            
            with gr.Column(scale=1, min_width=100):
                translate_button = gr.Button("Translate ➡️", variant="primary", elem_id="translate_button")

            with gr.Column(scale=5):
                target_text = gr.Textbox(lines=15, label="Translated Text", interactive=True)

        with gr.Row():
            # Transcribe buttons
            with gr.Column(scale=1):
                transcribe_button = gr.UploadButton("Transcribe Audio/Video (SRT) 🎵", file_types=["audio", "video"])
                whisper_model_dd = gr.Dropdown(label="Whisper Model", value="base", choices=["tiny", "base", "small", "medium", "large"])
            
            with gr.Column(scale=1):
                transcribe_image_button = gr.UploadButton("Transcribe Image (OCR) 🖼️", file_types=["image"])
                # OCR Model Selection
                ocr_model_dd = gr.Dropdown(
                    label="OCR Model",
                    value="gemini-2.5-flash",
                    choices=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                )

            transcription_status = gr.Label(visible=False, show_label=False)

        # Feedback/Save
        with gr.Row():
            feedback_label = gr.Label(visible=False)
            save_button = gr.Button("Save Modification", variant="stop", visible=False)

        # Events
        source_lang_dd.change(update_target_languages, inputs=source_lang_dd, outputs=[target_lang_dd, use_rag_cb])

        translate_button.click(
            translate_and_clean,
            inputs=[source_text, model_dd, temp, source_lang_dd, target_lang_dd, use_rag_cb],
            outputs=[target_text, original_translation_state, save_button, feedback_label]
        )

        target_text.input(show_save_button, None, [save_button, feedback_label])

        save_button.click(
            save_modification,
            inputs=[source_text, target_text, original_translation_state, source_lang_dd, target_lang_dd],
            outputs=[feedback_label, save_button]
        )

        transcribe_button.upload(lambda: gr.Label(value="Transcribing...", visible=True), None, transcription_status).then(
            handle_transcription_complete, [transcribe_button, whisper_model_dd], [source_text, transcription_status]
        )

        # Updated Image Handler
        transcribe_image_button.upload(lambda: gr.Label(value="Reading Image...", visible=True), None, transcription_status).then(
            handle_image_transcription_complete, [transcribe_image_button, source_lang_dd, ocr_model_dd], [source_text, transcription_status]
        )

    return demo