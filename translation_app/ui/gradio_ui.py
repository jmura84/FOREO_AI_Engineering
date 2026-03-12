import gradio as gr
import time
import os
import pandas as pd
from datetime import datetime

try:
    # --- STEP 1: Import Agent 1 (Translator) ---
    from modules.llm_call import llm_translation, clean_segment

    # --- STEP 2: Import TM Processor (with the new name) ---
    from modules.user_mods_corrector import review_and_correct

    # --- Import Transcriber ---
    from modules.audio2text import generate_srt_from_file

    # --- NEW: Import Image Transcriber ---
    from modules.img_transcriptor import image_ocr_llm_langchain

    # --- NEW: Import RAG Engine ---
    from modules.rag_engine import get_rag_engine

except ImportError:
    # --- Updated error message ---
    print(
        f"Error: ui.gradio_ui could not import modules. Check 'llm_call', 'user_mods_corrector', 'audio2text', or 'img_transcriptor'.")
    print("Make sure to run the app using 'main.py' from the project root.")
    raise

# --- Define CSV Path (robustly) ---
# We need the path to 'data', which is in the root.
# 1. Get this script's path (ui/gradio_ui.py)
current_script_path = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to the project root (translation_app)
root_path = os.path.dirname(current_script_path)
# 3. Build the path to the CSV (with the new name)
CSV_FILE_PATH = os.path.join(root_path, 'data', 'user_mods_tm.csv')

# --- Define Language List ---
LANGUAGES = ["English", "Spanish", "Polish", "Turkish"]


# --- UI Logic Functions ---

def translate_and_clean(text, model, temperature, source_lang, target_lang, use_rag=False):
    """
    Gradio wrapper: Calls the sequential pipeline.
    1. Calls llm_translation (Agent 1) for a raw translation.
    2. Calls review_and_correct (TM Processor) to post-edit the raw translation.
    """
    if not text:
        return "", "", gr.Button(visible=False), gr.Label(visible=False)

    rag_context = None
    # Enable RAG logic if requested
    is_rag_pair = (source_lang == "English" and target_lang == "Spanish") or (source_lang == "Spanish" and target_lang == "English")
    
    if use_rag and is_rag_pair:
        try:
            print("Orchestrator: RAG enabled. Retrieving context...")
            rag_engine = get_rag_engine()
            best_target, context_pairs = rag_engine.retrieve_context(text, source_lang=source_lang)
            
            # If perfect match found, we could potentially just return it. 
            # However, the user might want the full pipeline (corrector etc).
            # But usually if it's a perfect match from TM/RAG, we should probably trust it.
            # Let's pass it as context for now or if best_target is returnable.
            # The HF implementation returned best_target directly. Let's do that for consistency if desired.
            if best_target:
                 print("Orchestrator: Perfect RAG match found.")
                 clean_trans = best_target
                 return clean_trans, clean_trans, gr.Button(visible=False), gr.Label(value="Translation complete (Cached)", visible=True)

            # Otherwise prepare prompt
            rag_context = rag_engine.format_rag_prompt(text, context_pairs, source_lang=source_lang, target_lang=target_lang)
            
        except Exception as e:
            print(f"RAG Error: {e}")
            rag_context = None

    # --- STEP 1: Call Agent 1 (Raw Translation) ---
    print("Orchestrator: Calling Agent 1 (Raw Translation)...")
    raw_translation = llm_translation(model, text, temperature, source_lang, target_lang, rag_context=rag_context)

    # --- STEP 2: Call TM Processor (Post-Editing) ---
    print("Orchestrator: Calling TM Processor (Post-Editing and Review)...")
    final_translation = review_and_correct(
        raw_translation=raw_translation,
        source_text=text,
        csv_path=CSV_FILE_PATH,  # Pass the correct CSV path
        model_name=model,  # Not used by TM, but passed for consistency
        temp=temperature,  # Not used by TM
        source_lang=source_lang,
        target_lang=target_lang
    )

    clean_trans = final_translation

    # (1) target_text, (2) original_translation_state, (3) save_button, (4) feedback_label
    # We save the *final translation* in the state, not the raw one.
    return clean_trans, clean_trans, gr.Button(visible=False), gr.Label(visible=False)


def save_modification(source_text, modified_target_text, original_translation, source_lang, target_lang):
    """
    Saves the user's manual modification to the CSV file.
    Compares original and modified line by line and saves only changes.
    """

    # 1. Format the language pair
    language_pair = f"{source_lang} -> {target_lang}"

    # Get the current time once, formatted without microseconds
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    # 2. Convert text blocks into lists of lines
    source_lines = (source_text or "").split('\n')
    original_target_lines = (original_translation or "").split('\n')
    modified_target_lines = (modified_target_text or "").split('\n')

    new_data = []

    # 3. Iterate and find differences
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
        df_new = pd.DataFrame(new_data)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

        if os.path.exists(CSV_FILE_PATH):
            df_old = pd.read_csv(CSV_FILE_PATH)
            df_combined = pd.concat([df_old, df_new])
        else:
            df_combined = df_new

        # 1. Create a temporary 'source_normalized' column
        df_combined['source_normalized'] = df_combined['source'].astype(str) \
            .str.lower() \
            .str.strip() \
            .str.rstrip('.,!;')  # '?' is preserved

        # 2. Use this normalized key for deduplication
        df_combined.drop_duplicates(subset=['source_normalized', 'language_pairs'], keep='last', inplace=True)

        # 3. Drop the temporary column before saving
        df_combined.drop(columns=['source_normalized'], inplace=True)

        df_combined.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')

        return gr.Label(value="Modification(s) saved successfully!", visible=True), gr.Button(visible=False)

    except Exception as e:
        print(f"Error saving CSV: {e}")
        return gr.Label(value=f"Error: {e}", visible=True), gr.Button(visible=True)


def show_save_button():
    """
    Called when user types in the target box.
    Shows the save button and hides any old feedback.
    """
    return gr.Button(visible=True), gr.Label(visible=False)


def update_target_languages(source_lang):
    """
    Handles the logic for updating the target language dropdown
    based on the source language selection.
    """
    if source_lang == "English":
        new_choices = [lang for lang in LANGUAGES if lang != "English"]
        new_value = "Spanish"
        return gr.Dropdown(choices=new_choices, value=new_value, interactive=True)
    else:
        new_choices = ["English"]
        new_value = "English"
        return gr.Dropdown(choices=new_choices, value=new_value, interactive=False)


def update_rag_visibility(source_lang, target_lang):
    """
    Updates the RAG checkbox visibility based on language pair.
    Only enables it for English <-> Spanish.
    """
    if (source_lang == "English" and target_lang == "Spanish") or \
       (source_lang == "Spanish" and target_lang == "English"):
        return gr.Checkbox(interactive=True, label="RAG (English <-> Spanish only)", value=True)
    else:
        return gr.Checkbox(value=False, interactive=False, label="RAG (English <-> Spanish only)")


def handle_transcription_start():
    """Shows the 'Transcribing' status label."""
    return gr.Label(value="Transcribing... This may take a moment.", visible=True)


def handle_transcription_complete(file_temp_obj, whisper_model):
    """
    Called when a file is uploaded. Runs Whisper and returns the SRT.
    """
    if file_temp_obj is None:
        return None, gr.Label(value="File upload cancelled or failed.", visible=True)

    print(f"Starting transcription for: {file_temp_obj.name}")
    srt_output = generate_srt_from_file(file_temp_obj.name, whisper_model)

    if srt_output.startswith("Error:"):
        print(f"Transcription Error: {srt_output}")
        return None, gr.Label(value=srt_output, visible=True)

    print("Transcription successful.")
    return srt_output, gr.Label(value="Transcription successful! SRT added to source text.", visible=True)


# --- NEW IMAGE HANDLER FUNCTIONS ---
def handle_image_transcription_start():
    """Shows the 'Transcribing' status label."""
    # Re-use the same status label
    return gr.Label(value="Transcribing Image... This may take a moment.", visible=True)


def handle_image_transcription_complete(file_temp_obj, source_lang):
    """
    Called when an image is uploaded. Runs the multimodal LLM.
    """
    if file_temp_obj is None:
        return None, gr.Label(value="Image upload cancelled or failed.", visible=True)

    print(f"Starting image transcription for: {file_temp_obj.name}")

    # --- Hardcode the model name as requested ---
    # https://docs.unsloth.ai/new/deepseek-ocr-how-to-run-and-fine-tune
    # model_name = "gemma3:12b"
    model_name = "deepseek-ocr:3b"

    # Call the img_transcriptor module function
    transcription = image_ocr_llm_langchain(
        model_name=model_name,
        input_img_path=file_temp_obj.name,  # Pass the temp file path
        source_lang=source_lang
    )

    if transcription.startswith("Error:"):
        print(f"Image Transcription Error: {transcription}")
        return None, gr.Label(value=transcription, visible=True)

    print("Image transcription successful.")
    # Return transcription to source_text and a success message
    return transcription, gr.Label(value="Image transcription successful! Text added to source.", visible=True)


# --- END OF NEW FUNCTIONS ---


def create_app():
    """
    Creates and returns the Gradio app instance.
    """

    custom_css = """
    #translate_button {
        background: #4CAF50 !important; /* Green */
        color: white !important;
    }
    #translate_button:hover {
        background: #45a049 !important; /* Darker Green */
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# 🤖 FOREO SLM Translator")
        gr.Markdown("A multimodal and multilingual translator")

        original_translation_state = gr.State()

        with gr.Row():
            model_name = gr.Dropdown(
                label="Ollama Model",
                value="gemma3:4b-it-qat",
                # Added gemma3:12b to the list
                choices=["gemma2:9b", "gemma3:4b", "gemma3:4b-it-qat", "omercelik/gemmax2-9b:latest", "gemma3:12b", "gemma3:12b-it-qat", "thinkverse/towerinstruct:latest"]
            )
            temp = gr.Slider(
                label="Temperature (0 = More literal, 1 = More creative)",
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.3
            )

        gr.Markdown("## Text Translation")
        with gr.Row():
            source_lang_dd = gr.Dropdown(
                label="Source Language",
                choices=LANGUAGES,
                value="English"
            )
            target_lang_dd = gr.Dropdown(
                label="Target Language",
                # Antes tenía: choices=["Spanish", "Polish", "Turkish"],
                choices=LANGUAGES,
                value="Spanish"
            )
        
        # --- NEW: RAG Checkbox ---
        rag_checkbox = gr.Checkbox(label="RAG (English <-> Spanish only)", value=False, interactive=True)

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                source_text = gr.Textbox(
                    lines=15,
                    label="Source Text",
                    placeholder="Type your text here or transcribe a video, an audio, or an image file below..."
                )

            with gr.Column(scale=1, min_width=100, elem_id="translate-button-col"):
                translate_button = gr.Button(
                    value="Translate ➡️",
                    variant="primary",
                    size="md",  # Compact button
                    elem_id="translate_button"
                )

            with gr.Column(scale=5):
                target_text = gr.Textbox(
                    lines=15,
                    label="Translated Text",
                    interactive=True
                )

        with gr.Row():
            with gr.Column(scale=5):
                # --- MODIFIED: Re-arranged layout ---

                # --- NEW: Row for Buttons ---
                with gr.Row():
                    with gr.Column(scale=1):
                        transcribe_button = gr.UploadButton(
                            "Transcribe Audio/Video (SRT) 🎵",
                            file_types=["audio", "video"],
                        )
                    with gr.Column(scale=1):
                        transcribe_image_button = gr.UploadButton(  # <--- NEW BUTTON
                            "Transcribe Image (OCR) 🖼️",
                            file_types=["image"],  # Only allow images
                        )

                # --- NEW: Row for Controls/Labels Underneath ---
                with gr.Row():
                    with gr.Column(scale=1):
                        whisper_model_dd = gr.Dropdown(
                            label="Whisper Model (Audio/Video Transcriptor)",  # Simplified label
                            value="base",
                            choices=["tiny", "base", "small", "medium", "large"],
                        )
                        gr.Markdown("Smaller = faster but less efficient<br>Bigger = slower but better transcription quality",)
                    with gr.Column(scale=1):
                        gr.Markdown("OCR Model (Fixed: DeepSeek OCR 3b)")

                # This status label is now shared by both buttons
                transcription_status = gr.Label(visible=False, show_label=False)
                # --- END MODIFICATION ---

            with gr.Column(scale=1, min_width=100):
                pass

            with gr.Column(scale=5):
                with gr.Row():
                    feedback_label = gr.Label(visible=False, scale=3)
                    save_button = gr.Button(
                        value="Save Modification",
                        variant="stop",
                        visible=False,
                        scale=1
                    )

        source_lang_dd.change(
            fn=update_target_languages,
            inputs=source_lang_dd,
            outputs=target_lang_dd
        )

        # --- NEW: RAG Visibility Wiring ---
        source_lang_dd.change(update_rag_visibility, inputs=[source_lang_dd, target_lang_dd], outputs=rag_checkbox)
        target_lang_dd.change(update_rag_visibility, inputs=[source_lang_dd, target_lang_dd], outputs=rag_checkbox)

        translate_button.click(
            fn=translate_and_clean,
            inputs=[source_text, model_name, temp, source_lang_dd, target_lang_dd, rag_checkbox],
            outputs=[
                target_text,
                original_translation_state,
                save_button,
                feedback_label
            ]
        )

        target_text.input(
            fn=show_save_button,
            inputs=None,
            outputs=[save_button, feedback_label]
        )

        save_button.click(
            fn=save_modification,
            inputs=[
                source_text,
                target_text,
                original_translation_state,
                source_lang_dd,
                target_lang_dd
            ],
            outputs=[feedback_label, save_button]
        )

        transcribe_button.upload(
            fn=handle_transcription_start,
            inputs=None,
            outputs=[transcription_status]
        ).then(
            fn=handle_transcription_complete,
            inputs=[transcribe_button, whisper_model_dd],
            outputs=[source_text, transcription_status]
        )

        # --- NEW IMAGE TRANSCRIPTION EVENT LOGIC ---
        transcribe_image_button.upload(
            fn=handle_image_transcription_start,
            inputs=None,
            outputs=[transcription_status]
        ).then(
            fn=handle_image_transcription_complete,
            # Inputs: the button itself (for file) and the source language
            inputs=[transcribe_image_button, source_lang_dd],
            # Outputs: the source text box and the status label
            outputs=[source_text, transcription_status]
        )
        # --- END OF NEW LOGIC ---

    return demo