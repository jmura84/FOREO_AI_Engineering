import gradio as gr
import time
import os
import pandas as pd
from datetime import datetime

try:
    from modules.llm_call import llm_translation, clean_segment
    # --- MODIFIED: Import the transcription module ---
    from modules.audio2text import generate_srt_from_file
except ImportError:
    # --- MODIFIED: Updated error message ---
    print(f"Error: ui.gradio_ui could not import 'modules.llm_call' or 'modules.audio2text'.")
    print("Make sure to run the app using 'main.py' from the project root.")
    # Exiting is not ideal, but prevents a bigger crash
    raise

# --- Define CSV Path (robustly) ---
# We need the path to 'data', which is in the root.
# 1. Get this script's path (ui/gradio_ui.py)
current_script_path = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to the project root (translation_app)
root_path = os.path.dirname(current_script_path)
# 3. Build the path to the CSV
CSV_FILE_PATH = os.path.join(root_path, 'data', 'user_mods.csv')

# --- Define Language List ---
LANGUAGES = ["English", "Spanish", "Polish", "Turkish"]


# --- UI Logic Functions ---

def translate_and_clean(text, model, temperature, source_lang, target_lang):
    """
    Gradio wrapper: Calls the translation.
    The llm_translation function now handles its own cleaning.
    """
    if not text:
        return "", "", gr.Button(visible=False), gr.Label(visible=False)

    # Pass all arguments to the translation function
    translation = llm_translation(model, text, temperature, source_lang, target_lang)

    # --- CORRECTION ---
    # We NO LONGER call clean_segment(translation) here.
    # llm_translation is smart enough to handle SRT or simple text.
    clean_trans = translation
    # --- END OF CORRECTION ---

    # (1) target_text, (2) original_translation_state, (3) save_button, (4) feedback_label
    return clean_trans, clean_trans, gr.Button(visible=False), gr.Label(visible=False)


def save_modification(source_text, modified_target_text, original_translation, source_lang, target_lang):
    """
    Saves the user's manual modification to the CSV file.
    Compares original and modified line by line and saves only changes.
    """

    # --- NEW LINE-BY-LINE LOGIC ---

    # 1. Format the language pair
    language_pair = f"{source_lang} -> {target_lang}"

    # --- GET TIMESTAMP ---
    # Get the current time once, formatted without microseconds
    current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    # --- END TIMESTAMP ---

    # 2. Convert text blocks into lists of lines
    source_lines = (source_text or "").split('\n')
    original_target_lines = (original_translation or "").split('\n')
    modified_target_lines = (modified_target_text or "").split('\n')

    new_data = []

    # 3. Iterate and find differences
    for i in range(len(modified_target_lines)):

        # If the modified line is different from the original
        # Or if the modified line is new (longer than the original)

        line_changed = False
        if i >= len(original_target_lines):
            # It's a new line added by the user
            line_changed = True
        elif modified_target_lines[i] != original_target_lines[i]:
            # It's an existing line that has been modified
            line_changed = True

        # --- MODIFICATION ---
        # If the line changed AND there is a corresponding source line AND
        # the source line is NOT EMPTY, then save it.
        if line_changed and i < len(source_lines) and source_lines[i].strip() != "":
            new_data.append({
                'source': source_lines[i],
                'target': modified_target_lines[i],
                'language_pairs': language_pair,
                'datetime': current_time  # <--- ADDED COLUMN
            })
        # --- END OF MODIFICATION ---

    if not new_data:
        return gr.Label(value="No new changes detected.", visible=True), gr.Button(visible=False)

    # --- END OF NEW LOGIC ---

    try:
        df_new = pd.DataFrame(new_data)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

        if os.path.exists(CSV_FILE_PATH):
            df_old = pd.read_csv(CSV_FILE_PATH)
            df_combined = pd.concat([df_old, df_new])
        else:
            df_combined = df_new

        # --- UPDATED DEDUPLICATION LOGIC ---
        # Key is now 'source' + 'language_pairs'
        df_combined.drop_duplicates(subset=['source', 'language_pairs'], keep='last', inplace=True)
        # --- END OF DEDUPLICATION LOGIC ---

        df_combined.to_csv(CSV_FILE_PATH, index=False, encoding='utf-8')

        # Return new component states: (1) Feedback label, (2) Hide save button
        return gr.Label(value="Modification(s) saved successfully!", visible=True), gr.Button(visible=False)

    except Exception as e:
        print(f"Error saving CSV: {e}")
        return gr.Label(value=f"Error: {e}", visible=True), gr.Button(visible=True)  # Keep button visible if error


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
        # Rule 1: Target can be anything *but* English
        new_choices = [lang for lang in LANGUAGES if lang != "English"]
        new_value = "Spanish"  # Default to Spanish
        return gr.Dropdown(choices=new_choices, value=new_value, interactive=True)
    else:
        # Rule 2: Target *must* be English
        new_choices = ["English"]
        new_value = "English"
        # Set interactive=False to "lock" it, or True to just set the value
        return gr.Dropdown(choices=new_choices, value=new_value, interactive=False)


# --- NEW FUNCTION: Transcription Handler ---
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
    # Call your audio2text module
    srt_output = generate_srt_from_file(file_temp_obj.name, whisper_model)

    if srt_output.startswith("Error:"):
        print(f"Transcription Error: {srt_output}")
        # Returns None to the textbox and the error to the status label
        return None, gr.Label(value=srt_output, visible=True)

    print("Transcription successful.")
    # Returns the SRT to the textbox and a success message
    return srt_output, gr.Label(value="Transcription successful! SRT added to source text.", visible=True)


# --- END OF NEW FUNCTION ---


# --- Gradio Interface (Wrapped in a function) ---

def create_app():
    """
    Creates and returns the Gradio app instance.
    """
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 SLM Text Translator")
        gr.Markdown("An interface to translate text between selected languages.")

        # --- NEW: Hidden state for original translation ---
        original_translation_state = gr.State()

        # --- Shared Components (Model and Temperature) ---
        with gr.Row():
            model_name = gr.Dropdown(
                label="Ollama Model",
                value="gemma3:4b-it-qat",
                choices=["gemma2:9b", "gemma3:4b", "gemma3:4b-it-qat", "thinkverse/towerinstruct:latest"]
            )
            temp = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.3
            )

        # --- Language Selection Row ---
        gr.Markdown("## Text Translation")
        with gr.Row():
            source_lang_dd = gr.Dropdown(
                label="Source Language",
                choices=LANGUAGES,
                value="English"
            )
            target_lang_dd = gr.Dropdown(
                label="Target Language",
                choices=["Spanish", "Polish", "Turkish"],  # Initial state (Source is English)
                value="Spanish"
            )

        # --- Text Boxes Row ---
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                source_text = gr.Textbox(
                    lines=10,
                    label="Source Text",
                    placeholder="Type your text here or transcribe an audio file below..."
                )

                # --- NEW TRANSCRIPTION COMPONENTS ---
                with gr.Row():
                    whisper_model_dd = gr.Dropdown(
                        label="Whisper Model",
                        value="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        scale=3
                    )
                    transcribe_button = gr.UploadButton(
                        "Transcribe Audio/Video 🎵",
                        file_types=["audio", "video"],  # Allow audio and video
                        scale=2
                    )
                # --- END OF NEW TRANSCRIPTION COMPONENTS ---

                # --- MODIFIED: Moved the status label INSIDE the column ---
                transcription_status = gr.Label(visible=False, show_label=False)
                # --- END OF MODIFICATION ---

            with gr.Column(scale=1, min_width=100, elem_id="translate-button-col"):
                translate_button = gr.Button(
                    value="Translate ➡️",
                    variant="primary",
                    size="lg"
                )

            with gr.Column(scale=5):
                target_text = gr.Textbox(
                    lines=10,
                    label="Translated Text",
                    interactive=True  # Editable
                )

                # Save button and feedback label
                with gr.Row():
                    feedback_label = gr.Label(visible=False, scale=3)
                    save_button = gr.Button(
                        value="Save Modification",
                        variant="stop",  # Makes it red
                        visible=False,
                        scale=1
                    )

        # --- MODIFIED: The label is now defined above, so we remove it from here ---
        # transcription_status = gr.Label(visible=False, show_label=False)

        # --- Event Logic ---

        # 1. Update target dropdown based on source selection
        source_lang_dd.change(
            fn=update_target_languages,
            inputs=source_lang_dd,
            outputs=target_lang_dd
        )

        # 2. When Translate is clicked, call the model
        translate_button.click(
            fn=translate_and_clean,
            inputs=[source_text, model_name, temp, source_lang_dd, target_lang_dd],
            outputs=[
                target_text,
                original_translation_state,
                save_button,
                feedback_label
            ]
        )

        # 3. When the user *types* in the target box, show the save button
        target_text.input(
            fn=show_save_button,
            inputs=None,
            outputs=[save_button, feedback_label]
        )

        # 4. When Save is clicked, run the save function
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

        # --- NEW TRANSCRIPTION EVENT LOGIC ---
        # Use .then() to chain actions:
        # 1. (Upload) Show "Transcribing..."
        # 2. (Then) Execute transcription (which is slow)
        transcribe_button.upload(
            fn=handle_transcription_start,
            inputs=None,
            outputs=[transcription_status]
        ).then(
            fn=handle_transcription_complete,
            inputs=[transcribe_button, whisper_model_dd],
            outputs=[source_text, transcription_status]
        )
        # --- END OF NEW LOGIC ---

    # Returns the Gradio app for main.py to launch
    return demo