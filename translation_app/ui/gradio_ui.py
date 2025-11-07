import gradio as gr
import time
import os
import sys
import pandas as pd

# --- START OF IMPORT FIX ---
# 1. Get the current script path (ui/traductor_app_gradio.py)
current_script_path = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to the project root (translation_app)
root_path = os.path.dirname(current_script_path)

# 3. Add the root to sys.path so Python can find the 'modules' folder
if root_path not in sys.path:
    sys.path.append(root_path)
# --- END OF IMPORT FIX ---

try:
    # This import should now work correctly
    from modules.llm_call import llm_translation, clean_segment
except ImportError:
    print(f"Error: Could not import 'llm_call' from 'modules' folder.")
    print(f"Make sure the folder structure is correct.")
    print(f"Root path added: {root_path}")
    sys.exit(1)

# --- Define CSV Path ---
CSV_FILE_PATH = os.path.join(root_path, 'data', 'user_mods.csv')

# --- Define Language List ---
LANGUAGES = ["English", "Spanish", "Polish", "Turkish"]


# --- UI Logic Functions ---

def translate_and_clean(text, model, temperature, source_lang, target_lang):
    """
    Gradio wrapper: Calls the translation and then cleans the result.
    Returns:
    1. The translation (for the text box)
    2. The translation (for the hidden state)
    3. Hides the save button
    4. Hides the feedback label
    """
    if not text:
        return "", "", gr.Button(visible=False), gr.Label(visible=False)

    # Pass all arguments to the translation function
    translation = llm_translation(model, text, temperature, source_lang, target_lang)
    clean_trans = clean_segment(translation)

    # (1) target_text, (2) original_translation_state, (3) save_button, (4) feedback_label
    return clean_trans, clean_trans, gr.Button(visible=False), gr.Label(visible=False)


def save_modification(source_text, modified_target_text, original_translation, source_lang, target_lang):
    """
    Saves the user's manual modification to the CSV file.
    Compares original and modified line by line and saves only changes.
    """

    # --- NUEVA LÓGICA DE LÍNEA POR LÍNEA ---

    # 1. Formatear el par de idiomas
    language_pair = f"{source_lang} -> {target_lang}"

    # 2. Convertir bloques de texto en listas de líneas
    source_lines = (source_text or "").split('\n')
    original_target_lines = (original_translation or "").split('\n')
    modified_target_lines = (modified_target_text or "").split('\n')

    new_data = []

    # 3. Iterar y encontrar diferencias
    for i in range(len(modified_target_lines)):

        # Si la línea modificada es diferente de la original
        # O si la línea modificada es nueva (más larga que la original)

        line_changed = False
        if i >= len(original_target_lines):
            # Es una línea nueva añadida por el usuario
            line_changed = True
        elif modified_target_lines[i] != original_target_lines[i]:
            # Es una línea existente que ha sido modificada
            line_changed = True

        # Si la línea cambió Y hay una línea fuente correspondiente
        if line_changed and i < len(source_lines):
            new_data.append({
                'source': source_lines[i],
                'target': modified_target_lines[i],
                'language_pairs': language_pair
            })

    if not new_data:
        return gr.Label(value="No new changes detected.", visible=True), gr.Button(visible=False)

    # --- FIN DE LA NUEVA LÓGICA ---

    try:
        df_new = pd.DataFrame(new_data)

        # Ensure data directory exists
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)

        if os.path.exists(CSV_FILE_PATH):
            df_old = pd.read_csv(CSV_FILE_PATH)
            df_combined = pd.concat([df_old, df_new])
        else:
            df_combined = df_new

        # --- LÓGICA DE DEDUPLICACIÓN ACTUALIZADA ---
        # Clave ahora es 'source' + 'language_pairs'
        df_combined.drop_duplicates(subset=['source', 'language_pairs'], keep='last', inplace=True)
        # --- FIN DE LA LÓGICA DE DEDUPLICACIÓN ---

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


# --- Gradio Interface (Text-Only Tab) ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FOREO SLM Text Translator")
    gr.Markdown("An interface to translate text between selected languages.")

    # --- NUEVO: Estado oculto para la traducción original ---
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
                placeholder="Type your text here..."
            )

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

    # --- Event Logic ---

    # 1. Update target dropdown based on source selection
    source_lang_dd.change(
        fn=update_target_languages,
        inputs=source_lang_dd,
        outputs=target_lang_dd
    )

    # 2. When Translate is clicked, call the model
    #    ACTUALIZADO: Ahora también popu-la 'original_translation_state'
    translate_button.click(
        fn=translate_and_clean,
        inputs=[source_text, model_name, temp, source_lang_dd, target_lang_dd],
        outputs=[
            target_text,
            original_translation_state,  # <--- NUEVA SALIDA
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
    #    ACTUALIZADO: Ahora también pasa el estado y los idiomas
    save_button.click(
        fn=save_modification,
        inputs=[
            source_text,
            target_text,
            original_translation_state,  # <--- NUEVA ENTRADA
            source_lang_dd,  # <--- NUEVA ENTRADA
            target_lang_dd  # <--- NUEVA ENTRADA
        ],
        outputs=[feedback_label, save_button]
    )

if __name__ == "__main__":
    print("Starting Gradio application...")
    demo.launch()