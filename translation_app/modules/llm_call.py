import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)
import torch
import logging
import re
import time

logging.getLogger("langchain.utilities.requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# If the log comes directly from the LangChain core:
logging.getLogger("langchain.llms").setLevel(logging.WARNING)

# We check if the library is using the GPU:
print(f"CUDA available according to PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device detected: {torch.cuda.get_device_name(0)}")

# It can be 'medium' or 'high' precision:
torch.set_float32_matmul_precision('high')


def llm_translation(model_name: str, input_text: str, temp: float, source_lang: str, target_lang: str) -> str:
    """
    Translates the given text using the specified Ollama model and languages.
    Detects SRT format and translates segment by segment if present.
    """

    # --- DYNAMIC PROMPT (STRICTER) ---
    system_template = (
        f"You are a professional translator. "
        f"Your task is to translate *only* the user's text from {source_lang} to {target_lang}. "
        f"If the target language is Spanish, use Spanish from Spain dialect. "
        "The texts you translate are from a beauty products brand called FOREO. "
        "Translate the exact text provided by the user. "
        "Always assume that the entire user message is the text to be translated. "
        f"Return *only* the final {target_lang} translation. "

        # --- REMOVED INSTRUCTION FOR BLANK LINES (was being ignored) ---

        "DO NOT add any explanations, questions, or comments. "
        "DO NOT add segment numbers. DO NOT add timestamps. "
        "DO NOT add any text that was not in the original segment. "
        "Your response must be *only* the translated text and nothing else."
    )
    human_template = "{text}"
    # --- END DYNAMIC PROMPT ---

    # Initialize the LLM with temperature and sampling settings:
    llm = OllamaLLM(
        model=model_name,
        temperature=temp,
        top_p=0.9,
        top_k=50,
        num_predict=2048
    )

    # Build the LangChain prompt pipeline:
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template),
    ])

    chain = chat_prompt | llm

    # Start timing:
    start_time = time.time()  # <--- This is a float
    print(f"Translating from {source_lang} to {target_lang} with {model_name} @ temp={temp}...")

    # --- SRT DETECTION LOGIC ---
    # Regex to find an SRT segment (number, time, text)
    # re.DOTALL makes '.' match newlines as well
    srt_regex = re.compile(
        r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\n|\Z)',
        re.DOTALL
    )

    matches = list(srt_regex.finditer(input_text))

    if matches:
        # SRT MODE: Translating segment by segment
        print(f"SRT format detected. Translating {len(matches)} segments...")
        translated_segments = []
        for match in matches:
            # Renaming variables to avoid conflict
            number, srt_start, srt_end, text_to_translate = match.groups()

            # We call the LLM only with the segment text
            translation = chain.invoke({"text": text_to_translate})
            # --- MODIFIED: We tell clean_segment this is an SRT segment ---
            clean_trans = clean_segment(translation, is_srt_segment=True)

            # We use the new variables
            translated_segments.append(
                f"{number}\n{srt_start} --> {srt_end}\n{clean_trans}"
            )

        # We join all segments with the SRT double newline
        final_translation = "\n\n".join(translated_segments)

    else:
        # --- NEW SIMPLE TEXT MODE (Block-by-Block) ---
        # We split by blank lines (2 or more newlines) to translate chunk by chunk.
        # This prevents the LLM from stopping at the first blank line.
        print("Simple text format detected. Translating block-by-block...")

        # Split by the blank lines, but *keep* the separators
        blocks = re.split(r'(\n{2,})', input_text)
        translated_text_parts = []

        for part in blocks:
            if re.match(r'^\n{2,}$', part):
                # This is a blank line separator, just add it back
                translated_text_parts.append(part)
            elif part.strip() == "":
                # This is an empty string (e.g., from start/end), ignore it
                continue
            else:
                # This is a block of text to translate
                translation = chain.invoke({"text": part})
                # We use the GENTLE clean (is_srt_segment=False) because this
                # block might have its own *single* newlines we want to keep.
                clean_trans = clean_segment(translation, is_srt_segment=False)
                translated_text_parts.append(clean_trans)

        final_translation = "".join(translated_text_parts)

    # --- END DETECTION LOGIC ---

    # Measure time elapsed:
    # Now 'start_time' is still the original float
    elapsed_time = time.time() - start_time
    print(f"Model {model_name} took {elapsed_time:.2f} seconds.")

    return final_translation


def clean_segment(text, is_srt_segment=False):
    '''
    Cleans some translated segments from unwanted spaces and line breaks.
    :param text: The text to clean
    :param is_srt_segment: Boolean, applies aggressive cleaning for SRT segments
    :return: Cleaned text
    '''
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. "Un-escape" literal newlines (e.g., "\\n" -> "\n")
    # This is safe for both modes.
    text = text.replace('\\n', '\n')

    # 2. Remove whitespace from the *absolute* beginning and end.
    # This is safe for both modes.
    text = text.strip()

    if is_srt_segment:
        # --- AGGRESSIVE CLEANING for SRT segments ---

        # 3a. NEW ANTI-HALLUCINATION LOGIC:
        # We assume the real translation ends at the first double newline.
        if "\n\n" in text:
            text = text.split("\n\n")[0]

        # 4a. Replace multiple newlines *within* the segment with a single space.
        # (A single SRT segment should not have hard line breaks)
        text = re.sub(r'\n{1,}', ' ', text)

        # 5a. Replace multiple spaces/tabs with a single one.
        text = re.sub(r'[ \t]+', ' ', text)

    else:
        # --- GENTLE CLEANING for Simple Text ---
        # We must preserve line breaks and blank lines.

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # For each line, replace multiple spaces with one, then strip.
            # This turns a "blank line with spaces" into "" (an empty string).
            # This preserves the line count.
            cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
            cleaned_lines.append(cleaned_line)

        # 4b. Re-join the lines. Blank lines are preserved.
        text = "\n".join(cleaned_lines)

    return text


if __name__ == "__main__":
    model = "gemma3:4b-it-qat"
    temp = 0.3
    source = "English"
    target = "Spanish"

    # Test 1: SRT Mode
    srt_text = '''
    1
    00:00:01,000 --> 00:00:03,000
    Hello and welcome to our channel.

    2
    00:00:03,500 --> 00:00:05,000
    Today we talk about FOREO.
    '''
    translation_srt = llm_translation(model, srt_text, temp, source, target)
    print(f"\n🔹{source} to {target} Translation (SRT Mode):\n{translation_srt}")

    # Test 2: Simple Text Mode (with blank lines)
    simple_text = "LUNA™ 4 plus\n\nNear-infrared, red LED cleansing & microcurrent.\n   For aging skin."
    translation_simple = llm_translation(model, simple_text, temp, source, target)
    print(f"\n🔹{source} to {target} Translation (Simple Mode):\n{translation_simple}")

    # Test 3: Clean_segment logic test
    print("\n🔹Testing clean_segment...")
    srt_hallucination = "  Translated text.  \n\n2\n00:00:05,000 --> ...  "
    simple_with_blanks = "  Line 1\n    \n  Line 3  "
    print(f"SRT Clean Test: '{clean_segment(srt_hallucination, is_srt_segment=True)}'")
    print(f"Simple Clean Test (preserves newlines): '{clean_segment(simple_with_blanks, is_srt_segment=False)}'")