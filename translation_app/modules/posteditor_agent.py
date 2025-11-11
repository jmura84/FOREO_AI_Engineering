import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
import re


# --- Agent 2: Post-Editor ---

def load_translation_memory(csv_path, source_lang, target_lang):
    """
    Loads the user_mods.csv file and filters it for the current language pair.
    Returns a dictionary (glossary) of corrections.
    """
    try:
        if not os.path.exists(csv_path):
            print("Post-Editor: No 'user_mods.csv' found. Skipping corrections.")
            return {}

        df = pd.read_csv(csv_path)

        # 1. Create the language pair string
        language_pair = f"{source_lang} -> {target_lang}"

        # 2. Filter the DataFrame for the current language pair
        df_lang = df[df['language_pairs'] == language_pair].copy()

        if df_lang.empty:
            print(f"Post-Editor: No corrections found for {language_pair}.")
            return {}

        # 3. Create the normalized key for matching
        # (lowercase, strip whitespace, remove trailing punctuation)
        df_lang['source_normalized'] = df_lang['source'].astype(str) \
            .str.lower() \
            .str.strip() \
            .str.rstrip('.,!?;')  # Keep '?' as requested

        # 4. Create the glossary dictionary
        # Key: normalized_source, Value: target
        # We keep the *last* correction in case of duplicates
        glossary = pd.Series(df_lang.target.values, index=df_lang.source_normalized).to_dict()

        print(f"Post-Editor: Loaded {len(glossary)} corrections for {language_pair}.")
        return glossary

    except Exception as e:
        print(f"Post-Editor Error: Could not load or parse CSV '{csv_path}'. Error: {e}")
        return {}


def build_posteditor_chain(model_name, temp):
    """
    Creates the LangChain chain for the Post-Editor agent.
    This new prompt is procedural and much stricter.
    """

    system_prompt = (
        "You are a post-editor. You will receive a 'Source Text', its 'Raw Translation', and a 'Glossary' of corrections. "
        "You MUST follow these steps:\n"
        "1. Split the 'Source Text' into lines.\n"
        "2. Split the 'Raw Translation' into lines. (They should have the same number of lines).\n"
        "3. Create a 'Final Translation' (initially a copy of the 'Raw Translation').\n"
        "4. For each line in the 'Source Text':\n"
        "   a. Normalize the source line (lowercase, strip, remove trailing '.,!;').\n"
        "   b. Check if this normalized line exists as a key in the 'Glossary'.\n"
        "   c. If it EXISTS: You MUST replace the corresponding line in the 'Final Translation' with the 'target' value from the 'Glossary'.\n"
        "   d. If it does NOT exist: You MUST keep the line from the 'Raw Translation' as-is.\n"
        "5. Return the complete 'Final Translation' block, with all line breaks preserved.\n\n"
        "Example:\n"
        # --- FIX: Escaped the example JSON with double curly braces ---
        "Glossary: {{\"faq™ 202\": \"FAQ™ 202\"}}\n"
        # --- END FIX ---
        "Source Text:\n"
        "Line 1\n"
        "FAQ™ 202\n"
        "Line 3\n\n"
        "Raw Translation:\n"
        "Línea 1\n"
        "Preguntas frecuentes™ 202\n"
        "Línea 3\n"
        "\n\n"
        "--- START DATA ---\n"
        "GLOSSARY (Source -> Target Corrections):\n"
        "{glossary_json}\n\n"
        "SOURCE TEXT (Original):\n"
        "{source_text}\n\n"
        "RAW TRANSLATION (From Agent 1):\n"
        "{raw_translation}\n"
        "--- END DATA ---\n\n"
        "Final, Corrected Translation:"
    )

    prompt = ChatPromptTemplate.from_template(system_prompt)

    llm = OllamaLLM(
        model=model_name,
        temperature=temp,
        top_p=0.9,
        top_k=50,
        num_predict=4096  # Increased for safety with long inputs
    )

    chain = prompt | llm | StrOutputParser()
    return chain


def normalize_text_for_lookup(text):
    """
    Normalizes a single line of text for glossary lookup.
    """
    return text.lower().strip().rstrip('.,!?;')


def review_and_correct(raw_translation, source_text, csv_path, model_name, temp, source_lang, target_lang):
    """
    The main orchestration function for the Post-Editor agent.

    :param raw_translation: The output from llm_call.py (Agent 1)
    :param source_text: The original source text from the UI
    :param csv_path: Path to 'user_mods.csv'
    :param model_name: Name of the Ollama model
    :param temp: Temperature
    :param source_lang: Source language
    :param target_lang: Target language
    :return: The final, corrected translation
    """

    # 1. Load the glossary (Translation Memory)
    glossary = load_translation_memory(csv_path, source_lang, target_lang)

    if not glossary:
        # If no corrections exist for this language pair, just return the raw translation.
        print("Post-Editor: No corrections found. Skipping post-edit.")
        return raw_translation

    # 2. Check for simple, direct replacement (fast path)
    # This is much faster than calling an LLM if we don't need to.

    # We check if the *entire* source text (normalized) is a key in the glossary.
    # This only works for single-line or perfect paragraph matches.
    normalized_source = normalize_text_for_lookup(source_text)
    if normalized_source in glossary:
        print(f"Post-Editor: Found exact match replacement for '{source_text}'.")
        return glossary[normalized_source]

    # 3. If no exact match, and glossary is not empty, call the LLM to post-edit.
    # This handles complex cases (SRT, multi-line, partial replacements).

    try:
        print("Post-Editor: Calling LLM to review and apply corrections...")

        # Build the chain
        chain = build_posteditor_chain(model_name, temp)

        # Prepare inputs
        glossary_json = json.dumps(glossary, ensure_ascii=False, indent=2)

        # Run the chain
        final_translation = chain.invoke({
            "glossary_json": glossary_json,
            "source_text": source_text,
            "raw_translation": raw_translation
        })

        # Basic cleanup of the LLM's final output
        # Remove potential markdown fences or preamble
        final_translation = final_translation.replace("```", "").strip()

        print("Post-Editor: Review complete.")
        return final_translation

    except Exception as e:
        print(f"Post-Editor Error: LLM review failed. Returning raw translation. Error: {e}")
        # If the post-editor fails, it's safer to return the raw translation
        # than to return nothing.
        return raw_translation