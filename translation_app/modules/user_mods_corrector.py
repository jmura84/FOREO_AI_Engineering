import pandas as pd
import os
import re


# --- Agent 2: Post-Editor (Now a Rules-Based Processor) ---

def load_translation_memory(csv_path, source_lang, target_lang):
    """
    Loads the user_mods.csv file and filters it for the current language pair.
    Returns a dictionary (glossary) of corrections.
    """
    try:
        if not os.path.exists(csv_path):
            print("Post-Editor: No 'user_mods_tm.csv' found. Skipping corrections.")
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
        df_lang['source_normalized'] = df_lang['source'].astype(str) \
            .str.lower() \
            .str.strip() \
            .str.rstrip('.,!;')

        # 4. Create the glossary dictionary
        glossary = pd.Series(df_lang.target.values, index=df_lang.source_normalized).to_dict()

        print(f"Post-Editor: Loaded {len(glossary)} corrections for {language_pair}.")
        return glossary

    except Exception as e:
        print(f"Post-Editor Error: Could not load or parse CSV '{csv_path}'. Error: {e}")
        return {}


def normalize_text_for_lookup(text):
    """
    Normalizes a single line of text for glossary lookup.
    """
    return text.lower().strip().rstrip('.,!?;')


def review_and_correct(raw_translation, source_text, csv_path, model_name, temp, source_lang, target_lang):
    """
    The main orchestration function for the Post-Editor.

    THIS IS NOW A PYTHON FUNCTION THAT OPERATES LINE-BY-LINE.

    This function compares the source text and raw translation line by line
    and applies corrections from the CSV (glossary).
    """

    # 1. Load the glossary (Translation Memory)
    glossary = load_translation_memory(csv_path, source_lang, target_lang)

    if not glossary:
        # If no corrections exist, return the raw translation immediately.
        print("Post-Editor: No corrections found. Skipping post-edit.")
        return raw_translation

    print("Post-Editor: Applying Python-based corrections (line-by-line)...")

    try:
        # --- NEW LOGIC: LINE-BY-LINE ---
        # Agent 1 (llm_call) already gives us a 1-to-1 translation
        source_lines = source_text.split('\n')
        raw_lines = raw_translation.split('\n')

        # Failsafe: If Agent 1 failed and they don't have the same length,
        # return the raw translation to avoid a crash.
        if len(source_lines) != len(raw_lines):
            print(
                f"Post-Editor Warning: Mismatch in line count. ({len(source_lines)} vs {len(raw_lines)}). Returning raw translation.")
            return raw_translation

        corrected_lines = []

        # 5. Iterate over each line and apply the glossary
        for src_line, raw_line in zip(source_lines, raw_lines):

            # Blank lines are respected automatically
            if src_line.strip() == "":
                corrected_lines.append(raw_line)  # (Should be "" anyway)
                continue

            normalized_line = normalize_text_for_lookup(src_line)

            if normalized_line in glossary:
                # Correction found! Use the glossary target.
                corrected_lines.append(glossary[normalized_line])
            else:
                # No correction, use the raw translation.
                corrected_lines.append(raw_line)

        # 6. Rebuild the full text
        final_translation = "\n".join(corrected_lines)

        print("Post-Editor: Python-based review complete.")
        return final_translation
        # --- END OF NEW LOGIC ---

    except Exception as e:
        print(f"Post-Editor Error: Python review failed. Returning raw translation. Error: {e}")
        return raw_translation