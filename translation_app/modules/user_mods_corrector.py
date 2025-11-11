import pandas as pd
import os
import re


def load_translation_memory(csv_path, source_lang, target_lang):
    """
    Loads the user_mods.csv file and filters it for the current language pair.
    Returns a dictionary (glossary) of corrections.
    """
    try:
        if not os.path.exists(csv_path):
            print("Corrector: No 'user_mods.csv' found. Skipping corrections.")
            return {}

        df = pd.read_csv(csv_path)

        # 1. Create the language pair string
        language_pair = f"{source_lang} -> {target_lang}"

        # 2. Filter the DataFrame for the current language pair
        df_lang = df[df['language_pairs'] == language_pair].copy()

        if df_lang.empty:
            print(f"Corrector: No corrections found for {language_pair}.")
            return {}

        # 3. Create the normalized key for matching
        df_lang['source_normalized'] = df_lang['source'].astype(str) \
            .str.lower() \
            .str.strip() \
            .str.rstrip('.,!?;')  # Keep '?' as requested

        # 4. Create the glossary dictionary
        glossary = pd.Series(df_lang.target.values, index=df_lang.source_normalized).to_dict()

        print(f"Corrector: Loaded {len(glossary)} corrections for {language_pair}.")
        return glossary

    except Exception as e:
        print(f"Corrector Error: Could not load or parse CSV '{csv_path}'. Error: {e}")
        return {}


def normalize_text_for_lookup(text):
    """
    Normalizes a single line of text for glossary lookup.
    """
    return text.lower().strip().rstrip('.,!;')


def review_and_correct(raw_translation, source_text, csv_path, model_name, temp, source_lang, target_lang):
    """
    The main orchestration function for the Corrector.

    This function compares the source text and raw translation line by line
    and applies corrections from the CSV (glossary).
    """

    # 1. Load the glossary (Translation Memory)
    glossary = load_translation_memory(csv_path, source_lang, target_lang)

    if not glossary:
        # If no corrections exist, return the raw translation immediately.
        print("Corrector: No corrections found. Skipping post-edit.")
        return raw_translation

    print("Corrector: Applying corrections...")

    try:
        # 2. Split source and raw translation into paragraphs
        #    (Agent 1 translates paragraph by paragraph)
        source_paragraphs = source_text.split('\n\n')
        raw_paragraphs = raw_translation.split('\n\n')

        # Check for structure mismatch (failsafe)
        if len(source_paragraphs) != len(raw_paragraphs):
            print(
                f"Corrector Warning: Mismatch in paragraph count. ({len(source_paragraphs)} vs {len(raw_paragraphs)}). Returning raw translation.")
            return raw_translation

        corrected_paragraphs = []

        # 3. Iterate over each paragraph
        for src_para, raw_para in zip(source_paragraphs, raw_paragraphs):

            # 4. Split paragraphs into lines
            source_lines = src_para.split('\n')
            raw_lines = raw_para.split('\n')

            # Failsafe for line mismatch inside a paragraph
            if len(source_lines) != len(raw_lines):
                corrected_paragraphs.append(raw_para)  # Append the raw paragraph
                print(
                    f"Corrector Warning: Mismatch in line count for paragraph '{src_para}'. Using raw translation for this block.")
                continue  # Skip to the next paragraph

            corrected_lines = []

            # 5. Iterate over each line and apply glossary
            for src_line, raw_line in zip(source_lines, raw_lines):

                normalized_line = normalize_text_for_lookup(src_line)

                if normalized_line in glossary:
                    # Correction found! Use the glossary target.
                    corrected_lines.append(glossary[normalized_line])
                else:
                    # No correction, use the raw translation.
                    corrected_lines.append(raw_line)

            # 6. Rebuild the corrected paragraph
            corrected_paragraphs.append("\n".join(corrected_lines))

        # 7. Rebuild the full text
        final_translation = "\n\n".join(corrected_paragraphs)

        print("Corrector: review complete.")
        return final_translation

    except Exception as e:
        print(f"Corrector Error: review failed. Returning raw translation. Error: {e}")
        return raw_translation