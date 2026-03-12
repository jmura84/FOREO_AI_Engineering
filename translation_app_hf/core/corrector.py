import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_translation_memory(csv_path, source_lang, target_lang):
    try:
        if not os.path.exists(csv_path):
            return {}

        df = pd.read_csv(csv_path)
        language_pair = f"{source_lang} -> {target_lang}"
        df_lang = df[df['language_pairs'] == language_pair].copy()

        if df_lang.empty:
            return {}

        df_lang['source_normalized'] = df_lang['source'].astype(str).str.lower().str.strip().str.rstrip('.,!;')
        glossary = pd.Series(df_lang.target.values, index=df_lang.source_normalized).to_dict()
        
        logger.info(f"Loaded {len(glossary)} corrections for {language_pair}.")
        return glossary

    except Exception as e:
        logger.error(f"Error loading translation memory: {e}")
        return {}

def normalize_text(text):
    return text.lower().strip().rstrip('.,!?;')

def review_and_correct(raw_translation, source_text, csv_path, source_lang, target_lang):
    glossary = load_translation_memory(csv_path, source_lang, target_lang)
    
    if not glossary:
        return raw_translation

    source_lines = source_text.split('\n')
    raw_lines = raw_translation.split('\n')

    if len(source_lines) != len(raw_lines):
        logger.warning(f"Line count mismatch ({len(source_lines)} vs {len(raw_lines)}). Skipping corrections.")
        return raw_translation

    corrected_lines = []
    for src_line, raw_line in zip(source_lines, raw_lines):
        if src_line.strip() == "":
            corrected_lines.append(raw_line)
            continue

        normalized_line = normalize_text(src_line)
        if normalized_line in glossary:
            corrected_lines.append(glossary[normalized_line])
        else:
            corrected_lines.append(raw_line)

    return "\n".join(corrected_lines)
