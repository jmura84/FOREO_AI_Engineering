import os
import pandas as pd

# THIS IS AN AD HOC SCRIPT TO ALIGN THE FOREO_MERGED.CSV FILE PROPERLY AFTER MODIFYING THE TXT FILES SEGMENTS THAT WERE BROKEN OR THAT DID NOT HAVE A MATCH IN BOTH LANGUAGES, THUS CAUSING A MISALIGNMENT IN SOME OF THE URLS CONTENT, AND POTENTIALLY CAUSING A LOWER SCORE IN THE EVALUATIONS.

# Paths
TXT_FOLDER = "../data/corpus/txt"
OUTPUT_CSV = "../data/corpus/csv/merged_foreo.csv"

def read_txt(path):
    """Read text file preserving line breaks."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def merge_foreo_texts(txt_folder):
    files = [f for f in os.listdir(txt_folder) if f.endswith(".txt")]

    data = []

    # We filter endpoints without suffixes:
    endpoints = sorted(set(f[:-7] for f in files if f.endswith("_en.txt")) |
                       set(f[:-7] for f in files if f.endswith("_es.txt")))

    for endpoint in endpoints:
        en_path = os.path.join(txt_folder, f"{endpoint}_en.txt")
        es_path = os.path.join(txt_folder, f"{endpoint}_es.txt")

        en_lines = read_txt(en_path) if os.path.exists(en_path) else []
        es_lines = read_txt(es_path) if os.path.exists(es_path) else []

        max_len = max(len(en_lines), len(es_lines))
        en_lines += [""] * (max_len - len(en_lines))
        es_lines += [""] * (max_len - len(es_lines))

        for en, es in zip(en_lines, es_lines):
            data.append({"english": en, "spanish": es, "endpoint": endpoint})

        print(f"✅ {endpoint}: {len(en_lines)} lines merged")

    return pd.DataFrame(data)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    df = merge_foreo_texts(TXT_FOLDER)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print(f"\n✅ Merged CSV saved at: {OUTPUT_CSV}")
    print(f"Total rows: {len(df)}")
