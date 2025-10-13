import re
import os
import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def get_visible_text_from_html(html: str) -> list[str]:
    """Extract visible text blocks (paragraphs, headers, list items) from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    blocks = soup.find_all(["p", "h1", "h2", "h3", "li"])
    lines = [b.get_text(" ", strip=True) for b in blocks if b.get_text(strip=True)]

    text = "\n".join(lines)
    text = re.sub(r"\s+(™|TM)\s*", r"\1 ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(Paso \d+( 1/2)?)\n", r"\1 ", text)

    remove_strings = [
        "English", "Deutsch", "Español", "Français", "Italiano", "Português", "Polski",
        "Svenska", "Русский", "Türkçe", "简体中文", "繁體中文", "MYSA"
    ]
    pattern = r"^(?:" + "|".join(remove_strings) + r")$"
    text = re.sub(pattern, "", text, flags=re.MULTILINE)
    text = "\n".join([line for line in text.splitlines() if line.strip()])
    return text.splitlines()

def scrape_website(url: str, html_folder: str, delay: float = 2.0) -> list[str]:
    """Scrape a single URL, save HTML, and return extracted text."""
    os.makedirs(html_folder, exist_ok=True)
    driver = webdriver.Chrome()

    try:
        driver.get(url)
        time.sleep(delay)
        html = driver.page_source

        # Guardar HTML
        endpoint = urlparse(url).path.strip("/").replace("/", "_") or "index"
        html_path = os.path.join(html_folder, f"{endpoint}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        lines = get_visible_text_from_html(html)
        return lines

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

    finally:
        driver.quit()

def scrape_and_save_pair(url_en: str, url_es: str, base_folder: str):
    """Scrape a pair of EN/ES URLs and save HTML, TXT, and CSV files."""
    endpoint = urlparse(url_en).path.strip("/").replace("/", "_") or "index"

    html_folder_en = os.path.join(base_folder, "html/en")
    html_folder_es = os.path.join(base_folder, "html/es")
    txt_folder = os.path.join(base_folder, "txt")
    csv_folder = os.path.join(base_folder, "csv")

    os.makedirs(txt_folder, exist_ok=True)
    os.makedirs(csv_folder, exist_ok=True)

    print(f"Scraping endpoint: {endpoint} ...")

    en_lines = scrape_website(url_en, html_folder_en)
    es_lines = scrape_website(url_es, html_folder_es)

    # Guardar TXT
    with open(os.path.join(txt_folder, f"{endpoint}_en.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(en_lines))
    with open(os.path.join(txt_folder, f"{endpoint}_es.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(es_lines))

    # Crear CSV alineado
    max_len = max(len(en_lines), len(es_lines))
    en_lines += [""] * (max_len - len(en_lines))
    es_lines += [""] * (max_len - len(es_lines))
    df = pd.DataFrame({"english": en_lines, "spanish": es_lines})
    df.to_csv(os.path.join(csv_folder, f"{endpoint}.csv"), index=False)
    print(f"✅ Guardado CSV: {endpoint}.csv ({len(df)} líneas)")

    return df

if __name__ == "__main__":
    urls_en = [
        "https://www.foreo.com/",
        "https://www.foreo.com/black-friday-preregistration",
        "https://www.foreo.com/faq-swiss-202?p=1224&s=1",
        "https://www.foreo.com/faq-swiss-502",
        "https://www.foreo.com/special-offers",
        "https://www.foreo.com/bestsellers",
        "https://www.foreo.com/red-light-therapy",
    ]

    urls_es = [u.replace("www.foreo.com", "www.foreo.com/es") for u in urls_en]
    base_folder = "../data/corpus"

    all_dfs = []
    for en, es in zip(urls_en, urls_es):
        df = scrape_and_save_pair(en, es, base_folder)
        df["endpoint"] = urlparse(en).path.strip("/") or "index"
        all_dfs.append(df)

    # CSV global fusionado
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(os.path.join(base_folder, "csv", "merged_foreo.csv"), index=False)
    print("✅ CSV global guardado: merged_foreo.csv")
