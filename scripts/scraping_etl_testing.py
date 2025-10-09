import re
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import os

def get_visible_text_and_html(url: str, html_filename: str) -> list[str]:
    '''
    Function to get visible text and html from a url
    :param url:
    :param html_filename:
    :return:
    '''
    driver = webdriver.Chrome()
    driver.get(url)
    html = driver.page_source

    # Stores all raw HTMLs:
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html)

    driver.quit()

    soup = BeautifulSoup(html, "html.parser")

    # Extracts texts from paragraphs and headers:
    blocks = soup.find_all(["p", "h1", "h2", "h3", "li"])
    lines = []
    for b in blocks:
        txt = b.get_text(" ", strip=True)  # Space instead of line breaks
        if txt:
            lines.append(txt)

    # Join with line breaks:
    text = "\n".join(lines)

    # Corrects the specific cases where the TM or ™ separates the string into two:
    text = re.sub(r"\s+(™|TM)\s*", r"\1 ", text)

    # Corrects double spacing:
    text = re.sub(r"\s{2,}", " ", text)

    # It corrects multiple line breaks from the Spanish file:
    text = re.sub(r"(Paso \d+( 1/2)?)\n", r"\1 ", text)

    # Lists of non-translated strings to delete from the dataset:
    remove_strings = [
        "English", "Deutsch", "Español", "Français", "Italiano", "Português", "Polski",
        "Svenska", "Русский", "Türkçe", "简体中文", "繁體中文", "MYSA"
    ]

    # Generating the pattern to find those strings:
    pattern = r"^(?:" + "|".join(remove_strings) + r")$"

    # Replacing them with an empty string:
    text = re.sub(pattern, "", text, flags=re.MULTILINE)

    # Deleting the extra lines:
    text = "\n".join([line for line in text.splitlines() if line.strip()])

    # Separating all lines:
    return text.splitlines()

def scrape_to_txt_and_df(url_en: str, url_es: str):
    '''
    Function to scrape text and html from an HTML file
    :param url_en:
    :param url_es:
    :return:
    '''
    # Making folders to store HTMLs:
    os.makedirs("../data/html", exist_ok=True)

    en_lines = get_visible_text_and_html(url_en, os.path.join("../data/html", "english_raw.html"))
    es_lines = get_visible_text_and_html(url_es, os.path.join("../data/html", "spanish_raw.html"))

    # Storing all the segments from each language in a TXT file:
    with open("../data/english_testing.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(en_lines))
    with open("../data/spanish_testing.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(es_lines))

    # Generating a returning a DataFrame:
    max_len = max(len(en_lines), len(es_lines))
    en_lines += [""] * (max_len - len(en_lines))
    es_lines += [""] * (max_len - len(es_lines))
    df = pd.DataFrame({"english": en_lines, "spanish": es_lines})
    return df

if __name__ == "__main__":
    foreo_url_en = "https://www.foreo.com/"
    foreo_url_es = "https://www.foreo.com/es"

    df = scrape_to_txt_and_df(foreo_url_en, foreo_url_es)
    df.to_csv("../data/testing_segments.csv", index=False)
