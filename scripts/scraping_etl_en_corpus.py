import re
import os
import time
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.parse import urlparse

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

def scrape_url(driver, url: str, delay: float = 2.0) -> list[str]:
    """Scrape a single URL and return extracted text."""
    try:
        driver.get(url)
        time.sleep(delay)
        html = driver.page_source
        lines = get_visible_text_from_html(html)
        return lines
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

if __name__ == "__main__":
    urls = [
        "https://www.foreo.com/peach-2-pro-max",
        "https://www.foreo.com/peach-collection",
        "https://www.foreo.com/peach-2-go",
        "https://www.foreo.com/peach-cooling-prep-gel",
        "https://www.foreo.com/espada-plus-collection",
        "https://www.foreo.com/espada-collection",
        "https://www.foreo.com/espada-blemish-gel",
        "https://www.foreo.com/acne-treatment?filter=1754",
        "https://www.foreo.com/kiwi-derma-collection",
        "https://www.foreo.com/shop/skincare",
        "https://www.foreo.com/men-collection",
        "https://www.foreo.com/shop-all"
    ]

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_folder = os.path.join(BASE_DIR, "data", "en_corpus")
    os.makedirs(output_folder, exist_ok=True)
    
    driver = webdriver.Chrome()
    
    all_data = []

    try:
        for url in urls:
            print(f"Scraping: {url} ...")
            lines = scrape_url(driver, url)
            endpoint = urlparse(url).path.strip("/").replace("/", "_") or "index"
            
            for line in lines:
                all_data.append({"english": line, "endpoint": endpoint})
                
    finally:
        driver.quit()

    if all_data:
        df = pd.DataFrame(all_data)
        
        # Drop duplicates
        initial_len = len(df)
        df.drop_duplicates(subset=["english"], inplace=True)
        print(f"Dropped {initial_len - len(df)} duplicates.")
        
        output_path = os.path.join(output_folder, "en_corpus.csv")
        df.to_csv(output_path, index=False)
        print(f"✅ Stored CSV: {output_path} ({len(df)} lines)")
    else:
        print("No data extracted.")
