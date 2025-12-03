from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Vector DB ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    persist_directory="../rag",  # Correct path
    embedding_function=embeddings,
    collection_name="foreo_translations"
)

# --- Function to get translation from the vector DB ---
def get_translation_from_db(source_text):
    results = vector_db.similarity_search(source_text, k=5)  # retrieve closest matches

    for doc in results:
        # The actual translation is stored in metadata["target"]
        if doc.page_content.strip() == source_text.strip():
            return doc.metadata.get("target")
    return None

# --- Test segments ---
test_segments = [
    "Get your FOREO bestseller now",
    "Life is beautiful. But, with FOREO, it can also be extraordinary. Pamper yourself like never before with best-selling FOREO skincare and oral care devices, and indulge in moments of relaxation, rejuvenation and self-discovery. Over 20 million happy skincare lovers worldwide.",
    "Featured",
    "Featured",
    "Popularity",
    "Highest discount",
    "Highest price",
    "Lowest price",
    "FAQ™ 202"
]

# --- Run test ---
if __name__ == "__main__":
    for source in test_segments:
        raw_translation = "PLACEHOLDER"  # can be replaced by LLM later if needed
        corrected_translation = get_translation_from_db(source)

        print("Source:", source)
        print("Raw Translation:", raw_translation)
        print("DB Translation:", corrected_translation if corrected_translation else "No matching translation found")
        print("---")
