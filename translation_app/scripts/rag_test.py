from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- Vector DB ---
# Initialize embeddings using Ollama local model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to existing Chroma vector database
vector_db = Chroma(
    persist_directory="../rag",  # Correct directory
    embedding_function=embeddings,
    collection_name="foreo_translations"
)

# --- Function to search for exact translation ---
def get_translation_exact(source_text):
    # Perform similarity search in the vector DB
    results = vector_db.similarity_search(source_text, k=10)

    for doc in results:
        # Each doc is a Document object; translation is in metadata
        if "target" in doc.metadata and doc.page_content.strip() == source_text.strip():
            return doc.metadata["target"]

    return None

# --- List of segments to test ---
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

# --- Test each segment ---
if __name__ == "__main__":
    for source in test_segments:
        translation = get_translation_exact(source)

        if translation:
            print("Original:", source)
            print("Translation from DB:", translation)
            print("---")
        else:
            print("Original:", source)
            print("No matching translation found in the vector DB.")
            print("---")
