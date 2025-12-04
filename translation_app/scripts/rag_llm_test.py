from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from difflib import SequenceMatcher

# --- 1. Vector DB ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(
    persist_directory="../rag",
    embedding_function=embeddings,
    collection_name="foreo_translations"
)

# --- 2. LLM ---
llm = OllamaLLM(model="gemma3:4b-it-qat", temperature=0.2)

# --- 3. Segments to test (cleaned, no duplicates) ---
segments = [
    "Get your FOREO bestseller now",
    "Life is beautiful. But, with FOREO, it can also be extraordinary. Pamper yourself like never before with best-selling FOREO skincare and oral care devices, and indulge in moments of relaxation, rejuvenation and self-discovery. Over 20 million happy skincare lovers worldwide.",
    "Featured",
    "Popularity",
    "Highest discount",
    "Highest price",
    "Lowest price",
    "We are having FOREO's BEAR™ lowest price tomorrow!",
    "Get ready to use your FAQ™ 202 device for a proper skincare!",
    "FAQ™ 202"
]

# --- 4. Helper: find closest DB translation ---
def get_best_db_translation(segment):
    results = vector_db.similarity_search(segment, k=5) # 1 = Closest to the most similar document, 10 = Closest to 10 similar documents
    if not results:
        return None, 0.0
    # Use difflib to find the best matching page_content
    best_score = 0.0
    best_translation = None
    for doc in results:
        sim = SequenceMatcher(None, segment.lower(), doc.page_content.lower()).ratio()
        if sim > best_score:
            best_score = sim
            best_translation = doc.metadata.get("target")
    return best_translation, best_score

# --- 5. RAG translation ---
def rag_translate(segment, similarity_threshold=0.7):
    db_translation, score = get_best_db_translation(segment)
    if db_translation and score >= similarity_threshold:
        # If partial match, could refine remaining text with LLM
        return db_translation
    # Otherwise fallback to raw LLM translation
    prompt = f"Translate this text to Spanish and don't add explanations:\n{segment}"
    return llm.generate([prompt]).generations[0][0].text.strip()

# --- 6. Run test ---
if __name__ == "__main__":
    for segment in segments:
        # Step 1: Raw LLM translation
        raw_prompt = f"Translate this text to Spanish and don't add explanations:\n{segment}"
        raw_translation = llm.generate([raw_prompt]).generations[0][0].text.strip()

        # Step 2: RAG-corrected translation
        rag_corrected = rag_translate(segment)

        # Step 3: Print results
        print("\n---")
        print("Source:", segment)
        print("Raw Translation:", raw_translation)
        print("RAG Corrected:", rag_corrected)
