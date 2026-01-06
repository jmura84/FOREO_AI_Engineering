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

# ---------------------------------------------------------
# 3. Retrieve contextual translation pairs from DB
# ---------------------------------------------------------
def retrieve_context(segment, k=5):
    """
    Returns a list of (source, target) pairs from the vector DB.
    """
    results = vector_db.similarity_search(segment, k=k)
    context_pairs = []

    for d in results:
        src = d.page_content
        tgt = d.metadata.get("target", None)
        if tgt:
            context_pairs.append((src, tgt))

    return context_pairs


# ---------------------------------------------------------
# 4. Score best matching segment
# ---------------------------------------------------------
def best_match_score(segment, context_pairs):
    """
    Returns (score, best_target_translation).
    """
    best_score = 0.0
    best_target = None

    for src, tgt in context_pairs:
        score = SequenceMatcher(None, segment.lower(), src.lower()).ratio()
        if score > best_score:
            best_score = score
            best_target = tgt

    return best_score, best_target


# ---------------------------------------------------------
# 5. RAG-enhanced translation
# ---------------------------------------------------------
def rag_translate(segment, similarity_threshold=0.92):
    context_pairs = retrieve_context(segment, k=5)
    score, best_target = best_match_score(segment, context_pairs)

    # --- CASE A: Strong match => reuse translation ---
    if score >= similarity_threshold:
        return best_target

    # --- CASE B: Weaker match => inject context to shape style ---
    context_text = "\n".join(
        [f"- Source: {src}\n  Target: {tgt}" for src, tgt in context_pairs]
    )

    prompt = f"""
    You are a professional translator for the brand FOREO.
    Translate the text into Spanish, following the vocabulary, tone and wording style used in the examples.
    
    ### Style Examples
    {context_text}
    
    ### Task
    Translate the following text to Spanish.
    Follow the style and terminology of the examples.
    Do NOT add explanations.
    
    Text to translate:
    \"\"\"{segment}\"\"\"
    """

    # Correct call using invoke()
    return llm.invoke(prompt).strip()


# ---------------------------------------------------------
# --- 6. Run test ---
# ---------------------------------------------------------
if __name__ == "__main__":
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
