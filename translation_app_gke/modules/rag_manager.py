import os
import logging
from difflib import SequenceMatcher
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# RAG Configuration
PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag")
COLLECTION_NAME = "foreo_translations_gemini"

# Ensure API Key is available for Embeddings
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.warning("GOOGLE_API_KEY not found. RAG will not work.")

# Initialize Embeddings (Singleton-ish pattern by module level)
def get_embeddings():
    if not api_key:
        return None
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

def get_vector_db():
    embeddings = get_embeddings()
    if not embeddings:
        return None
    
    if not os.path.exists(PERSIST_DIRECTORY):
        return None

    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def retrieve_context(segment, k=5):
    """
    Returns a list of (source, target) pairs from the vector DB.
    """
    vector_db = get_vector_db()
    if not vector_db:
        return []

    try:
        results = vector_db.similarity_search(segment, k=k)
    except Exception as e:
        logger.error(f"RAG Retrieval error: {e}")
        return []

    context_pairs = []
    for d in results:
        src = d.page_content
        tgt = d.metadata.get("target", None)
        if tgt:
            context_pairs.append((src, tgt))
    
    return context_pairs

def best_match_score(segment, context_pairs):
    """
    Returns (score, best_target_translation).
    Checks fuzzy string matching ratio.
    """
    best_score = 0.0
    best_target = None

    for src, tgt in context_pairs:
        # Simple ratio
        score = SequenceMatcher(None, segment.lower(), src.lower()).ratio()
        if score > best_score:
            best_score = score
            best_target = tgt

    return best_score, best_target

def get_rag_context_string(segment, k=5):
    """
    Retrieves context and returns a formatted string for the LLM prompt.
    Returns: (context_string, best_match_translation_if_very_high_score)
    """
    context_pairs = retrieve_context(segment, k=k)
    if not context_pairs:
        return "", None

    score, best_target = best_match_score(segment, context_pairs)

    # If extremely high match, we might want to suggest using it directly
    # The original script used 0.92 threshold to return it directly.
    # We will return it as a candidate for the caller to decide, 
    # but here we mainly construct the context string.
    
    context_text = "\n".join(
        [f"- Source: {src}\n  Target: {tgt}" for src, tgt in context_pairs]
    )
    
    return context_text, best_target if score >= 0.95 else None
