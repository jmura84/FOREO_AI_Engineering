import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, rag_path="rag", collection_name="foreo_translations"):
        self.rag_path = rag_path
        self.collection_name = collection_name
        self.vector_db = None
        self.embeddings = None
        
        # Determine RAG path relative to project root if needed
        # Assuming run from root, rag folder is at ./rag
        if not os.path.exists(self.rag_path):
            # Try finding it relative to current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            potential_path = os.path.join(os.path.dirname(current_dir), rag_path)
            if os.path.exists(potential_path):
                self.rag_path = potential_path
            else:
                logger.warning(f"RAG path {self.rag_path} not found. RAG will not work.")

    def initialize(self):
        if self.vector_db is not None:
            return

        try:
            logger.info("Initializing RAG Engine with HuggingFace Embeddings...")
            # Use a CPU-friendly embedding model compatible with 'nomic-embed-text' dimensions (768)
            # nomic-ai/nomic-embed-text-v1.5 is a good standard choice
            model_name = "nomic-ai/nomic-embed-text-v1.5"
            model_kwargs = {'trust_remote_code': True, 'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            self.vector_db = Chroma(
                persist_directory=self.rag_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            logger.info(f"✓ RAG Engine initialized from {self.rag_path}")
        except Exception as e:
            logger.error(f"✗ Failed to initialize RAG Engine: {e}")
            self.vector_db = None

    def retrieve_context(self, query_text: str, source_lang: str = "English", k=5, similarity_threshold=0.92):
        if not self.vector_db:
            self.initialize()
            if not self.vector_db:
                return None, None 

        try:
            results = self.vector_db.similarity_search(query_text, k=k)
            context_pairs = []
            for d in results:
                src = d.page_content # Always English in current DB
                tgt = d.metadata.get("target", None) # Always Spanish in current DB
                if tgt:
                    context_pairs.append((src, tgt))
            
            # Check for strong match
            best_score = 0.0
            best_target = None
            
            for src_db, tgt_db in context_pairs:
                # If source is English, we match query against src_db (English)
                # If source is Spanish, we match query against tgt_db (Spanish)
                if source_lang == "English":
                    db_compare_text = src_db
                else: # Spanish
                    db_compare_text = tgt_db

                score = SequenceMatcher(None, query_text.lower(), db_compare_text.lower()).ratio()
                if score > best_score:
                    best_score = score
                    # If match found, we return the translation (the OTHER side)
                    if source_lang == "English":
                        best_target = tgt_db
                    else:
                        best_target = src_db
            
            if best_score >= similarity_threshold and best_target:
                return best_target, context_pairs # Direct match found
            
            return None, context_pairs
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return None, []

    def format_rag_prompt(self, query_text, context_pairs, source_lang="English", target_lang="Spanish"):
        if not context_pairs:
            return None
            
        # Format examples based on direction
        formatted_examples = []
        for src_db, tgt_db in context_pairs:
            if source_lang == "English":
                # Eng -> Spa : Source=Eng, Target=Spa
                formatted_examples.append(f"- Source: {src_db}\n  Target: {tgt_db}")
            else:
                # Spa -> Eng : Source=Spa, Target=Eng
                formatted_examples.append(f"- Source: {tgt_db}\n  Target: {src_db}") # Inverted
        
        context_text = "\n".join(formatted_examples)
        
        # Determine format instruction
        if target_lang == "Spanish":
            date_format = "dd/mm/yyyy"
            exclude_note = "Do NOT add any quotation marks or extra characters."
        else: # target English
            date_format = "mm/dd/yyyy"
            exclude_note = ""

        prompt = f"""
### Style and Context Examples
{context_text}

### Task
Translate the text below to {target_lang}.
Follow the style and terminology of the examples above.
Also convert datetime to the {target_lang} format ({date_format}).
Do not translate product names.
Do NOT add explanations, questions, or comments.
{exclude_note}
Return only the translation.

Text to translate:
'''{query_text}'''
"""
        return prompt

# Singleton
_rag_engine = None
def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
