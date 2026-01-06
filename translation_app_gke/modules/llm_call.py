import google.generativeai as genai
import os
import re
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DEFAULT_MODEL_NAME = "gemini-1.5-flash"

# Global cache for models to avoid re-initialization overhead
_model_cache = {}

def configure_gemini():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in environment variables.")
        raise ValueError("GOOGLE_API_KEY is required for Gemini models.")
    genai.configure(api_key=api_key)

configure_gemini()

def get_cached_model(model_name: str, temperature: float, top_p: float = 0.9, top_k: int = 40):
    """
    Retrieves a configured GenerativeModel from cache or creates a new one.
    """
    cache_key = (model_name, temperature, top_p, top_k)
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )
    
    _model_cache[cache_key] = model
    return model

def llm_translation(model_name: str, input_text: str, temp: float, source_lang: str, target_lang: str, context_text: str = None) -> str:
    """
    Translates the given text using Google Gemini models via direct API (no LangChain).
    Includes retry logic for RESOURCE_EXHAUSTED errors.
    """
    
    # Fallback/Default if model name is generic or missing
    if "gemini" not in model_name.lower():
        model_name = DEFAULT_MODEL_NAME
        
    logger.info(f"Initializing Gemini Translation with model: {model_name}, Temp: {temp}")

    # Get model instance
    model = get_cached_model(model_name, temp)

    # --- PROMPT CONSTRUCTION ---
    system_prompt = (
        f"You are a professional translator. "
        f"Your task is to translate *only* the user's text from {source_lang} to {target_lang}. "
        f"If the target language is Spanish, use Spanish from Spain dialect. "
        "The texts you translate are from a beauty products brand called FOREO.\n"
    )

    if context_text:
        system_prompt += (
            "\n### TERMINOLOGY & STYLE GUIDE ###\n"
            "Use the following examples as a reference for style, tone, and specific terminology:\n"
            f"{context_text}\n"
            "### END GUIDE ###\n\n"
        )

    system_prompt += (
        "Translate the exact text provided by the user. "
        "Always assume that the entire user message is the text to be translated. "
        f"Return *only* the final {target_lang} translation. "
        "DO NOT add any explanations, questions, or comments. "
        "DO NOT add segment numbers (unless they are part of SRT structure). "
        "DO NOT add any text that was not in the original segment. "
        "Your response must be *only* the translated text and nothing else."
    )
    
    # Check for SRT structure
    is_srt = bool(re.search(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', input_text))
    if is_srt:
         system_prompt += "\nKEEP the SRT format (numbers and timestamps) EXACTLY as they are. Translate only the content lines."

    # Combine into final prompt for generate_content
    full_prompt = f"{system_prompt}\n\nContent:\n{input_text}"

    # Start timing
    start_time = time.time()
    logger.info(f"Translating from {source_lang} to {target_lang}...")

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = model.generate_content(full_prompt)
            
            # Extract text
            final_translation = response.text.strip()
            
            # Clean up Markdown code blocks
            if final_translation.startswith("```"):
                 final_translation = re.sub(r"^```(?:json|text|srt)?\n|```$", "", final_translation, flags=re.MULTILINE).strip()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Translation successful on attempt {attempt+1}. Took {elapsed_time:.2f}s.")
            return final_translation

        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt) # Exponential backoff: 2, 4, 8, 16...
                    logger.warning(f"Quota exceeded (429). Retrying in {sleep_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error(f"Max retries reached for 429 error: {error_str}")
                    return f"Error: Quota exceeded after {max_retries} attempts."
            else:
                logger.error(f"Error during translation: {e}")
                return f"Error: {error_str}"

    return "Error: Unknown failure"

# Keep for compatibility
def clean_segment(text, model_name=None, is_srt_segment=False):
    if not text: return ""
    return text.strip()

if __name__ == "__main__":
    # Test
    res = llm_translation(DEFAULT_MODEL_NAME, "Hello world", 0.3, "English", "Spanish")
    print(res)