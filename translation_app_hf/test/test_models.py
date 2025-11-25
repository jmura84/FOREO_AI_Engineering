import sys
import os
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_engine import get_llm_engine
from core.audio_engine import get_audio_engine
# Vision engine requires an image, so we'll skip it for this quick test or mock it if needed

logging.basicConfig(level=logging.INFO)

def test_llm():
    print("\n--- Testing LLM Engine ---")
    try:
        engine = get_llm_engine()
        # Just load it first
        engine.load_model()
        print("LLM Model loaded successfully.")
        
        # Simple translation test
        text = "Hello, how are you?"
        translation = engine.translate(text, "English", "Spanish")
        print(f"Translation: {translation}")
    except Exception as e:
        print(f"LLM Test Failed: {e}")

def test_audio():
    print("\n--- Testing Audio Engine ---")
    try:
        engine = get_audio_engine()
        engine.load_model()
        print("Audio Model loaded successfully.")
    except Exception as e:
        print(f"Audio Test Failed: {e}")

if __name__ == "__main__":
    test_audio()
    test_llm()
