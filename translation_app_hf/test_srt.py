import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from core.llm_engine import get_llm_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_srt():
    print("Initializing engine...")
    try:
        engine = get_llm_engine()
        
        # Create sample file if it doesn't exist
        if not os.path.exists("test/sample_subtitle.srt"):
            with open("test/sample_subtitle.srt", "w", encoding="utf-8") as f:
                f.write("1\n00:00:00,000 --> 00:00:03,500\nHola chicas, mirad que he pasado lo que acabo de recibir\n\n2\n00:00:03,500 --> 00:00:06,340\nel Foreo Luna 4\n\n3\n00:00:06,340 --> 00:00:09,300\nel Foreo Bird 2\n\n4\n00:00:09,300 --> 00:00:13,520\nel Foreo Bird 2 para ojos y labios")

        with open("test/sample_subtitle.srt", "r", encoding="utf-8") as f:
            content = f.read()
            
        print("Translating SRT from Spanish to English...")
        result = engine.translate(content, "Spanish", "English")
        
        print("\n--- Result ---")
        print(result)
        print("--------------")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_srt()
