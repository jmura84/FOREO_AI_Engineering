import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.rag_engine import get_rag_engine
from modules.llm_call import llm_translation

def verify_rag():
    print("--- Verifying RAG Engine ---")
    rag = get_rag_engine()
    rag.initialize()
    
    if rag.vector_db is None:
        print("FAIL: RAG vector DB not initialized. Check path.")
        return

    query = "LUNA 4 go" # A product likely in the DB
    print(f"Querying: {query}")
    
    # Test retrieval
    target, context_pairs = rag.retrieve_context(query)
    print(f"Direct match target: {target}")
    print(f"Context pairs found: {len(context_pairs)}")
    
    if len(context_pairs) > 0:
        print("\n--- Verifying Prompt Formatting ---")
        prompt = rag.format_rag_prompt(query, context_pairs)
        print("Generated RAG Context Prompt (excerpt):")
        print(prompt[:200] + "...")
        
        print("\n--- Verifying LLM Integration ---")
        # We won't actually call the LLM to save cost/time/setup, or checks if it runs.
        # But we can check if llm_translation accepts the argument.
        
        # We mock the LLM chain invoke if we want to be safe, but let's just see if the function runs 
        # and seemingly tries to call Ollama. 
        # If Ollama is not running, this might fail connection, which is fine, 
        # as long as the code path is exercised.
        print("Calling llm_translation with rag_context (Dry Run - checking for signature errors)...")
        # We expect connection error to Ollama if not running, or success if running.
        # We just want to ensure no TypeError on unexpected argument.
        try:
            llm_translation("dummy-model", "test text", 0.3, "English", "Spanish", rag_context=prompt)
        except Exception as e:
            # If it's a connection error or model not found, that's expected and means the code reached the LLM call.
            # If it's a TypeError (unexpected argument), that's a failure.
            if "unexpected keyword argument" in str(e):
                print(f"FAIL: llm_translation does not accept rag_context: {e}")
            else:
                print(f"SUCCESS: llm_translation accepted rag_context. (Stopped at: {type(e).__name__})")
                    
    else:
        print("WARNING: No context found for query. RAG might be empty or path wrong.")

if __name__ == "__main__":
    verify_rag()
