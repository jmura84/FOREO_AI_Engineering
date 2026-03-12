import os
import pandas as pd
import time
from langchain_chroma import Chroma
# --- MODIFICATION: Use Ollama for local embeddings ---
from langchain_ollama import OllamaEmbeddings
# --- END MODIFICATION ---
from langchain_core.documents import Document
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()

# We no longer need to check for GOOGLE_API_KEY because we'll use local Ollama

# Define dynamic paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Paths to data and DB
PERSIST_DIRECTORY = os.path.join(project_root, "rag")
CSV_PATH = os.path.join(project_root, "data", "foreo_uniques_en-es_TM.csv")


def main():
    print("--- Initializing Embeddings with Ollama (Local) ---")

    # 1. Create the Embeddings function using Ollama
    # Make sure you have run 'ollama pull nomic-embed-text'
    embeddings_model = OllamaEmbeddings(
        # https://ollama.com/library/nomic-embed-text
        # https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
        model="nomic-embed-text",
    )

    print(f"--- Connecting to ChromaDB at: {PERSIST_DIRECTORY} ---")

    if not os.path.exists(PERSIST_DIRECTORY):
        try:
            os.makedirs(PERSIST_DIRECTORY)
            print(f"Directory created: {PERSIST_DIRECTORY}")
        except Exception as e:
            print(f"⚠️ Could not create directory {PERSIST_DIRECTORY}: {e}")

    vector_db = Chroma(
        collection_name="foreo_translations",
        embedding_function=embeddings_model,
        persist_directory=PERSIST_DIRECTORY
    )

    # 3. Load data
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found at: {CSV_PATH}")
        return

    print(f"--- Loading data from: {CSV_PATH} ---")
    try:
        df = pd.read_csv(CSV_PATH)

        source_col = 'english'
        target_col = 'spanish'
        meta_col = 'endpoint'

        missing_cols = [col for col in [source_col, target_col] if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
            return

        docs = []
        print("Preparing documents...")

        for index, row in df.iterrows():
            source_text = str(row.get(source_col, '')).strip()
            target_text = str(row.get(target_col, '')).strip()
            endpoint_val = str(row.get(meta_col, '')).strip()

            if source_text:
                docs.append(
                    Document(
                        page_content=source_text,
                        metadata={
                            "target": target_text,
                            "endpoint": endpoint_val,
                            "row_id": index,
                            "source_file": "foreo_uniques_en-es_TM.csv"
                        }
                    )
                )

        print(f"✅ Prepared {len(docs)} documents for indexing.")

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 4. Add documents (Ollama is fast, big batches are fine)
    if docs:
        print("--- Adding documents and generating embeddings locally... ---")

        # With local Ollama we can be more aggressive
        batch_size = 100
        total_batches = (len(docs) // batch_size) + 1

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            try:
                vector_db.add_documents(batch)
                print(f"   Processed batch {i // batch_size + 1}/{total_batches} ({len(batch)} docs)")
                # No sleep needed with local Ollama
            except Exception as e:
                print(f"❌ Error processing batch {i // batch_size + 1}: {e}")

        print("✅ Vector database created successfully!")
    else:
        print("⚠️ No valid documents found in the CSV.")

    # 5. Quick search test
    test_query = "facial cleansing"
    print(f"\n--- Semantic search test for: '{test_query}' ---")

    # Chroma/LangChain handles search type automatically with Ollama
    results = vector_db.similarity_search(test_query, k=2)

    for i, res in enumerate(results):
        print(f"\nResult {i + 1}:")
        print(f"📄 Source (EN): {res.page_content}")
        print(f"🎯 Target (ES): {res.metadata.get('target')}")
        endpoint = res.metadata.get('endpoint')
        if endpoint:
            print(f"🔗 Endpoint: {endpoint}")


if __name__ == "__main__":
    main()