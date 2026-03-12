import os
import pandas as pd
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# --- CONFIGURATION ---
# Path relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Paths
PERSIST_DIRECTORY = os.path.join(project_root, "rag")
CSV_PATH = os.path.join(project_root, "data", "foreo_uniques_en-es_TM.csv")
COLLECTION_NAME = "foreo_translations_gemini"

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env. Cannot create embeddings.")
        return

    print("--- Initializing Gemini Embeddings ---")
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    print(f"--- Connecting to ChromaDB at: {PERSIST_DIRECTORY} ---")
    if not os.path.exists(PERSIST_DIRECTORY):
        try:
            os.makedirs(PERSIST_DIRECTORY)
        except Exception as e:
            print(f"⚠️ Could not create directory {PERSIST_DIRECTORY}: {e}")

    # Initialize Chroma
    # We will overwrite or append? For cleaner setup, let's assume we want to rebuild or using a new collection name.
    # The previous app used "foreo_translations", we use "foreo_translations_gemini" to avoid conflict/errors with old embeddings.
    
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
        persist_directory=PERSIST_DIRECTORY
    )

    # Load Data
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: CSV file not found at: {CSV_PATH}")
        return

    print(f"--- Loading data from: {CSV_PATH} ---")
    try:
        df = pd.read_csv(CSV_PATH)
        docs = []
        
        source_col = 'english'
        target_col = 'spanish'

        # Basic validation
        if source_col not in df.columns or target_col not in df.columns:
            print(f"❌ Error: CSV missing columns '{source_col}' or '{target_col}'")
            return

        print("Preparing documents...")
        for index, row in df.iterrows():
            source_text = str(row.get(source_col, '')).strip()
            target_text = str(row.get(target_col, '')).strip()

            if source_text and target_text:
                docs.append(
                    Document(
                        page_content=source_text,
                        metadata={
                            "target": target_text,
                            "row_id": index
                        }
                    )
                )
        
        print(f"✅ Prepared {len(docs)} documents.")

        if docs:
            print("--- Adding documents to Vector DB (this may take a while) ---")
            # Batching
            batch_size = 100
            total_batches = (len(docs) // batch_size) + 1
            
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                try:
                    vector_db.add_documents(batch)
                    print(f"   Processed batch {i // batch_size + 1}/{total_batches}")
                except Exception as e:
                    print(f"❌ Error processing batch {i // batch_size + 1}: {e}")
            
            print("✅ Vector database created successfully!")
            
            # Test
            print("--- Testing Retrieval ---")
            res = vector_db.similarity_search("facial cleansing", k=1)
            if res:
                print(f"Test Result: {res[0].page_content} -> {res[0].metadata.get('target')}")

        else:
            print("⚠️ No valid documents to add.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
