from operator import itemgetter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 1. Configure embeddings and vector store (assuming the DB already exists)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_db = Chroma(
    persist_directory="./rag",          # Path to your existing vector DB
    embedding_function=embeddings,
    collection_name="foreo_translations"
)

retriever = vector_db.as_retriever(search_kwargs={"k": 1})  # Retrieve the most similar doc


# 2. Configure LLM
llm = OllamaLLM(model="gemma3:4b-it-qat")


# 3. RAG Prompt
rag_prompt_template = """
You are an expert translation proofreader.
Your task is to refine a translation based on a reference example retrieved from a translation memory.

Context (similar example from the database):
{context}

Source Text: {source_text}
Translation to Review: {raw_translation}

Instructions:
- If the context is highly relevant to the source text, use the terminology or style from the context to correct the translation.
- If the context is not relevant, keep the translation as it is or improve it slightly.
Return ONLY the final corrected translation.
"""

prompt = ChatPromptTemplate.from_template(rag_prompt_template)


# 4. Build the RAG chain
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


rag_chain = (
    {
        "context": itemgetter("source_text") | retriever | format_docs,
        "source_text": itemgetter("source_text"),
        "raw_translation": itemgetter("raw_translation"),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# 5. Example usage
if __name__ == "__main__":
    input_data = {
        "source_text": "facial cleansing",
        "raw_translation": "limpieza de cara",  # bad initial translation
    }

    corrected_text = rag_chain.invoke(input_data)

    print(f"Original: {input_data['source_text']}")
    print(f"Raw: {input_data['raw_translation']}")
    print(f"Corrected with RAG: {corrected_text}")
