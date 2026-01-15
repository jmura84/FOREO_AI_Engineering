# FOREO AI Engineering Project

This repository contains various AI engineering experiments, prototypes, and applications focused on **Translation**, **Transcription**, and **RAG (Retrieval Augmented Generation)** workflows for FOREO content.

## 📂 Applications

The project is divided into several application variants, each capable of running independently with different technology stacks:

### 1. **translation_app** (Local / Ollama)
*   **Location**: `translation_app/`
*   **Tech Stack**: [Ollama](https://ollama.com/), LangChain, Gradio.
*   **Purpose**: Designed for **local execution** using open-source models (like Gemma 2 and 3, DeepSeek OCR, Mistral, and so on) running via Ollama.
*   **Features**:
    *   SRT (Subtitle) support.
    *   RAG integration (English <-> Spanish).
    *   Video transcription using Whisper.
    *   **Best for**: Zero-cost local development and testing of open weights models.

### 2. **translation_app_gke** (Google Kubernetes Engine)
*   **Location**: `translation_app_gke/`
*   **Tech Stack**: Google Gemini API (`gemini-1.5-flash`), Docker, Kubernetes (GKE).
*   **Purpose**: Production-ready containerized application optimized for deployment on **Google Cloud Kubernetes Engine**.
*   **Features**:
    *   Uses **Gemini** models for high speed and long context windows.
    *   Includes `deploy_all.ps1` and Kubernetes manifests for easy cloud deployment.
    *   Robust error handling (retries for quota limits).
    *   **Best for**: Scalable, high-throughput translation tasks in the cloud.

### 3. **translation_app_hf** (Hugging Face Spaces)
*   **Location**: `translation_app_hf/`
*   **Tech Stack**: Hugging Face Transformers, Google Gemini API, Gradio SDK.
*   **Purpose**: A hybrid application designed to run on **Hugging Face Spaces**.
*   **URL**: https://huggingface.co/spaces/jmura84/Multimodal_translator
*   **Features**:
    *   **Hybrid Engine**: Can run lightweight models locally (via Transformers) OR connect to Gemini API.
    *   Optimized for CPU-tier spaces (using quantization where applicable) or API-based inference.
    *   **Best for**: Public demos and hosting on HF Spaces.

---

## 📓 Notebooks

Located in `notebooks/`, these files are used for quantitative evaluation and experiments:

*   **`llms_evaluation_rag_test.ipynb`**: Evaluates the performance improvement of RAG vs Standard translation using COMET/BLEU scores.
*   **`llms_evaluation_difftemps.ipynb`**: Compares LLM translation quality at different temperature settings.
*   **`llms_quality_estimation_rag_test.ipynb`**: Uses Quality Estimation (QE) models (like CometKiwi) to score translations without reference text.
*   **`llms_evaluation_biggercorpus.ipynb`**: Logic for running evaluations on larger datasets.

---

## 🛠️ Common Modules

*   **`rag/`**: Contains the shared RAG (Retrieval Augmented Generation) engine logic, dealing with ChromaDB and context retrieval.
*   **`data/`**: Stores corpora, translation memories, and evaluation datasets (SRT/Text files).

---
## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone http://foreogitlab.loclx.io/jmura84/foreo-ai-engineering-project.git
    cd foreo-ai-engineering-project
    ```

2.  **Choose your app**:
    *   Go to `translation_app` for local Ollama testing.
    *   Go to `translation_app_gke` for cloud deployment.

3.  **Install dependencies**:
    Each app has its own `requirements.txt`.
    ```bash
    cd translation_app
    pip install -r requirements.txt
    ```
