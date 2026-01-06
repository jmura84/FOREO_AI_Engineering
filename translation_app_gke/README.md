# Translation App (Gemini Edition)

This version of the Translation App has been refactored to use **Google Gemini 1.5 Flash** for both Translation and OCR, removing the need for heavy local GPUs and vLLM deployments. It also features a RAG (Retrieval-Augmented Generation) system for English-to-Spanish translations using a local ChromaDB with Gemini Embeddings.

## Features

-   **Multilingual Translation**: Supports English, Spanish, Polish, Turkish using `gemini-1.5-flash`.
-   **OCR**: Transcribe text from images using `gemini-1.5-flash` (Vision).
-   **Audio Transcription**: Uses `openai-whisper` (Local CPU) for audio/video transcription.
-   **RAG**: Context-aware translation for English -> Spanish, retrieving terminology from `data/foreo_uniques_en-es_TM.csv`.
-   **Lightweight**: No longer requires NVIDIA GPUs for the backend.

## Prerequisites

1.  **Google Cloud API Key**: You need a valid `GOOGLE_API_KEY` with access to:
    -   Generative Language API (Gemini).
2.  **Docker Desktop**: For building the image.
3.  **Kubectl**: For GKE deployment (optional).
4.  **FFmpeg**: Required locally if running without Docker (for Whisper).

## Local Setup

1.  **Clone & Install**:
    ```bash
    git clone <repo>
    cd translation_app_gke
    pip install -r requirements.txt
    ```

2.  **Environment Setup**:
    Create a `.env` file in the root directory:
    ```
    GOOGLE_API_KEY=AIzaSy...
    ```

3.  **Generate Vector DB (RAG)**:
    Before running the app, generate the local vector database:
    ```bash
    python scripts/create_vector_db.py
    ```
    *Note: This requires the `.env` file with a valid API Key.*

4.  **Run the App**:
    ```bash
    python main.py
    ```
    Open http://localhost:7860

## Deployment to GKE

### 1. Build and Push (Optional if using local only)
```powershell
.\build_and_push.ps1 -ProjectId "YOUR_PROJECT_ID"
```

### 2. Configure Secrets
Create a secret for your Google API Key:
```bash
kubectl create secret generic google-api-secret --from-literal=api-key=YOUR_ACTUAL_API_KEY
```
*(Note: You no longer need `hf-secret` unless you have other dependencies.)*

### 3. Deploy
Deploy the application. The vLLM services (`01-vllm-text.yaml`, `02-vllm-ocr.yaml`) are **DEPRECATED** and no longer needed.

```bash
kubectl apply -f k8s/03-app.yaml
```

## Changes from vLLM Version
-   **Removed**: Dependence on `vllm-text` and `vllm-ocr` services.
-   **Removed**: Local model selection (Ollama/DeepSeek).
-   **Added**: `modules/rag_manager.py` for Chroma/Gemini RAG.
-   **Added**: `modules/img_transcriptor.py` now uses Gemini Vision.
-   **Added**: `scripts/create_vector_db.py` for RAG setup.