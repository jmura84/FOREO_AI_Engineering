# Translation App with vLLM

This is a specialized version of the Translation App designed to run on **Google Kubernetes Engine (GKE)**, leveraging **vLLM** for high-performance inference of Large Language Models.

It orchestrates three components:
1.  **Translation App**: The main FastAPI/Gradio user interface.
2.  **vLLM Text Service**: Serves the `google/gemma-2-9b-it` model for translation.
3.  **vLLM OCR Service**: Serves the `deepseek-ai/deepseek-vl-1.3b-chat` model for optical character recognition.

## Prerequisites

Before running this application, ensure you have the following tools installed and configured:

1.  **Google Cloud Platform Project**: An active project with billing enabled.
2.  **Google Cloud SDK**: [Install gcloud CLI](https://cloud.google.com/sdk/docs/install).
3.  **Docker Desktop**: Required to build the images.
4.  **Kubectl**: The Kubernetes command-line tool.

## Deployment Instructions

### 1. Initial Setup & Authentication

Open a PowerShell terminal and run these commands to set up your environment (do this once):

```powershell
# 1. Login to Google Cloud
gcloud auth login

# 2. Configure Docker to use Google credentials
gcloud auth configure-docker

# 3. Install the GKE Auth Plugin (Critical for GKE access)
gcloud components install gke-gcloud-auth-plugin

# 4. Connect to your GKE Cluster
# Replace CLUSTER_NAME, ZONE, and PROJECT_ID with your actual values
gcloud container clusters get-credentials CLUSTER_NAME --zone ZONE --project PROJECT_ID
```

### 2. Create Required Secrets

You need a Hugging Face token to download the gated models (like Gemma).

```powershell
kubectl create secret generic hf-secret --from-literal=token=YOUR_HUGGING_FACE_TOKEN
```

### 3. Build and Push Docker Image

We use a custom PowerShell script to build the app image and push it to Google Container Registry.

```powershell
# Navigate to this directory
cd translation_app_vllm

# Run the build script
.\build_and_push.ps1 -ProjectId "YOUR_PROJECT_ID"
```
*Example: `.\build_and_push.ps1 -ProjectId "YOUR_PROJECT_ID"`*

### 4. Deploy to GKE

Deploy the backend services and the application using kubectl.

**Note**: The manifests in `k8s/` are configured for **GKE Autopilot** using **NVIDIA L4** GPUs.

```powershell
# 1. Deploy the Text Generation Service
kubectl apply -f k8s/01-vllm-text.yaml

# 2. Deploy the OCR Service
kubectl apply -f k8s/02-vllm-ocr.yaml

# 3. Deploy the Application
kubectl apply -f k8s/03-app.yaml
```

### 5. Accessing the App

It takes **5-10 minutes** for the pods to start because they need to download large model files (approx. 20GB+).

1.  **Check Status**:
    ```powershell
    kubectl get pods --watch
    ```
    Wait until `translation-app`, `vllm-text`, and `vllm-ocr` are all in `Running` state.

2.  **Get Authenticated URL**:
    ```powershell
    kubectl get svc translation-app-service
    ```
    Copy the `EXTERNAL-IP` address.

3.  **Run**:
    Open `http://<EXTERNAL-IP>` in your web browser.

## Using the App

The interface provides a familiar translation experience:

*   **Source/Target Language**: Select your desired languages.
*   **Input**: Type text directly or use the "OCR" tab to upload an image.
*   **Translate**: Click the button to get the result from the vLLM-powered Gemma model.