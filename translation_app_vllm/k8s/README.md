# Deployment Instructions for GKE

This folder contains the Kubernetes manifests to deploy the Translation App and its vLLM dependencies.

## Prerequisites

1.  **GKE Cluster with GPU Nodes**: Ensure your cluster has nodes with NVIDIA GPUs (e.g., L4, T4, A100) and drivers installed.
2.  **Hugging Face Token**: You need a token to access gated models (like Gemma).

## Steps

### 1. Create Secrets
Create a secret for headers/tokens if needed.
```bash
kubectl create secret generic hf-secret --from-literal=token=YOUR_HF_TOKEN
```

### 2. Deploy vLLM Services
Deploy the text and OCR model servers.
```bash
kubectl apply -f 01-vllm-text.yaml
kubectl apply -f 02-vllm-ocr.yaml
```
Wait for them to be ready.
```bash
kubectl get pods
```

### 3. Build and Deploy the App
Build your Docker image:
```bash
docker build -t gcr.io/YOUR_PROJECT/translation-app-vllm:latest ..
docker push gcr.io/YOUR_PROJECT/translation-app-vllm:latest
```

Update `03-app.yaml` with your image name if different.

Deploy the app:
```bash
kubectl apply -f 03-app.yaml
```

### 4. Access the App
Get the external IP:
```bash
kubectl get svc translation-app-service
```
Open `http://EXTERNAL-IP` in your browser.

## Configuration

- **Models**: To change models, update the `--model` arg in `01-vllm-text.yaml` or `02-vllm-ocr.yaml`.
- **Aliases**: The `--served-model-name` argument ensures the vLLM server responds to the model names expected by the UI (`gemma2:9b`, `deepseek-ocr:3b`).
