---
title: Multimodal Translator
emoji: 🏃
colorFrom: red
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: Multimodal translator for FOREO
---

# Multimodal Translator for FOREO

A hybrid translation application designed to run on **Hugging Face Spaces**, combining API-based and local open-source models for English ↔ Spanish translation.

🔗 **Live Demo**: [huggingface.co/spaces/jmura84/Multimodal_translator](https://huggingface.co/spaces/jmura84/Multimodal_translator)

## Available Models

| Model | Type | Notes |
|-------|------|-------|
| `gemini-2.5-flash-lite` | API (Google) | Lowest latency — Highest rate limit |
| `gemini-2.5-flash` | API (Google) | Good quality — High rate limit |
| `gemini-2.5-pro` | API (Google) | Best quality — Lower rate limit |
| `google/gemma-2-2b-it` | Open-source (HF) | No rate limits — Runs locally |

## Features

- **Hybrid Engine** — Can run lightweight models locally (via Transformers) or connect to the Gemini API.
- **SRT (Subtitle) support** — Translate `.srt` subtitle files while preserving timestamps.
- **RAG integration** — Retrieval-Augmented Generation for improved translation quality.
- **Video transcription** — Transcribe video audio using Whisper.
- Optimized for CPU-tier Spaces (quantization where applicable) or API-based inference.

## Tech Stack

- **LLM**: Google Gemini API (2.5 family) / Hugging Face Transformers
- **UI**: Gradio SDK
- **Containerization**: Docker

## Configuration Reference

Check out the Spaces configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
