# 🤖 FOREO SLM Translator

> A multimodal and multilingual translator with built-in Translation Memory.

This project is an advanced, multimodal translation application built with Gradio and LangChain, designed to run locally using the power of Ollama models. It serves as a powerful interface for text, audio, and image translation, leveraging a sophisticated two-agent pipeline that learns from user corrections.

## ✨ Key Features

* **Multilingual Text Translation:** Supports translations between English, Spanish, Polish, and Turkish.

* **Sequential Translation Pipeline:** Uses a two-agent architecture for maximum accuracy:
    1.  **Agent 1 (NMT):** The `llm_call.py` module generates a raw translation using a selected Ollama LLM.
    2.  **Script (TM Corrector):** The `user_mods_corrector.py` module (a Python-based rules processor) reviews the raw translation against a user-generated CSV, and if the segment matches 100% it automatically applies the correction.

* **Evolving Translation Memory (TM):** Any manual corrections saved via the UI are stored in `data/user_mods_tm.csv`. This "memory" is used by the script in all future translations to prevent repeated errors.

* **Audio/Video Transcription:** Utilizes `openai-whisper` (via `audio2text.py`) to transcribe audio or video files (`.mp3`, `.wav`, `.mp4`) into `.SRT` formatted text, which is then populated in the source text box.

* **Image Transcription (OCR):** Employs a multimodal model (`gemma3:12b` via `img_transcriptor.py`) to extract text from uploaded images (`.png`, `.jpg`), placing the transcribed text into the source box.

## 🏛️ Architecture & Project Structure

The application is launched from `main.py`, which initializes the `gradio_ui.py` interface. The UI acts as an **Orchestrator**, dispatching tasks to the appropriate modules in the `modules/` directory.

```text
translation_app/
├── data/
│   ├── foreo_uniques_en-es_TM.csv
│   ├── merged_foreo_en-es.csv
│   └── user_mods_tm.csv        <-- 💾 Your Evolving Translation Memory!
├── media/
│   └── (Test media files)
├── modules/
│   ├── audio2text.py           (Whisper processor for Audio/Video)
│   ├── img_transcriptor.py     (OCR processor for Images)
│   ├── llm_call.py             (Agent 1: Raw NMT Translator)
│   ├── user_mods_corrector.py   (Agent 2: Python-based TM Corrector)
│── rag/
│   └── (ChromaDB Vector Store)
├── ui/
│   ├── gradio_ui.py            (All UI logic and orchestration)
├── .gitignore
├──__init__.py
├── main.py                     <-- 🚀 Run this file!
├── README.md
└── requirements.txt
```

## 🏁 Setup and Installation Guide

Follow these 3 steps to get the application running.

### Step 1: Install and Configure Ollama (Prerequisite)

This application is **entirely dependent on Ollama** to function.

1.  **Install Ollama:** Download and install the Ollama desktop application from [ollama.com](https://ollama.com/).

2.  **Run Ollama:** Always ensure the Ollama application is running in the background.

3.  **Pull Required Models:** This application requires several models to be downloaded before use. Open your terminal and run `ollama pull` for each model listed below. **The app will fail if the models are not pulled first.**

    ```bash
    # --- Translation LLMs (from the dropdown) ---
    ollama pull gemma2:9b
    ollama pull gemma3:12b
    ollama pull gemma3:4b
    ollama pull gemma3:4b-it-qat
    ollama pull thinkverse/towerinstruct:latest
    ollama pull deepseek-ocr:3b
    ```

You can skip the bigger Gemma 3 model (12b) if your hardware is not as powerful.
    
### Step 2: Install Python Dependencies

1.  **Create a Virtual Environment:** (Recommended)

    ```bash
    python -m venv .venv
    
2.  **Activate the Environment:**

    * On Windows: `.\.venv\Scripts\activate`
    * On macOS/Linux: `source .venv/bin/activate`

3.  **Install Requirements:** Install all necessary Python libraries (like Gradio, LangChain, Pandas, etc.) using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    
4.  **(Audio Dependency):** The audio transcriber (`audio2text.py`) requires **`ffmpeg`**. You must install this separately on your system.

    * *Windows (via Chocolatey):* `choco install ffmpeg`
    * *macOS (via Homebrew):* `brew install ffmpeg`

### Step 3: Run the Application

Once Ollama is running and your Python dependencies are installed, launch the app by running `main.py` from the project root (`translation_app/`):

```bash
python main.py
```

Open your browser and navigate to the local URL shown in your terminal (usually `http://127.0.0.1:7860`).

## 🚀 How to Use the App

* **Text Translation:**
    1.  Type your text into the "Source Text" box.
    2.  Always select the correct "Source Language" and "Target Language".
    3.  Choose the "Ollama Model" you wish to use.
    4.  Click "Translate ➡️".

* **Audio/Video Transcription (SRT):**
    1.  Click "Transcribe Audio/Video 🎵".
    2.  Select a "Whisper Model" (e.g., `base`, `small`).
    3.  Upload your `.mp3`, `.wav`, or `.mp4` file.
    4.  The transcribed `.SRT` text will appear in the "Source Text" box, ready to be translated.

* **Image Transcription (OCR):**
    1.  Click "Transcribe Image 🖼️".
    2.  Upload your `.png` or `.jpg` file.
    3.  The transcribed text (using the fixed `gemma3:12b` model) will appear in the "Source Text" box.

* **Save Corrections (Improve the TM):**<br><br>
    **WARNING:** DON'T add line breaks to the texts in the Source or Target language content since the correction is done based on the segment corresponding to the same line in both boxes.
    1.  After a translation, edit the text in the "Translated Text" box.
    2.  A "Save Modification" button will appear.
    3.  Click it. Your correction is now saved to `data/user_mods_tm.csv` and will be used automatically the next time you translate that segment.