🤖 FOREO SLM TranslatorA multimodal and multilingual translator with built-in Translation Memory.This project is an advanced, multimodal translation application built with Gradio and LangChain, designed to run locally using the power of Ollama models. It serves as a powerful interface for text, audio, and image translation, leveraging a sophisticated two-agent pipeline that learns from user corrections.✨ Key FeaturesMultilingual Text Translation: Supports translations between English, Spanish, Polish, and Turkish.Sequential Translation Pipeline: Uses a two-agent architecture for maximum accuracy:Agent 1 (NMT): The llm_call.py module generates a raw translation using a selected Ollama LLM.Agent 2 (TM Corrector): The user_mods_corrector.py module (a Python-based rules processor) reviews the raw translation against a user-generated CSV, automatically applying corrections.Evolving Translation Memory (TM): The app learns! Any manual corrections saved via the UI are stored in data/user_mods_tm.csv. This "memory" is used by Agent 2 in all future translations to prevent repeated errors.Audio/Video Transcription: Utilizes openai-whisper (via audio2text.py) to transcribe audio or video files (.mp3, .wav, .mp4) into .SRT formatted text, which is then populated in the source text box.Image Transcription (OCR): Employs a multimodal model (gemma3:12b via img_transcriptor.py) to extract text from uploaded images (.png, .jpg), placing the transcribed text into the source box.🏛️ Architecture & Project StructureThe application is launched from main.py, which initializes the gradio_ui.py interface. The UI acts as an Orchestrator, dispatching tasks to the appropriate modules in the modules/ directory.translation_app/
├── data/
│   ├── ... (other CSVs)
│   └── user_mods_tm.csv        <-- 💾 Your Evolving Translation Memory!
├── media/
│   └── (Test media files)
├── modules/
│   ├── audio2text.py           (Whisper processor for Audio/Video)
│   ├── img_transcriptor.py     (OCR processor for Images)
│   ├── llm_call.py             (Agent 1: Raw NMT Translator)
│   ├── user_mods_corrector.py   (Agent 2: Python-based TM Corrector)
│   ├── __init__.py
│   └── rag/
├── ui/
│   ├── gradio_ui.py            (All UI logic and orchestration)
│   ├── __init__.py
├── .gitignore
├── main.py                     <-- 🚀 Run this file!
├── README.md
└── requirements.txt
🏁 Setup and Installation GuideFollow these 3 steps to get the application running.Step 1: Install and Configure Ollama (Prerequisite)This application is entirely dependent on Ollama to function.Install Ollama: Download and install the Ollama desktop application from ollama.com.Run Ollama: Ensure the Ollama application is running in the background.Pull Required Models: This application requires several models to be downloaded before use. Open your terminal and run ollama pull for each model listed below. The app will fail if the models are not pulled first.# --- Translation LLMs (from the dropdown) ---
ollama pull gemma2:9b
ollama pull gemma3:12b
ollama pull gemma3:4b
ollama pull gemma3:4b-it-qat
ollama pull thinkverse/towerinstruct:latest

# --- Image Transcription (OCR) LLM ---
# (gemma3:12b is also used for this,
# so no extra download is needed if pulled above)
Step 2: Install Python DependenciesCreate a Virtual Environment: (Recommended)python -m venv .venv
Activate the Environment:On Windows: .\.venv\Scripts\activateOn macOS/Linux: source .venv/bin/activateInstall Requirements: Install all necessary Python libraries (like Gradio, LangChain, Pandas, etc.) using the requirements.txt file.pip install -r requirements.txt
(Audio Dependency): The audio transcriber (audio2text.py) requires ffmpeg. You must install this separately on your system.Windows (via Chocolatey): choco install ffmpegmacOS (via Homebrew): brew install ffmpegStep 3: Run the ApplicationOnce Ollama is running and your Python dependencies are installed, launch the app by running main.py from the project root (translation_app/):python main.py
Open your browser and navigate to the local URL shown in your terminal (usually http://127.0.0.1:7860).🚀 How to Use the AppText Translation:Type your text into the "Source Text" box.Select the "Source Language" and "Target Language".Choose the "Ollama Model" you wish to use.Click "Translate ➡️".Audio/Video Transcription (SRT):Click "Transcribe Audio/Video 🎵".Select a "Whisper Model" (e.g., base, small).Upload your .mp3, .wav, or .mp4 file.The transcribed .SRT text will appear in the "Source Text" box, ready to be translated.Image Transcription (OCR):Click "Transcribe Image 🖼️".Upload your .png or .jpg file.The transcribed text (using the fixed gemma3:12b model) will appear in the "Source Text" box.Save Corrections (Improve the TM):After a translation, edit the text in the "Translated Text" box.A red "Save Modification" button will appear.Click it. Your correction is now saved to data/user_mods_tm.csv and will be used automatically the next time you translate that segment.