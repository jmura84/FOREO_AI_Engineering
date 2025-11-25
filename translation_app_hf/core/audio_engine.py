import whisper
import os
import logging
import datetime

logger = logging.getLogger(__name__)

class AudioEngine:
    def __init__(self, model_name="base"):
        self.model_name = model_name
        self.model = None
        logger.info(f"AudioEngine initialized with model {self.model_name}")

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading Whisper model {self.model_name}...")
        try:
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise e

    def _format_timestamp(self, seconds: float) -> str:
        delta = datetime.timedelta(seconds=seconds)
        total_seconds = int(delta.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = delta.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    def transcribe(self, file_path: str) -> str:
        if self.model is None:
            self.load_model()

        if not os.path.exists(file_path):
            return "Error: File not found."

        try:
            logger.info(f"Transcribing {file_path}...")
            result = self.model.transcribe(file_path, verbose=False)
            
            srt_content = []
            for i, segment in enumerate(result["segments"]):
                start = self._format_timestamp(segment["start"])
                end = self._format_timestamp(segment["end"])
                text = segment["text"].strip()
                
                srt_content.append(str(i + 1))
                srt_content.append(f"{start} --> {end}")
                srt_content.append(text)
                srt_content.append("")
                
            return "\n".join(srt_content)

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return f"Error: {e}"

_audio_engine = None

def get_audio_engine(model_name="base"):
    global _audio_engine
    if _audio_engine is None:
        _audio_engine = AudioEngine(model_name)
    return _audio_engine
