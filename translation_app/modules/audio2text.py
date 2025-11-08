import whisper
import datetime
import os

# Make sure you have ffmpeg installed on your system.
# On Windows: choco install ffmpeg
# On Mac: brew install ffmpeg
# On Linux: sudo apt install ffmpeg

# Model loading (can be changed to "small", "medium", "large", etc.)
# We use a simple cache so we don't reload the model every time.
_model_cache = {}


def _load_whisper_model(model_name: str = "base"):
    """Loads a Whisper model into cache."""
    if model_name not in _model_cache:
        print(f"Loading Whisper model '{model_name}'... (This may take a moment)")
        try:
            _model_cache[model_name] = whisper.load_model(model_name)
            print(f"Model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            return None
    return _model_cache[model_name]


def _format_timestamp_srt(seconds: float) -> str:
    """
    Converts seconds (float) to the SRT timestamp format (HH:MM:SS,mmm).
    Example: 32.450 -> "00:00:32,450"
    """
    assert seconds >= 0, "Non-negative timestamp expected"

    # We use timedelta to handle the time conversion
    delta = datetime.timedelta(seconds=seconds)

    # Extract hours, minutes, seconds
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    # Extract milliseconds (from the microseconds part)
    millis = delta.microseconds // 1000

    # Format to HH:MM:SS,mmm
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def generate_srt_from_file(media_file_path: str, model_name: str = "base") -> str:
    """
    Transcribes an audio/video file and returns the content in SRT format.

    :param media_file_path: Path to the audio or video file.
    :param model_name: Name of the Whisper model (e.g., "base", "small", "medium").
    :return: A string with the complete content of the .srt file.
    """
    if not media_file_path or not os.path.exists(media_file_path):
        return "Error: Media file not found."

    model = _load_whisper_model(model_name)
    if model is None:
        return "Error: Could not load Whisper model."

    try:
        print(f"Starting transcription for: {media_file_path}")
        # verbose=False so it doesn't print the transcription to the console.
        result = model.transcribe(media_file_path, verbose=False)

        print("Transcription complete. Generating SRT content...")

        srt_content = []
        for i, segment in enumerate(result["segments"]):
            start_time = _format_timestamp_srt(segment["start"])
            end_time = _format_timestamp_srt(segment["end"])
            text = segment["text"].strip()

            # SRT Format:
            # 1
            # 00:00:01,000 --> 00:00:05,000
            # Subtitle text
            # (blank line)

            srt_content.append(str(i + 1))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")  # Blank line between segments

        print("SRT content generated.")
        return "\n".join(srt_content)

    except Exception as e:
        print(f"Error during transcription or SRT generation: {e}")
        return f"Error: {e}"


# --- Test block ---
# You can run this file directly to test it:
# python modules/srt_generator.py
if __name__ == "__main__":
    print("Testing SRT Generator Module...")

    # Simulate an audio file (create a fake one if it doesn't exist)
    # You will need 'ffmpeg' installed for this.
    test_file = "test_audio.mp3"
    if not os.path.exists(test_file):
        print("Creating a dummy test audio file (requires ffmpeg)...")
        try:
            # Create 5 seconds of silence
            os.system(f"ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 5 -q:a 9 -acodec libmp3lame {test_file}")
            print(f"Dummy file '{test_file}' created.")
        except Exception as e:
            print(f"Could not create dummy file. Install ffmpeg. Error: {e}")

    if os.path.exists(test_file):
        srt_data = generate_srt_from_file(test_file, model_name="base")
        print("\n--- GENERATED SRT (for dummy file) ---")
        print(srt_data)
        # Clean up the test file
        os.remove(test_file)
    else:
        print("Skipping test, dummy file could not be created.")