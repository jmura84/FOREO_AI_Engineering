import os
import time
import base64
import mimetypes
import logging
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default to Flash which supports multimodal input
DEFAULT_VISION_MODEL = "gemini-1.5-flash"

def get_image_base64(image_path: str):
    """
    Reads an image file and returns its base64 encoded string and mime type.
    """
    if not os.path.exists(image_path):
        return None, None, f"Error: Image path not found: {image_path}"

    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image'):
        return None, None, f"Error: Not a valid image file: {image_path}"

    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string, mime_type, None
    except Exception as e:
        return None, None, f"Error reading image file: {e}"


def image_ocr_llm_langchain(model_name: str, input_img_path: str, source_lang: str) -> str:
    """
    Transcribes text from an image using Google Gemini Vision capabilities.
    """
    
    # Map or fallback model name
    if "gemini" not in model_name.lower():
         model_name = DEFAULT_VISION_MODEL

    start_time = time.time()
    logger.info(f"Image transcriber: Transcribing image in {source_lang} with {model_name}...")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not found."

    chat_model = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        google_api_key=api_key,
        max_output_tokens=8192
    )

    image_base64, mime_type, err = get_image_base64(input_img_path)
    if err:
        logger.error(err)
        return err

    prompt_text = (
         f"Extract all text from this image. The language is {source_lang}. "
         "Are there any subtitles? If so, extract them exactly. "
         "Return ONLY the extracted text. DO NOT add any explanations."
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
            }
        ]
    )

    try:
        response = chat_model.invoke([message])
        final_transcription = response.content.strip()

    except Exception as e:
        logger.error(f"Image transcriber: LLM invocation failed. Error: {e}")
        return f"Error: LLM invocation failed. {e}"

    elapsed_time = time.time() - start_time
    logger.info(f"Image transcriber: Model {model_name} took {elapsed_time:.2f} seconds.")

    return final_transcription

if __name__ == "__main__":
    # Test
    test_path = "../media/FOREO_main_en_2.png" 
    # Ensure this path exists or put a dummy check
    if os.path.exists(test_path):
        res = image_ocr_llm_langchain(DEFAULT_VISION_MODEL, test_path, "English")
        print(res)
    else:
        print(f"Test image not found at {test_path}")