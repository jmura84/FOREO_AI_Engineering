import pandas as pd
# --- Importamos las herramientas de LangChain ---
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
# ---
import torch
import logging
import re
import time
import base64
import os
import mimetypes

logging.getLogger("langchain.utilities.requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# If the log comes directly from the LangChain core:
logging.getLogger("langchain.llms").setLevel(logging.WARNING)

# We check if the library is using the GPU:
print(f"CUDA available according to PyTorch: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device detected: {torch.cuda.get_device_name(0)}")

# It can be 'medium' or 'high' precision:
torch.set_float32_matmul_precision('high')


# --- Necesitamos esta función de ayuda para LangChain ---
def get_image_base64(image_path: str):
    """
    Reads an image file and returns its base64 encoded string and mime type.
    LangChain wrappers (unlike the native 'ollama' lib) need this.
    Returns (base64_string, mime_type, error_message)
    """
    if not os.path.exists(image_path):
        return None, None, f"Error: Image path not found: {image_path}"

    # Guess the MIME type of the file
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image'):
        return None, None, f"Error: Not a valid image file: {image_path}"

    try:
        # Read the image file in binary mode
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string, mime_type, None
    except Exception as e:
        return None, None, f"Error reading image file: {e}"


def image_ocr_llm_langchain(model_name: str, input_img_path: str, source_lang: str) -> str:
    """
    Transcribes text from an image using the LangChain ChatOllama wrapper.
    (This is the GKE-ready approach)
    """

    # --- Start timing: ---
    start_time = time.time()
    print(f"Image transcriber (LangChain): Transcribing the image in {source_lang} language with {model_name}...")

    # 1. Initialize the Chat Model
    chat_model = ChatOllama(
        model=model_name,
        temperature=0.1,
        top_p = 0.9,
        top_k = 50,
        num_predict = 2048
    )

    # 2. Encode the image to base64 (LangChain lo necesita así)
    image_base64, mime_type, err = get_image_base64(input_img_path)
    if err:
        print(f"Image transcriber (LangChain): {err}")
        return f"Error: {err}"

    # 3. Create the list of messages
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f'Extract all text from this image. The language is {source_lang}. DO NOT add any content or explanation from your side.'
                },
                {
                    "type": "image_url",
                    # Formateamos el data URL que LangChain espera
                    "image_url": f"data:{mime_type};base64,{image_base64}"
                }
            ]
        )
    ]

    try:
        # 4. Invoke the chat model with the message list
        response = chat_model.invoke(messages)

        # The response object has a 'content' attribute with the text
        final_transcription = response.content

    except Exception as e:
        print(f"Image transcriber (LangChain): LLM invocation failed. Error: {e}")
        return f"Error: LLM invocation failed. {e}"

    # --- Measure time elapsed: ---
    elapsed_time = time.time() - start_time

    print("\n--- Transcription Result ---")
    print(final_transcription)
    print("---------------------------------")

    print(f"Image transcriber (LangChain): Model {model_name} took {elapsed_time:.2f} seconds.")

    return final_transcription


# --- TEST BLOCK ---
if __name__ == "__main__":

    # --- USER: Define your test parameters here ---
    # (Usando los mismos valores de tu script funcional)
    # test_model_name = "gemma3:4b-it-qat"
    # test_model_name = "gemma3:4b"
    test_model_name = "gemma3:12b"
    test_image_path = r"../media/FOREO_main_en_2.png"
    test_source_lang = "English"
    # --- End of user parameters ---

    print(f"--- Running Image Transcriptor Test (LangChain Version) ---")
    print(f"Model: {test_model_name}")
    print(f"Image: {test_image_path}")

    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at '{test_image_path}'.")
        print(f"Please update the 'test_image_path' variable in the script.")
    else:
        # Llama a la nueva función de LangChain
        transcription = image_ocr_llm_langchain(
            model_name=test_model_name,
            input_img_path=test_image_path,
            source_lang=test_source_lang
        )