import torch
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

class VisionEngine:
    """Vision engine for OCR using Qwen2-VL (or fallback to generic causal LM)."""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = None):
        self.model_name = model_name
        # Prefer GPU if available; otherwise CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        logger.info(f"VisionEngine initialized on {self.device} with model {self.model_name}")

    def load_model(self):
        """Load the model and processor.
        Uses `device_map='auto'` to let transformers place layers on the appropriate device.
        The processor is loaded with `use_fast=True` (default) for speed.
        """
        if self.model is not None:
            return
        logger.info(f"Loading vision model {self.model_name}...")
        try:
            if "Qwen" in self.model_name:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto",
                )
            # Fast processor (default) – explicit for clarity
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, use_fast=True)
            logger.info("Vision model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading vision model: {e}")
            raise e

    def transcribe_image(self, image_path: str, source_lang: str) -> str:
        """Run OCR on an image and return the extracted text.
        The method builds a chat‑style prompt, processes vision inputs via `process_vision_info`,
        and generates the answer with the loaded model.
        """
        if self.model is None:
            self.load_model()
        try:
            image = Image.open(image_path).convert("RGB")
            prompt = f"Extract all text from this image. The language is {source_lang}. Return ONLY the extracted text."
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            # Prepare text tokenization
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Vision processing (fast utility)
            image_inputs, video_inputs = process_vision_info(messages)
            # Build model inputs
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            # Generate answer
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            # Trim the prompt tokens from the output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        except Exception as e:
            logger.error(f"Error transcribing image: {e}")
            return f"Error: {e}"

_vision_engine = None

def get_vision_engine(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    global _vision_engine
    if _vision_engine is None:
        _vision_engine = VisionEngine(model_name)
    if _vision_engine.model_name != model_name:
        _vision_engine = VisionEngine(model_name)
    return _vision_engine
