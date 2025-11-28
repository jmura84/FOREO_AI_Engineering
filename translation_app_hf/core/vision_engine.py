import torch
import logging
import os
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from huggingface_hub import login, snapshot_download   # ❌ Comentado según petición
from qwen_vl_utils import process_vision_info
from dotenv import load_dotenv

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)


class VisionEngine:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None):
        """
        Vision model engine with dotenv support.
        HuggingFace authentication code is commented to disable HF Hub dependency.
        """
        # ---------------------------
        # 🔐 Load HF token (but don't use HF Hub)
        # ---------------------------
        token = os.getenv("HF_ACCESS_TOKEN")
        if token:
            logger.info("✓ HF_ACCESS_TOKEN loaded from .env (HF login disabled by request)")
        else:
            logger.warning("⚠ No HF_ACCESS_TOKEN found in .env")

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

        logger.info(f"VisionEngine initialized on {self.device} with model {self.model_name}")

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading vision model {self.model_name}...")

        try:
            # ----------------------------------------------------------
            # ❌ HF HUB DOWNLOAD DISABLED (commented)
            # model_path = snapshot_download(self.model_name, use_auth_token=os.getenv("HF_ACCESS_TOKEN"))
            # ----------------------------------------------------------

            # ✔ Instead load directly from model_name (HuggingFace will handle cache)
            model_path = self.model_name

            device_map = "auto" if self.device == "cuda" else None
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=dtype,
                device_map=device_map
            )
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=True
            )

            logger.info("✓ Vision model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading vision model: {e}")
            raise e

    def transcribe_image(self, image_path: str, source_lang: str) -> str:
        if self.model is None:
            self.load_model()

        try:
            image = Image.open(image_path).convert("RGB")
            prompt = f"Extract all text from this image. Language: {source_lang}. Return only extracted text."

            messages = [
                {"role": "user", "content":[
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return output_text[0]

        except Exception as e:
            logger.error(f"Error transcribing image: {e}")
            return f"Error: {e}"


# Singleton
_vision_engine = None

def get_vision_engine(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    global _vision_engine
    if _vision_engine is None or _vision_engine.model_name != model_name:
        _vision_engine = VisionEngine(model_name)
    return _vision_engine
