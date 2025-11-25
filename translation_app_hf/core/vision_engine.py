import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class VisionEngine:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None):
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
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("Vision model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading vision model: {e}")
            raise e

    def transcribe_image(self, image_path: str, source_lang: str) -> str:
        if self.model is None:
            self.load_model()

        try:
            image = Image.open(image_path)
            
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

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = self.processor.process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            
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

def get_vision_engine(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    global _vision_engine
    if _vision_engine is None:
        _vision_engine = VisionEngine(model_name)
    return _vision_engine
