import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, model_name="google/gemma-2-2b-it", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        logger.info(f"LLMEngine initialized on {self.device} with model {self.model_name}")

    def load_model(self):
        if self.model is not None:
            return

        logger.info(f"Loading model {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e

    def translate(self, text: str, source_lang: str, target_lang: str, temperature: float = 0.3) -> str:
        if self.model is None:
            self.load_model()

        # Check for SRT format
        is_srt = bool(re.search(r'^\d+\s+\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}', text.strip(), re.MULTILINE))

        if is_srt:
            system_prompt = (
                f"You are a professional translator. "
                f"Translate the following SRT subtitles from {source_lang} to {target_lang}. "
                f"Preserve the SRT format (sequence numbers and timestamps) EXACTLY. "
                f"Translate ONLY the subtitle text. Do not add any explanations."
            )
        else:
            system_prompt = (
                f"You are a professional translator. "
                f"Translate the following text from {source_lang} to {target_lang}. "
                f"Return ONLY the translated text. Do not add any explanations."
            )

        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nText to translate:\n{text}"}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1]:]
        decoded_text = self.tokenizer.decode(response, skip_special_tokens=True)
        
        return decoded_text.strip()

# Singleton instance for easy access, though in a real app we might want dependency injection
# We'll initialize it lazily
_engine = None

def get_llm_engine(model_name="google/gemma-2-2b-it"):
    global _engine
    if _engine is None:
        _engine = LLMEngine(model_name)
    return _engine
