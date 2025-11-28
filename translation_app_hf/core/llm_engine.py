import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import google.generativeai as genai
import re

logger = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, model_name="gemini-2.5-flash-lite", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_gemini = "gemini" in model_name.lower()
        logger.info(f"LLMEngine initialized on {self.device} with model {self.model_name}")

    def load_model(self, temperature=0.3, top_p=0.9, top_k=40, max_output_tokens=8000):
        if self.model is not None:
            return

        if self.is_gemini:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is required for Gemini models")
            
            genai.configure(api_key=api_key)
            
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": "text/plain",
            }
            
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config
            )
            logger.info(f"✓ Gemini model {self.model_name} configured")
            return

        # Local HF fallback
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_kwargs = {"trust_remote_code": True, "device_map": "auto" if self.device=="cuda" else None}
            load_kwargs["dtype"] = torch.float16 if self.device=="cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
            logger.info(f"✓ Local HF model loaded on {self.device}")
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            raise e

    def translate(self, text: str, source_lang: str, target_lang: str, temperature=0.3, top_p=0.9, top_k=40):
        if self.model is None:
            self.load_model(temperature=temperature, top_p=top_p, top_k=top_k)

        if self.is_gemini:
            system_prompt = (
                f"You are a professional translator. Translate the following content from {source_lang} to {target_lang}. "
                "If the input is SRT format, preserve numbering, timestamps, and formatting. "
                "Return ONLY the translated content maintaining the original format."
            )
            try:
                response = self.model.generate_content(f"{system_prompt}\n\nContent:\n{text}")
                cleaned_text = response.text.strip()
                cleaned_text = re.sub(r"^```.*\n|```$", "", cleaned_text)
                return cleaned_text
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return f"[Error: {e}]"

        # Local HF
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=2048,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Singleton
_engine = None
def get_llm_engine(model_name="gemini-2.5-flash-lite", temperature=0.3):
    global _engine
    if _engine is None or _engine.model_name != model_name:
        _engine = LLMEngine(model_name)
        _engine.load_model(temperature=temperature)
    return _engine
