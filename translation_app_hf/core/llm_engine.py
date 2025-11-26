import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import re
import os
import google.generativeai as genai
import time

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

    def load_model(self, use_quantization=True):
        """Load model with GPU acceleration and optional quantization."""
        if self.model is not None:
            return

        # Google Gemini API Support
        if "gemini" in self.model_name:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found in environment variables")
                raise ValueError("GOOGLE_API_KEY is required for Gemini models")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"✓ Google Gemini model {self.model_name} configured")
            return

        logger.info(f"Loading model {self.model_name} on {self.device}...")
        try:
            # Load tokenizer with proper configuration for Gemma models
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                legacy=False
            )
            
            # Fix for CUDA device-side assert: ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("✓ Set pad_token to eos_token to prevent CUDA errors")
            
            # CRITICAL: For Gemma-3, ensure padding is on the left to avoid assertion errors
            if "gemma-3" in self.model_name.lower():
                self.tokenizer.padding_side = "left"
                logger.info("✓ Set padding_side='left' for Gemma-3 compatibility")
            
            # Configure model loading parameters
            load_kwargs = {
                "device_map": "auto",  # CRITICAL FIX: Auto-distribute layers across GPU
                "trust_remote_code": True,
                "attn_implementation": "eager"  # Use eager attention (fixes flash attention issues with Gemma-3)
            }
            
            # Enable 4-bit quantization for GPU (like Ollama's llama.cpp)
            if use_quantization and self.device == "cuda":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",  # Normal Float 4-bit (best quality)
                    bnb_4bit_use_double_quant=True  # Nested quantization for even better memory
                )
                load_kwargs["quantization_config"] = quantization_config
                logger.info("✓ Using 4-bit quantization for faster GPU inference")
            else:
                # Full precision fallback
                load_kwargs["dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                if not use_quantization:
                    logger.info("Quantization disabled, using float16 precision on GPU")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Log device placement for diagnostics
            if hasattr(self.model, 'hf_device_map'):
                logger.info(f"✓ Model device map: {self.model.hf_device_map}")
            else:
                device = next(self.model.parameters()).device
                logger.info(f"✓ Model loaded on device: {device}")
            
            # Log GPU memory usage if CUDA is available
            if self.device == "cuda" and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"✓ GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            raise e

    def translate(self, text: str, source_lang: str, target_lang: str, temperature: float = 0.3) -> str:
        if self.model is None:
            self.load_model()

        # Import SRT parser
        from core.srt_parser import SRTParser
        
        # --- STRATEGY 1: Google Gemini API (Full Text) ---
        if "gemini" in self.model_name:
            is_srt = SRTParser.is_srt_format(text)
            logger.info(f"Translating full text with {self.model_name} (API) - SRT: {is_srt}")
            
            system_prompt = (
                f"You are a professional translator. "
                f"Translate the following text from {source_lang} to {target_lang}. "
                f"If the target language is Spanish, use Spanish from Spain dialect. "
                "The texts you translate are from a beauty products brand called FOREO. "
            )
            
            if is_srt:
                system_prompt += (
                    "The input is a SubRip Subtitle (SRT) file. "
                    "You MUST preserve the exact structure: segment numbers and timestamps must remain unchanged. "
                    "Translate ONLY the subtitle text content. "
                    "Do not merge or split segments. Keep the exact same number of lines and segments. "
                    "Return the complete valid SRT file. "
                    "IMPORTANT: Do NOT wrap the output in markdown code blocks (e.g. ```srt ... ```). "
                    "Return ONLY the raw SRT content."
                )
            else:
                system_prompt += (
                    "Translate the exact text provided. Preserve the original formatting, including newlines. "
                    "Do not add any explanations, questions, or conversational filler. "
                    "Return ONLY the translated text."
                )
            
            try:
                # Add retry logic for 429 errors
                max_retries = 3
                base_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        response = self.model.generate_content(
                            f"{system_prompt}\n\nText to translate:\n{text}"
                        )
                        logger.info("✓ Gemini translation successful")
                        
                        # Post-processing: Strip markdown code blocks if present
                        cleaned_text = response.text.strip()
                        if cleaned_text.startswith("```"):
                            # Remove first line (```srt or just ```)
                            cleaned_text = re.sub(r"^```\w*\n", "", cleaned_text)
                            # Remove last line (```)
                            cleaned_text = re.sub(r"\n```$", "", cleaned_text)
                            
                        return cleaned_text.strip()
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            wait_time = base_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit. Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        raise e
                        
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                return f"[Error: {e}]"

        # --- STRATEGY 2: Local Models (Segment-by-Segment / Line-by-Line) ---
        
        # Check if input is SRT format
        if SRTParser.is_srt_format(text):
            logger.info("✓ SRT format detected - translating segment by segment...")
            
            # Parse SRT into segments
            segments = SRTParser.parse_srt(text)
            translated_segments = []
            
            # Translate segment by segment to guarantee format preservation
            for i, (seg_num, timestamp, text_lines) in enumerate(segments):
                original_text = "\n".join(text_lines)
                
                # Skip empty segments
                if not original_text.strip():
                    translated_segments.append((seg_num, timestamp, text_lines))
                    continue
                
                # Construct prompt for single segment
                system_prompt = (
                    f"You are a professional translator. "
                    f"Translate the following text from {source_lang} to {target_lang}. "
                    f"If the target language is Spanish, use Spanish from Spain dialect. "
                    "The texts you translate are from a beauty products brand called FOREO. "
                    "Return ONLY the translated text. Do not add any explanations."
                )
                
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\nText to translate:\n{original_text}"}
                ]
                
                # Hugging Face Transformers generation
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    padding=False,
                    truncation=False
                )
                
                input_ids = input_ids.to(self.model.device)
                attention_mask = torch.ones_like(input_ids, device=self.model.device)
                
                terminators = [self.tokenizer.eos_token_id]
                end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                if end_of_turn_id != self.tokenizer.unk_token_id and end_of_turn_id >= 0:
                    terminators.append(end_of_turn_id)
                
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                response = outputs[0][input_ids.shape[-1]:]
                translated_text = self.tokenizer.decode(response, skip_special_tokens=True).strip()
                
                # Split lines if translation has multiple lines
                translated_lines = translated_text.split('\n')
                translated_segments.append((seg_num, timestamp, translated_lines))
                
                # Log progress every 5 segments
                if (i + 1) % 5 == 0:
                    logger.info(f"Translated {i+1}/{len(segments)} segments")
            
            # Reconstruct full SRT
            result = SRTParser.reconstruct_srt(translated_segments)
            logger.info("✓ All segments successfully translated")
            return result
        
        # Regular text translation (non-SRT)
        else:
            system_prompt = (
                f"You are a professional translator. "
                f"Your task is to translate *only* the user's text from {source_lang} to {target_lang}. "
                f"If the target language is Spanish, use Spanish from Spain dialect. "
                "The texts you translate are from a beauty products brand called FOREO. "
                "Translate the exact text provided by the user. "
                "Always assume that the entire user message is the text to be translated. "
                f"Return *only* the final {target_lang} translation. "
                "DO NOT add any explanations, questions, or comments. "
                "DO NOT add segment numbers. DO NOT add timestamps. "
                "DO NOT add any text that was not in the original segment. "
                "Your response must be *only* the translated text and nothing else."
            )

            # Split text into segments (lines)
            lines = text.split('\n')
            translated_lines = []
            
            # Count non-empty segments for progress logging
            total_segments = sum(1 for line in lines if line.strip())
            processed_segments = 0
            
            logger.info(f"Translating {total_segments} text segments...")
            
            for line in lines:
                if not line.strip():
                    translated_lines.append(line) # Preserve empty lines
                    continue
                
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\nText to translate:\n{line}"}
                ]

                # Hugging Face Transformers generation
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    padding=False,
                    truncation=False
                )
                
                input_ids = input_ids.to(self.model.device)
                attention_mask = torch.ones_like(input_ids, device=self.model.device)

                terminators = [self.tokenizer.eos_token_id]
                end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                if end_of_turn_id != self.tokenizer.unk_token_id and end_of_turn_id >= 0:
                    terminators.append(end_of_turn_id)

                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                response = outputs[0][input_ids.shape[-1]:]
                decoded_text = self.tokenizer.decode(response, skip_special_tokens=True).strip()
                translated_lines.append(decoded_text)
                
                processed_segments += 1
                if processed_segments % 5 == 0:
                    logger.info(f"Translated {processed_segments}/{total_segments} segments")
            
            logger.info("✓ All segments successfully translated")
            return "\n".join(translated_lines)

# Singleton instance
_engine = None

def get_llm_engine(model_name="google/gemma-2-2b-it"):
    global _engine
    if _engine is None:
        _engine = LLMEngine(model_name)
    return _engine
