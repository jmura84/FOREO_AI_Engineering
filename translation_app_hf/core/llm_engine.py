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

    def load_model(self, use_quantization=True):
        """Load model with GPU acceleration and optional quantization.
        
        Args:
            use_quantization: Enable 4-bit quantization for faster inference (default: True)
                             NOTE: Works well with Gemma-2; Gemma-3 has CUDA compatibility issues
        """
        if self.model is not None:
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
        
        # Check if input is SRT format
        if SRTParser.is_srt_format(text):
            logger.info("✓ SRT format detected - translating segment by segment...")
            
            # Parse SRT into segments
            segments = SRTParser.parse_srt(text)
            translated_segments = []
            
            # Translate segment by segment to guarantee format preservation
            # This is slower but much more reliable for format correctness
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
                
                # Tokenize and generate
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
                    max_new_tokens=512,  # Shorter max tokens for single segment
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

                # Apply chat template and prepare inputs with proper padding
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    padding=False,  # Don't pad the input
                    truncation=False
                )
                
                # Move to device and create attention mask
                input_ids = input_ids.to(self.model.device)
                attention_mask = torch.ones_like(input_ids, device=self.model.device)

                # Build terminator list, filtering out invalid IDs (prevents CUDA assert errors)
                terminators = [self.tokenizer.eos_token_id]
                
                # Try to add custom terminators if they exist in vocabulary
                end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
                if end_of_turn_id != self.tokenizer.unk_token_id and end_of_turn_id >= 0:
                    terminators.append(end_of_turn_id)

                # Generate with explicit attention mask to prevent CUDA errors
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # Explicitly pass attention mask
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id  # Explicitly set pad token
                )

                response = outputs[0][input_ids.shape[-1]:]
                decoded_text = self.tokenizer.decode(response, skip_special_tokens=True).strip()
                translated_lines.append(decoded_text)
                
                processed_segments += 1
                if processed_segments % 5 == 0:
                    logger.info(f"Translated {processed_segments}/{total_segments} segments")
            
            logger.info("✓ All segments successfully translated")
            return "\n".join(translated_lines)

# Singleton instance for easy access, though in a real app we might want dependency injection
# We'll initialize it lazily
_engine = None

def get_llm_engine(model_name="google/gemma-2-2b-it"):
    global _engine
    if _engine is None:
        _engine = LLMEngine(model_name)
    return _engine
