from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
import torch

model_name = "deepseek-ai/DeepSeek-OCR"

print(f"Checking config for {model_name}...")
try:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"Config class: {config.__class__}")
    print(f"Architectures: {config.architectures}")
    
    # Try to register it manually to AutoModelForCausalLM if possible
    # But usually trust_remote_code does this.
    
except Exception as e:
    print(f"Error loading config: {e}")

print("\nAttempting to load with AutoModel...")
try:
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu")
    print(f"Loaded with AutoModel. Class: {model.__class__}")
except Exception as e:
    print(f"Error loading with AutoModel: {e}")
