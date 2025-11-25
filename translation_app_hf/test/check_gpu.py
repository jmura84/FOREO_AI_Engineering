"""
Diagnostic script to check GPU availability and current model device placement.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("=" * 60)
print("GPU DIAGNOSTICS")
print("=" * 60)

# Check PyTorch CUDA availability
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    print(f"✓ Current GPU: {torch.cuda.current_device()}")
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
    
    # Memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"\n✓ GPU Memory:")
    print(f"  - Total: {total_memory:.2f} GB")
    print(f"  - Allocated: {allocated:.2f} GB")
    print(f"  - Reserved: {reserved:.2f} GB")
    print(f"  - Free: {total_memory - reserved:.2f} GB")
else:
    print("\n⚠ WARNING: CUDA is NOT available!")
    print("  Inference will run on CPU, which is much slower.")
    print("  To fix this:")
    print("  1. Ensure you have an NVIDIA GPU")
    print("  2. Install CUDA toolkit")
    print("  3. Install PyTorch with CUDA support:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 60)
print("TESTING MODEL LOADING")
print("=" * 60)

# Test loading a small model to check device placement
model_name = "google/gemma-2-2b-it"
print(f"\nAttempting to load: {model_name}")
print("This may take a moment if not cached...")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTarget device: {device}")
    
    # Load tokenizer (always on CPU)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Tokenizer loaded")
    
    # Load model with device_map="auto" (recommended approach)
    print(f"\nLoading model with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Auto distributes across available devices
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    print(f"✓ Model loaded")
    
    # Check where model layers are placed
    print(f"\n✓ Model device map:")
    if hasattr(model, 'hf_device_map'):
        for layer, device in model.hf_device_map.items():
            print(f"  - {layer}: {device}")
    else:
        print(f"  - Model device: {next(model.parameters()).device}")
    
    # Check memory usage after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"\n✓ GPU Memory after model load:")
        print(f"  - Allocated: {allocated:.2f} GB")
        print(f"  - Reserved: {reserved:.2f} GB")
    
    print(f"\n✓ SUCCESS: Model loaded on {device.upper()}")
    
except Exception as e:
    print(f"\n✗ ERROR loading model: {e}")

print("\n" + "=" * 60)
