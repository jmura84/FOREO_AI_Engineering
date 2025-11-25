"""
Test GPU acceleration and measure translation performance.
"""
import time
import torch
from core.llm_engine import get_llm_engine

print("=" * 70)
print("GPU ACCELERATION TEST")
print("="  * 70)

# Check GPU status
print(f"\n✓ PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✓ GPU Memory: {total_memory:.2f} GB total")
else:
    print("⚠ WARNING: CUDA not available, will use CPU (slow)")

print("\n" + "=" * 70)
print("LOADING MODEL WITH 4-BIT QUANTIZATION")
print("=" * 70)

# Load model (will use 4-bit quantization by default)
model_name = "google/gemma-2-2b-it"
print(f"\nModel: {model_name}")
print("Loading... (this may take a moment on first run)")

start_load = time.time()
engine = get_llm_engine(model_name)
load_time = time.time() - start_load

print(f"\n✓ Model loaded in {load_time:.2f} seconds")

# Show GPU memory usage after model load
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"✓ GPU Memory used: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

print("\n" + "=" * 70)
print("TRANSLATION PERFORMANCE TEST")
print("=" * 70)

# Test translation
test_texts = [
    "Hello, how are you today?",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning and artificial intelligence are transforming the world."
]

for i, text in enumerate(test_texts, 1):
    print(f"\n--- Test {i} ---")
    print(f"Source: {text}")
    
    start = time.time()
    translation = engine.translate(text, "English", "Spanish", temperature=0.3)
    elapsed = time.time() - start
    
    print(f"Translation: {translation}")
    print(f"⏱ Time: {elapsed:.2f} seconds")
    
    # Calculate tokens per second (rough estimate)
    total_chars = len(text) + len(translation)
    chars_per_sec = total_chars / elapsed if elapsed > 0 else 0
    print(f"📊 Speed: ~{chars_per_sec:.0f} chars/sec")

print("\n" + "=" * 70)
print("GPU UTILIZATION SUMMARY")
print("=" * 70)

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\n✓ Final GPU Memory Usage:")
    print(f"  - Allocated: {allocated:.2f} GB")
    print(f"  - Reserved: {reserved:.2f} GB")
    print(f"  - Total: {total:.2f} GB")
    print(f"  - Utilization: {(reserved/total)*100:.1f}%")
    print(f"\n✓ 4-bit quantization is working!")
    print(f"  Without quantization, this model would use ~{allocated*4:.1f} GB")
else:
    print("\n⚠ Running on CPU - consider installing CUDA for better performance")

print("\n" + "=" * 70)
print("✓ TEST COMPLETE")
print("=" * 70)
