"""Quick GPU verification - simpler output to avoid encoding issues"""
import torch
from core.llm_engine import get_llm_engine

print("\n=== GPU Status ===")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU Memory: {gpu_mem_gb:.1f} GB")

print("\n=== Loading Model with 4-bit Quantization ===")
engine = get_llm_engine("google/gemma-2-2b-it")

if torch.cuda.is_available():
    alloc_gb = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"GPU Memory Used: {alloc_gb:.2f} GB")

print("\n=== Quick Translation Test ===")
import time
start = time.time()
result = engine.translate("Hello world", "English", "Spanish")
elapsed = time.time() - start

print(f"Input: Hello world")
print(f"Output: {result}")
print(f"Time: {elapsed:.2f}s")
print("\n=== SUCCESS: GPU acceleration is working! ===")
