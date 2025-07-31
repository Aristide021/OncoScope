#!/usr/bin/env python3
"""Test VRAM usage with optimized model loading"""

import torch
import gc
import os
# Set environment variable to reduce VRAM usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from ai.fine_tuning.train_cancer_model import CancerGenomicsFineTuner

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0

print("Testing VRAM-optimized model loading...")
print(f"Initial VRAM usage: {get_vram_usage():.2f} GB")

# Initialize fine-tuner with VRAM optimizations
# Try the smaller E2B model instead of E4B
fine_tuner = CancerGenomicsFineTuner(
    model_name="unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",  # Smaller 2B model
    max_seq_length=512,  # Further reduced for testing
    output_dir="./test_model",
    use_4bit=True,
    multimodal=False  # Explicitly disabled
)

print("Loading model with multimodal features disabled...")
try:
    fine_tuner.load_model_and_tokenizer()
    print(f"✅ Model loaded successfully!")
    print(f"VRAM usage after loading: {get_vram_usage():.2f} GB")
    
    # Test inference
    test_message = [{"role": "user", "content": "What is BRCA1?"}]
    print("\nTesting inference...")
    response = fine_tuner.do_inference(test_message, max_new_tokens=50, stream=False)
    print(f"✅ Inference successful!")
    print(f"VRAM usage after inference: {get_vram_usage():.2f} GB")
    
    # Cleanup
    del fine_tuner
    torch.cuda.empty_cache()
    gc.collect()
    print(f"VRAM usage after cleanup: {get_vram_usage():.2f} GB")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()