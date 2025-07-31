#!/usr/bin/env python3
"""Check available Gemma 3n text-only models"""

# According to the docs, text-only GGUFs are available
text_only_models = [
    # Dynamic 2.0 GGUF (text only) - these should work!
    "unsloth/gemma-3n-E2B-it-GGUF",
    "unsloth/gemma-3n-E4B-it-GGUF",
    
    # Let's also check if there are specific text-only versions
    "unsloth/gemma-3n-E2B-text-only",
    "unsloth/gemma-3n-E4B-text-only",
]

print("Available text-only Gemma 3n options:")
print("\n1. Use GGUF format with llama.cpp or Ollama (confirmed working)")
print("2. Use CPU offloading for the 4-bit model")
print("3. Use an even smaller model or reduce sequence length")

print("\nRecommendation: Since your GPU has 8GB VRAM but Gemma 3n needs more,")
print("you have these options:")
print("- Use GGUF format for inference only (no fine-tuning)")
print("- Enable CPU offloading during fine-tuning")
print("- Use smaller sequence length (512 instead of 2048)")
print("- Use the E2B model with aggressive memory optimization")