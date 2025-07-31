#!/usr/bin/env python3
"""Test with text-only Gemma model to avoid multimodal VRAM usage"""

import torch
import gc
import os

# Set environment variables for memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0

print("Testing text-only Gemma model loading...")
print(f"Initial VRAM usage: {get_vram_usage():.2f} GB")

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    
    # Try standard Gemma 2B model (text-only, not multimodal)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-2b-it-bnb-4bit",  # Regular Gemma, not 3n
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"VRAM usage after loading: {get_vram_usage():.2f} GB")
    
    # Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    print(f"✅ LoRA configured!")
    print(f"VRAM usage after LoRA: {get_vram_usage():.2f} GB")
    
    # Setup chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")
    
    # Test inference
    messages = [{"role": "user", "content": "What is BRCA1?"}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    from transformers import TextStreamer
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    print("\nTesting inference...")
    _ = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.1,
        streamer=streamer,
    )
    
    print(f"\n✅ Inference successful!")
    print(f"VRAM usage after inference: {get_vram_usage():.2f} GB")
    
    # Cleanup
    del model, tokenizer, inputs
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"VRAM usage after cleanup: {get_vram_usage():.2f} GB")
    print("\n✅ SUCCESS: Consider using regular Gemma models instead of multimodal 3n!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()