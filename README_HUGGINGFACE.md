# Deploying OncoScope to Hugging Face Spaces

## Overview

This guide explains how to deploy OncoScope as a Gradio app on Hugging Face Spaces without needing Ollama or local infrastructure.

## Key Differences from Local Version

1. **No Ollama Required**: Uses Hugging Face's Inference API instead
2. **No Backend Server**: Everything runs in a single Gradio app
3. **Simplified Architecture**: Direct model inference without FastAPI
4. **Cloud-Based**: Runs on Hugging Face's infrastructure

## Deployment Steps

### 1. Create a New Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - Space name: `oncoscope`
   - Select "Gradio" as the SDK
   - Choose "Public" or "Private" visibility
   - Select appropriate hardware (CPU or GPU)

### 2. Upload Your Fine-Tuned Model

First, upload your fine-tuned Gemma model to Hugging Face:

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login

# Upload your model (from the model directory)
huggingface-cli upload your-username/oncoscope-gemma-3n ./oncoscope-gemma-3n-gguf/
```

### 3. Configure Space Files

Upload these files to your Space:

1. **app.py** (rename `gradio_app_huggingface.py` to `app.py`)
2. **requirements.txt** (use `requirements_huggingface.txt`)

### 4. Set Environment Variables

In your Space settings, add these secrets:

- `HF_TOKEN`: Your Hugging Face access token (for private models)

### 5. Update Model Reference

In `app.py`, update the model reference:

```python
# Replace this line:
response = client.text_generation(
    prompt,
    model="google/gemma-2-2b-it",  # Replace with your model
    ...
)

# With:
response = client.text_generation(
    prompt,
    model="your-username/oncoscope-gemma-3n",
    ...
)
```

## Alternative: Using Transformers Directly

If you want to load the model directly instead of using the Inference API:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "your-username/oncoscope-gemma-3n"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.1,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Hardware Requirements

- **CPU Space**: Works but slower (30-60 seconds per mutation)
- **GPU Space**: Recommended for better performance
  - T4 GPU: Good balance of cost/performance
  - A10G: Better for heavy usage

## Limitations

1. **No Real Database**: Uses mock data for gene information
2. **Simplified Analysis**: Some features from the full version are simplified
3. **Rate Limits**: Subject to Hugging Face's rate limits
4. **Model Size**: Limited by Space storage (default 50GB)

## Testing Your Deployment

Once deployed, test with these example mutations:
- `BRAF c.1799T>A` (melanoma)
- `EGFR c.2369C>T` (lung cancer)
- `BRCA1 c.5266dupC` (breast/ovarian cancer)

## Converting GGUF to Hugging Face Format

If your model is in GGUF format, you may need to convert it:

```python
# This is a simplified example - actual conversion depends on your model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load from original format and save in HF format
model = AutoModelForCausalLM.from_pretrained("path/to/original")
tokenizer = AutoTokenizer.from_pretrained("path/to/original")

model.save_pretrained("./oncoscope-gemma-hf")
tokenizer.save_pretrained("./oncoscope-gemma-hf")
```

## Security Considerations

- Don't include sensitive data in the public Space
- Use private Spaces for production deployment
- Implement proper input validation
- Add usage disclaimers for medical applications

## Support

For issues specific to Hugging Face Spaces:
- Check [Hugging Face Documentation](https://huggingface.co/docs/hub/spaces)
- Visit [Hugging Face Forums](https://discuss.huggingface.co/)

For OncoScope-specific questions:
- Open an issue on the GitHub repository
- Contact: aristide021@gmail.com