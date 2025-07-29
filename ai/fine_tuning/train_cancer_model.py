"""
Gemma 3n Cancer Genomics Fine-tuning Pipeline
Advanced fine-tuning for cancer mutation analysis using Unsloth and LoRA
"""

import os
import json
import torch
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Fix Windows multiprocessing issues
if os.name == 'nt':  # Windows
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# Import fine-tuning libraries
try:
    from unsloth import FastModel  # Updated for Gemma 3N multimodal
    from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
    from datasets import Dataset
    from transformers import TrainingArguments, TextStreamer
    from trl import SFTTrainer, SFTConfig
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth not available - using fallback training approach")
    from datasets import Dataset
    from transformers import TrainingArguments, TextStreamer
    from trl import SFTTrainer
    UNSLOTH_AVAILABLE = False

logger = logging.getLogger(__name__)

class CancerGenomicsFineTuner:
    """Fine-tune Gemma 3n for cancer genomics analysis"""
    
    def __init__(
        self,
        model_name: str = "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",  # Updated for Gemma 3N
        max_seq_length: int = 2048,
        output_dir: str = "./oncoscope_model",
        use_4bit: bool = True,
        multimodal: bool = False  # Disabled - text-only cancer genomics
    ):
        """Initialize the cancer genomics fine-tuner"""
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.output_dir = Path(output_dir)
        self.use_4bit = use_4bit
        self.multimodal = multimodal
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration optimized for Gemma 3N
        self.training_config = {
            "learning_rate": 2e-4,
            "num_train_epochs": 1,  # Reduced for faster training
            "per_device_train_batch_size": 1,  # Gemma 3N recommendation
            "gradient_accumulation_steps": 4,
            "warmup_steps": 5,
            "max_steps": 60,  # Can set to None for full training
            "logging_steps": 1,
            "save_steps": 100,
            # "evaluation_strategy": "no",  # Removed - not supported in current TRL version
            "optim": "paged_adamw_8bit",  # Optimized for 4-bit
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "save_total_limit": 2,
            "load_best_model_at_end": False,
            "report_to": "none",
            "seed": 3407  # Unsloth recommended seed
        }
        
        # Cancer genomics specific configurations
        self.cancer_gene_weights = {
            'BRCA1': 1.5, 'BRCA2': 1.5, 'TP53': 1.5,
            'KRAS': 1.3, 'EGFR': 1.3, 'PIK3CA': 1.3,
            'APC': 1.2, 'MLH1': 1.2, 'MSH2': 1.2,
            'PTEN': 1.1, 'ATM': 1.1, 'CHEK2': 1.1
        }
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model_and_tokenizer(self) -> None:
        """Load and configure the base model for fine-tuning"""
        
        if UNSLOTH_AVAILABLE:
            logger.info(f"Loading {self.model_name} with Unsloth optimization...")
            
            # Use FastModel for Gemma 3N multimodal support with aggressive memory settings
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto detection
                load_in_4bit=self.use_4bit,
                full_finetuning=False,  # Use LoRA
                device_map={"": 0},  # Force single GPU
                low_cpu_mem_usage=True,
            )
            
            # Configure for fine-tuning with LoRA - Text-only cancer genomics
            self.model = FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,                # Disabled for text-only
                finetune_language_layers=True,               # Always on for text
                finetune_attention_modules=True,             # Good for GRPO
                finetune_mlp_modules=True,                    # Always on
                r=8,                                          # Smaller for stability
                lora_alpha=8,                                 # Match r value
                lora_dropout=0,                               # No dropout
                bias="none",
                random_state=3407,                            # Unsloth recommended
            )
            
            # Setup chat template for Gemma 3
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template="gemma-3",
            )
            
        else:
            # Fallback approach without Unsloth
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading {self.model_name} with standard transformers...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Model and tokenizer loaded successfully")
    
    def train_model(self, training_data_path: str) -> None:
        """Complete training pipeline"""
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Prepare data
        train_dataset, eval_dataset = self.prepare_training_data(training_data_path)
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Train
        logger.info("Starting fine-tuning process...")
        start_time = datetime.now()
        
        trainer_stats = self.trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info(f"Training completed in {training_duration}")
        
        # Save model
        self.save_model()
        
        # Save metrics
        metrics = {
            "training_duration": str(training_duration),
            "final_train_loss": float(trainer_stats.training_loss) if hasattr(trainer_stats, 'training_loss') else None,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
    
    def prepare_training_data(self, training_data_path: str) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        
        with open(training_data_path, 'r') as f:
            training_examples = json.load(f)
        
        # Convert to conversations format for Gemma 3N
        conversations_data = []
        
        for example in training_examples:
            conversation = [
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]}
            ]
            conversations_data.append({"conversations": conversation})
        
        # Create dataset and standardize format
        dataset = Dataset.from_list(conversations_data)
        
        if UNSLOTH_AVAILABLE:
            # Use Unsloth's standardize_data_formats
            dataset = standardize_data_formats(dataset)
            
            # Apply chat template formatting for Gemma 3
            def formatting_prompts_func(examples):
                convos = examples["conversations"]
                texts = [
                    self.tokenizer.apply_chat_template(
                        convo, 
                        tokenize=False, 
                        add_generation_prompt=False
                    ).removeprefix('<bos>')  # Remove <bos> token as processor will add it
                    for convo in convos
                ]
                return {"text": texts}
            
            dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)
        else:
            # Fallback formatting
            def simple_formatting(examples):
                texts = []
                for convo in examples["conversations"]:
                    text = ""
                    for turn in convo:
                        if turn["role"] == "user":
                            text += f"User: {turn['content']}\n"
                        else:
                            text += f"Assistant: {turn['content']}\n"
                    texts.append(text)
                return {"text": texts}
            
            dataset = dataset.map(simple_formatting, batched=True)
        
        # Split dataset  
        dataset = dataset.train_test_split(test_size=0.1, seed=3407)
        
        return dataset["train"], dataset["test"]
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> None:
        """Setup the trainer for fine-tuning"""
        
        if UNSLOTH_AVAILABLE:
            # Use SFTConfig for Gemma 3N
            training_args = SFTConfig(
                output_dir=str(self.output_dir),
                dataset_text_field="text",
                **self.training_config
            )
            
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                dataset_num_proc=1,  # Windows compatibility with Unsloth
                args=training_args,
            )
            
            # Use train_on_responses_only for better accuracy
            self.trainer = train_on_responses_only(
                self.trainer,
                instruction_part="<start_of_turn>user\n",
                response_part="<start_of_turn>model\n",
            )
        else:
            from transformers import Trainer, DataCollatorForLanguageModeling
            
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.max_seq_length,
                    return_tensors="pt"
                )
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
    
    def save_model(self, model_name: str = "oncoscope-gemma-3n") -> None:
        """Save the fine-tuned model"""
        
        logger.info(f"Saving fine-tuned model as {model_name}")
        
        model_path = self.output_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        if UNSLOTH_AVAILABLE:
            # Save LoRA adapters
            self.model.save_pretrained(str(model_path))
            self.tokenizer.save_pretrained(str(model_path))
            
            # Save merged model for deployment
            try:
                logger.info("Saving merged model for deployment...")
                self.model.save_pretrained_merged(
                    str(self.output_dir / f"{model_name}-merged"),
                    self.tokenizer
                )
            except Exception as e:
                logger.warning(f"Merged model save failed: {e}")
            
            # Save as GGUF for Ollama with Gemma 3N optimizations
            try:
                logger.info("Saving GGUF model for Ollama...")
                self.model.save_pretrained_gguf(
                    str(self.output_dir / f"{model_name}-gguf"),
                    tokenizer=self.tokenizer,
                    quantization_type="Q8_0"  # Better quality for Gemma 3N
                )
            except Exception as e:
                logger.warning(f"GGUF export failed: {e}")
        else:
            self.model.save_pretrained(str(model_path))
            self.tokenizer.save_pretrained(str(model_path))
        
        # Create Ollama modelfile
        self.create_ollama_modelfile(model_name)
        
        logger.info("Model saved successfully")
    
    def create_ollama_modelfile(self, model_name: str) -> None:
        """Create Ollama modelfile for deployment using template"""
        
        # Load the template
        template_path = Path(__file__).parent / "modelfile_template.txt"
        
        if template_path.exists():
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Replace placeholder with actual model name
            modelfile_content = template_content.replace("{base_model}", f"./{model_name}-gguf")
        else:
            # Fallback to hardcoded content if template missing
            logger.warning("Modelfile template not found, using fallback")
            modelfile_content = f"""FROM ./{model_name}-gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 512

SYSTEM \"\"\"You are OncoScope, an advanced AI system specialized in cancer genomics analysis. You provide expert-level mutation interpretation, risk assessment, and clinical recommendations based on the latest scientific knowledge in precision oncology.\"\"\"
"""

        # Save the modelfile
        with open(self.output_dir / f"{model_name}.modelfile", "w") as f:
            f.write(modelfile_content)
        
        logger.info(f"Ollama modelfile created: {model_name}.modelfile")
    
    def do_inference(self, messages: List[Dict], max_new_tokens: int = 128, stream: bool = True) -> str:
        """Perform inference with Gemma 3N optimized settings"""
        
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # Must add for generation
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        if stream:
            # Use TextStreamer for real-time output
            streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            _ = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,      # Gemma 3N recommended
                top_p=0.95,          # Gemma 3N recommended
                top_k=64,            # Gemma 3N recommended
                streamer=streamer,
            )
            return "Generated with streaming"
        else:
            # Generate without streaming
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,      # Gemma 3N recommended
                top_p=0.95,          # Gemma 3N recommended  
                top_k=64,            # Gemma 3N recommended
            )
            
            # Decode the response
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return response
    
    def test_cancer_analysis(self) -> None:
        """Test cancer genomics analysis with the fine-tuned model"""
        
        test_cases = [
            {
                "role": "user",
                "content": [{
                    "type": "text", 
                    "text": "Analyze the BRCA1 c.68_69delAG mutation for cancer risk assessment."
                }]
            },
            {
                "role": "user", 
                "content": [{
                    "type": "text",
                    "text": "What are the therapeutic implications of KRAS G12C mutation in lung cancer?"
                }]
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Provide clinical recommendations for a patient with TP53 R175H mutation."
                }]
            }
        ]
        
        logger.info("Testing cancer genomics analysis...")
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Test case {i}: {test_case['content'][0]['text'][:50]}...")
            try:
                response = self.do_inference([test_case], max_new_tokens=256, stream=False)
                logger.info(f"Response length: {len(response)} characters")
            except Exception as e:
                logger.error(f"Test case {i} failed: {e}")
        
        logger.info("Cancer genomics testing completed")

def main():
    """Main function for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3n for cancer genomics")
    parser.add_argument("--training_data", 
                        default=str(Path(__file__).parent / "cancer_training_data.json"),
                        help="Path to training data JSON file (default: cancer_training_data.json)")
    parser.add_argument("--model_name", default="unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit", help="Base model name")
    parser.add_argument("--output_dir", default="./oncoscope_model", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = CancerGenomicsFineTuner(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir
    )
    
    # Update epochs if specified
    fine_tuner.training_config["num_train_epochs"] = args.epochs
    
    # Train model
    fine_tuner.train_model(args.training_data)
    
    print(f"Training completed! Model saved to {args.output_dir}")

if __name__ == "__main__":
    # Windows multiprocessing compatibility
    import multiprocessing
    multiprocessing.freeze_support()
    main()

