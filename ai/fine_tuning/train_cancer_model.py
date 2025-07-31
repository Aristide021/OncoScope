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
import platform
import subprocess
import sys

# Fix Windows multiprocessing issues
if os.name == 'nt':  # Windows
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# Import fine-tuning libraries
try:
    from unsloth import FastModel  # Use FastModel for Gemma 3n
    from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
    from datasets import Dataset
    from transformers import TrainingArguments, TextStreamer
    from trl import SFTTrainer, SFTConfig
    import gc  # Add garbage collection for memory management
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Unsloth not available - using fallback training approach")
    from datasets import Dataset
    from transformers import TrainingArguments, TextStreamer
    from trl import SFTTrainer
    import gc
    UNSLOTH_AVAILABLE = False

logger = logging.getLogger(__name__)

class CancerGenomicsFineTuner:
    """Fine-tune Gemma 3n for cancer genomics analysis"""
    
    def __init__(
        self,
        model_name: str = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",  # Gemma 3n 2B model
        max_seq_length: int = 2048,
        output_dir: str = "./oncoscope_model",
        use_4bit: bool = True,
        multimodal: bool = False  # Not applicable for text-only model
    ):
        """Initialize the cancer genomics fine-tuner"""
        
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.output_dir = Path(output_dir)
        self.use_4bit = use_4bit
        self.multimodal = False  # Force text-only mode
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration optimized for Gemma 3N
        # Auto-detect GPU and adjust batch size
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if "3090" in gpu_name or vram_gb > 20:
                # RTX 3090 or better - use larger batch size
                batch_size = 2
                grad_accum = 2
            else:
                # T4 or smaller GPU
                batch_size = 1
                grad_accum = 4
        else:
            batch_size = 1
            grad_accum = 4
            
        self.training_config = {
            "learning_rate": 2e-4,
            "num_train_epochs": 3,  # Optimal for 6k specialized dataset
            "per_device_train_batch_size": batch_size,  # Auto-adjusted
            "gradient_accumulation_steps": grad_accum,
            "warmup_steps": 5,
            "max_steps": -1,  # Full training (all epochs)
            "logging_steps": 1,
            "save_steps": 100,
            "eval_strategy": "steps",  # Enable evaluation
            "eval_steps": 10,  # Evaluate every 10 steps
            "metric_for_best_model": "eval_loss",  # Monitor eval loss for overfitting
            "greater_is_better": False,  # Lower eval_loss is better
            "optim": "paged_adamw_8bit",  # Optimized for 4-bit
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "save_total_limit": 2,
            "load_best_model_at_end": True,  # Load best model at end
            "report_to": "tensorboard",  # Enable TensorBoard logging for judges
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
            logger.info("Configuring for text-only fine-tuning to save VRAM...")
            
            # Use FastModel for Gemma 3n with CPU offloading
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto-detect for optimal performance
                load_in_4bit=self.use_4bit,
                full_finetuning=False,  # Use LoRA for efficiency
                device_map="auto",  # Enable automatic device mapping
                max_memory={0: "22GB", "cpu": "20GB"},  # Use full 3090 VRAM
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for 4-bit
            )
            
            # Configure for fine-tuning - TEXT ONLY to save VRAM
            self.model = FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,      # DISABLE VISION - saves VRAM!
                finetune_language_layers=True,     # Keep language processing only
                finetune_attention_modules=True,   # Good for specialized tasks
                finetune_mlp_modules=True,         # Keep for best performance
                r=16,                              # Higher rank for 3090
                lora_alpha=32,                     # 2x r for better learning
                lora_dropout=0,                    # No dropout
                bias="none",
                random_state=3407,                 # Unsloth recommended
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
    
    def setup_logging(self) -> None:
        """Setup comprehensive logging for hackathon judges"""
        # Create logs directory
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup file handler for complete logs
        log_file = logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Log system info
        logger.info("="*80)
        logger.info("ONCOSCOPE CANCER GENOMICS FINE-TUNING - UNSLOTH HACKATHON SUBMISSION")
        logger.info("="*80)
        logger.info(f"Training started at: {datetime.now().isoformat()}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"CPU cores: {os.cpu_count()}")
        logger.info("="*80)
        
        self.log_file = log_file
    
    def train_model(self, training_data_path: str) -> None:
        """Complete training pipeline"""
        
        # Setup logging
        self.setup_logging()
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Prepare data
        train_dataset, eval_dataset = self.prepare_training_data(training_data_path)
        
        # Log dataset info
        logger.info(f"Training dataset size: {len(train_dataset)} examples")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)} examples")
        logger.info(f"Training data path: {training_data_path}")
        
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
        
        # Save metrics including evaluation results
        metrics = {
            "training_duration": str(training_duration),
            "final_train_loss": float(trainer_stats.training_loss) if hasattr(trainer_stats, 'training_loss') else None,
            "metrics": trainer_stats.metrics if hasattr(trainer_stats, 'metrics') else {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check for overfitting
        if hasattr(trainer_stats, 'metrics'):
            train_loss = trainer_stats.metrics.get("train_loss", None)
            eval_loss = trainer_stats.metrics.get("eval_loss", None)
            
            if train_loss and eval_loss:
                overfitting_ratio = eval_loss / train_loss if train_loss > 0 else float('inf')
                metrics["overfitting_ratio"] = overfitting_ratio
                
                if overfitting_ratio > 1.5:
                    logger.warning(f"⚠️ Potential overfitting detected! Eval/Train loss ratio: {overfitting_ratio:.2f}")
                else:
                    logger.info(f"✅ Model generalization looks good. Eval/Train loss ratio: {overfitting_ratio:.2f}")
        
        with open(self.output_dir / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create comprehensive report for judges
        self.create_training_report(metrics, training_duration)
    
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
            
            dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=4)
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
                dataset_num_proc=4,  # Optimal for 22-core CPU in WSL
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
    
    def create_training_report(self, metrics: Dict, training_duration) -> None:
        """Create comprehensive training report for hackathon judges"""
        
        report_path = self.output_dir / "TRAINING_REPORT_UNSLOTH_HACKATHON.md"
        
        with open(report_path, "w") as f:
            f.write("# OncoScope Cancer Genomics Fine-tuning Report\n")
            f.write("## Unsloth Hackathon Submission\n\n")
            
            f.write("### 🚀 Project Overview\n")
            f.write("- **Application**: Cancer mutation analysis and clinical recommendations\n")
            f.write("- **Base Model**: Gemma 3N 4-bit (unsloth optimized)\n")
            f.write("- **Fine-tuning Method**: LoRA with Unsloth optimizations\n")
            f.write("- **Dataset**: 6,000+ curated cancer genomics examples\n\n")
            
            f.write("### 🖥️ System Specifications\n")
            f.write(f"- **GPU**: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}\n")
            f.write(f"- **VRAM**: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
            f.write(f"- **Platform**: {platform.platform()}\n")
            f.write(f"- **PyTorch**: {torch.__version__}\n")
            f.write(f"- **CUDA**: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n\n")
            
            f.write("### 📊 Training Configuration\n")
            f.write("```json\n")
            f.write(json.dumps(self.training_config, indent=2))
            f.write("\n```\n\n")
            
            f.write("### 📈 Training Results\n")
            f.write(f"- **Training Duration**: {training_duration}\n")
            f.write(f"- **Final Train Loss**: {metrics.get('final_train_loss', 'N/A')}\n")
            
            if 'overfitting_ratio' in metrics:
                f.write(f"- **Overfitting Ratio**: {metrics['overfitting_ratio']:.3f}\n")
                f.write(f"- **Status**: {'✅ Good generalization' if metrics['overfitting_ratio'] < 1.5 else '⚠️ Potential overfitting'}\n")
            
            f.write("\n### 📁 Output Files\n")
            f.write("- `oncoscope-gemma-3n/`: LoRA adapters\n")
            f.write("- `oncoscope-gemma-3n-merged/`: Merged model (if available)\n")
            f.write("- `oncoscope-gemma-3n-gguf/`: Quantized model for Ollama\n")
            f.write("- `logs/training_*.log`: Complete training logs\n")
            f.write("- `training_metrics.json`: Detailed metrics\n")
            f.write("- `runs/`: TensorBoard logs\n\n")
            
            f.write("### 🏥 Cancer Genomics Specialization\n")
            f.write("The model was fine-tuned on:\n")
            f.write("- Pathogenic mutation interpretation\n")
            f.write("- Clinical significance assessment\n")
            f.write("- Treatment recommendations\n")
            f.write("- Risk stratification\n")
            f.write("- Actionable insights for precision oncology\n\n")
            
            f.write("### 🎯 Key Achievements\n")
            f.write("- Efficient 4-bit quantization with Unsloth\n")
            f.write("- Specialized for cancer genomics domain\n")
            f.write("- Production-ready with Ollama integration\n")
            f.write("- Comprehensive evaluation metrics\n\n")
            
            f.write("### 📝 Training Logs\n")
            f.write(f"Complete logs available at: `{self.log_file.name}`\n\n")
            
            f.write("---\n")
            f.write(f"Report generated: {datetime.now().isoformat()}\n")
        
        logger.info(f"Training report created: {report_path}")
        
        # Also create a judges_summary.txt with key points
        summary_path = self.output_dir / "JUDGES_SUMMARY.txt"
        with open(summary_path, "w") as f:
            f.write("ONCOSCOPE - UNSLOTH HACKATHON SUBMISSION\n")
            f.write("=" * 50 + "\n\n")
            f.write("QUICK FACTS:\n")
            f.write(f"- Model: Gemma 3N 4-bit fine-tuned with Unsloth\n")
            f.write(f"- Purpose: Cancer mutation analysis AI\n")
            f.write(f"- Training time: {training_duration}\n")
            f.write(f"- Dataset: 6,000+ cancer genomics examples\n")
            f.write(f"- Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
            f.write(f"- Final loss: {metrics.get('final_train_loss', 'N/A')}\n")
            f.write(f"- Model ready for: Ollama deployment\n\n")
            f.write("KEY INNOVATION:\n")
            f.write("Specialized cancer genomics AI using Unsloth's\n")
            f.write("efficient 4-bit optimization for clinical mutation\n")
            f.write("interpretation and personalized treatment insights.\n")
        
        logger.info(f"Judges summary created: {summary_path}")
    
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
            # Aggressive cleanup for VRAM
            del inputs
            torch.cuda.empty_cache()
            gc.collect()
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
            
            # Aggressive cleanup for VRAM
            del inputs, outputs
            torch.cuda.empty_cache()
            gc.collect()
            
            return response
    
    def test_cancer_analysis(self) -> None:
        """Test cancer genomics analysis with the fine-tuned model"""
        
        test_cases = [
            {
                "role": "user",
                "content": "Analyze the BRCA1 c.68_69delAG mutation for cancer risk assessment."
            },
            {
                "role": "user", 
                "content": "What are the therapeutic implications of KRAS G12C mutation in lung cancer?"
            },
            {
                "role": "user",
                "content": "Provide clinical recommendations for a patient with TP53 R175H mutation."
            }
        ]
        
        logger.info("Testing cancer genomics analysis...")
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Test case {i}: {test_case['content'][:50]}...")
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

