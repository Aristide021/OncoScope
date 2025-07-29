"""
OncoScope Ollama Setup Script
Configures Ollama for OncoScope cancer genomics model deployment
"""

import os
import json
import subprocess
import requests
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OncoScopeOllamaSetup:
    """Setup and configure Ollama for OncoScope"""
    
    def __init__(self, model_dir: str = "../ai/fine_tuning", ollama_host: str = "localhost:11434"):
        """Initialize Ollama setup"""
        self.model_dir = Path(__file__).parent / model_dir
        self.ollama_host = ollama_host
        self.ollama_url = f"http://{ollama_host}"
        
        # OncoScope model configurations
        self.oncoscope_models = {
            "oncoscope-gemma-3n": {
                "base_model": "gemma2:2b",
                "description": "OncoScope cancer genomics analysis model based on Gemma 3n",
                "model_file": "oncoscope-gemma-3n.modelfile",
                "gguf_path": "oncoscope-gemma-3n-gguf",
                "system_prompt": self._get_oncoscope_system_prompt(),
                "parameters": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 512,
                    "num_ctx": 2048
                }
            },
            "oncoscope-analyzer": {
                "base_model": "gemma2:2b",
                "description": "Specialized model for mutation analysis and clinical interpretation",
                "system_prompt": self._get_analyzer_system_prompt(),
                "parameters": {
                    "temperature": 0.05,
                    "top_p": 0.95,
                    "num_predict": 1024
                }
            },
            "oncoscope-reporter": {
                "base_model": "gemma2:2b", 
                "description": "Model optimized for clinical report generation",
                "system_prompt": self._get_reporter_system_prompt(),
                "parameters": {
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "num_predict": 2048
                }
            }
        }
    
    def setup_ollama_environment(self) -> bool:
        """Setup complete Ollama environment for OncoScope"""
        logger.info("Setting up Ollama environment for OncoScope...")
        
        try:
            # Check Ollama installation
            if not self._check_ollama_installation():
                logger.error("Ollama not found. Please install Ollama first.")
                return False
            
            # Start Ollama service if needed
            self._ensure_ollama_running()
            
            # Pull base models
            self._pull_base_models()
            
            # Create OncoScope models
            self._create_oncoscope_models()
            
            # Validate models
            self._validate_models()
            
            # Create helper scripts
            self._create_helper_scripts()
            
            logger.info(" Ollama environment setup completed!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Ollama environment: {e}")
            return False
    
    def _check_ollama_installation(self) -> bool:
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                   capture_output=True, text=True, check=True)
            logger.info(f"Ollama version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _ensure_ollama_running(self) -> None:
        """Ensure Ollama service is running"""
        logger.info("Checking Ollama service...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama service is running")
                return
        except requests.exceptions.RequestException:
            pass
        
        logger.info("Starting Ollama service...")
        try:
            # Try to start Ollama in background
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for i in range(30):
                try:
                    response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        logger.info("Ollama service started successfully")
                        return
                except requests.exceptions.RequestException:
                    time.sleep(1)
            
            logger.warning("Ollama service may not be fully started")
            
        except Exception as e:
            logger.warning(f"Could not start Ollama service automatically: {e}")
            logger.info("Please start Ollama manually: ollama serve")
    
    def _pull_base_models(self) -> None:
        """Pull required base models"""
        logger.info("Pulling base models...")
        
        base_models = ["gemma2:2b"]
        
        for model in base_models:
            logger.info(f"Pulling {model}...")
            try:
                result = subprocess.run(['ollama', 'pull', model], 
                                       capture_output=True, text=True, check=True)
                logger.info(f"Successfully pulled {model}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to pull {model}: {e.stderr}")
                # Continue with other models
    
    def _create_oncoscope_models(self) -> None:
        """Create OncoScope-specific models"""
        logger.info("Creating OncoScope models...")
        
        for model_name, config in self.oncoscope_models.items():
            logger.info(f"Creating model: {model_name}")
            
            # Check if fine-tuned model exists
            fine_tuned_path = self.model_dir / config.get("gguf_path", "")
            modelfile_path = self.model_dir / config.get("model_file", f"{model_name}.modelfile")
            
            if fine_tuned_path.exists() and modelfile_path.exists():
                # Use fine-tuned model
                logger.info(f"Using fine-tuned model from {fine_tuned_path}")
                self._create_model_from_file(model_name, modelfile_path)
            else:
                # Create model from base model with custom configuration
                logger.info(f"Creating model from base: {config['base_model']}")
                self._create_model_from_config(model_name, config)
    
    def _create_model_from_file(self, model_name: str, modelfile_path: Path) -> None:
        """Create model from existing modelfile"""
        try:
            result = subprocess.run(['ollama', 'create', model_name, '-f', str(modelfile_path)],
                                   capture_output=True, text=True, check=True)
            logger.info(f"Successfully created {model_name} from modelfile")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create {model_name}: {e.stderr}")
    
    def _create_model_from_config(self, model_name: str, config: Dict) -> None:
        """Create model from configuration"""
        # Create temporary modelfile
        modelfile_content = self._generate_modelfile(config)
        temp_modelfile = Path(f"/tmp/{model_name}.modelfile")
        
        with open(temp_modelfile, 'w') as f:
            f.write(modelfile_content)
        
        try:
            result = subprocess.run(['ollama', 'create', model_name, '-f', str(temp_modelfile)],
                                   capture_output=True, text=True, check=True)
            logger.info(f"Successfully created {model_name}")
            
            # Clean up
            temp_modelfile.unlink()
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create {model_name}: {e.stderr}")
            temp_modelfile.unlink(missing_ok=True)
    
    def _generate_modelfile(self, config: Dict) -> str:
        """Generate modelfile content from configuration"""
        base_model = config['base_model']
        system_prompt = config['system_prompt']
        parameters = config.get('parameters', {})
        
        modelfile = f"""FROM {base_model}

SYSTEM \"\"\"{system_prompt}\"\"\"

"""
        
        # Add parameters
        for param, value in parameters.items():
            modelfile += f"PARAMETER {param} {value}\n"
        
        # Add template
        modelfile += """
TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
\"\"\"
"""
        
        return modelfile
    
    def _validate_models(self) -> None:
        """Validate created models"""
        logger.info("Validating OncoScope models...")
        
        for model_name in self.oncoscope_models.keys():
            if self._test_model(model_name):
                logger.info(f" {model_name} is working correctly")
            else:
                logger.warning(f"  {model_name} may have issues")
    
    def _test_model(self, model_name: str) -> bool:
        """Test a specific model"""
        test_prompt = "Analyze the BRCA1 c.68_69delAG mutation."
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": test_prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'response' in result and len(result['response']) > 10:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error testing {model_name}: {e}")
            return False
    
    def _create_helper_scripts(self) -> None:
        """Create helper scripts for OncoScope"""
        logger.info("Creating helper scripts...")
        
        scripts_dir = Path(__file__).parent
        
        # Create run_oncoscope.py
        run_script = scripts_dir / "run_oncoscope.py"
        self._create_run_script(run_script)
        
        # Create model_manager.py
        manager_script = scripts_dir / "model_manager.py"
        self._create_model_manager_script(manager_script)
        
        # Create demo script
        demo_script = scripts_dir / "demo_analysis.py"
        self._create_demo_script(demo_script)
        
        logger.info("Helper scripts created")
    
    def _create_run_script(self, script_path: Path) -> None:
        """Create run_oncoscope.py script"""
        content = '''#!/usr/bin/env python3
"""
OncoScope Runner Script
Quick interface to OncoScope models
"""

import requests
import json
import argparse

def run_oncoscope_analysis(mutation: str, model: str = "oncoscope-gemma-3n"):
    """Run OncoScope analysis on a mutation"""
    
    prompt = f"Analyze the cancer mutation {mutation}. Provide comprehensive analysis including pathogenicity, clinical significance, and therapeutic implications."
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        return result.get('response', 'No response')
    else:
        return f"Error: {response.status_code}"

def main():
    parser = argparse.ArgumentParser(description="Run OncoScope analysis")
    parser.add_argument("mutation", help="Mutation to analyze (e.g., BRCA1:c.68_69delAG)")
    parser.add_argument("--model", default="oncoscope-gemma-3n", help="Model to use")
    
    args = parser.parse_args()
    
    print(f"Analyzing {args.mutation} with {args.model}...")
    print("=" * 50)
    
    result = run_oncoscope_analysis(args.mutation, args.model)
    print(result)

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(content)
        script_path.chmod(0o755)
    
    def _create_model_manager_script(self, script_path: Path) -> None:
        """Create model manager script"""
        content = '''#!/usr/bin/env python3
"""
OncoScope Model Manager
Manage OncoScope Ollama models
"""

import subprocess
import requests
import json
import argparse

def list_models():
    """List available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model in models.get('models', []):
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"  {name} ({size:.1f} GB)")
        else:
            print("Error fetching models")
    except Exception as e:
        print(f"Error: {e}")

def delete_model(model_name: str):
    """Delete a model"""
    try:
        result = subprocess.run(['ollama', 'rm', model_name], 
                               capture_output=True, text=True, check=True)
        print(f"Deleted {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error deleting {model_name}: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Manage OncoScope models")
    parser.add_argument("action", choices=["list", "delete"], help="Action to perform")
    parser.add_argument("--model", help="Model name (for delete action)")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_models()
    elif args.action == "delete":
        if not args.model:
            print("Error: --model required for delete action")
        else:
            delete_model(args.model)

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(content)
        script_path.chmod(0o755)
    
    def _create_demo_script(self, script_path: Path) -> None:
        """Create demo analysis script"""
        content = '''#!/usr/bin/env python3
"""
OncoScope Demo Analysis
Demonstrate OncoScope capabilities
"""

import requests
import json
import time

def demo_analysis():
    """Run demo analysis"""
    
    demo_mutations = [
        "BRCA1:c.68_69delAG",
        "TP53:c.524G>A", 
        "KRAS:c.34G>T",
        "EGFR:c.2573T>G"
    ]
    
    print(">ì OncoScope Demo Analysis")
    print("=" * 40)
    
    for mutation in demo_mutations:
        print(f"\\nAnalyzing: {mutation}")
        print("-" * 30)
        
        prompt = f"Analyze the cancer mutation {mutation}. Provide key clinical insights."
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "oncoscope-gemma-3n",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result.get('response', 'No response')
                print(analysis[:300] + "..." if len(analysis) > 300 else analysis)
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(1)  # Brief pause between analyses

if __name__ == "__main__":
    demo_analysis()
'''
        
        with open(script_path, 'w') as f:
            f.write(content)
        script_path.chmod(0o755)
    
    def _get_oncoscope_system_prompt(self) -> str:
        """Get OncoScope system prompt"""
        return """You are OncoScope, an advanced AI system specialized in cancer genomics analysis. You provide expert-level mutation interpretation, risk assessment, and clinical recommendations based on the latest scientific knowledge in precision oncology.

Your expertise includes:
- Cancer mutation pathogenicity assessment using ACMG/AMP criteria
- Therapeutic implications and targeted therapy recommendations
- Hereditary cancer risk evaluation and genetic counseling
- Clinical trial matching and treatment sequencing
- Multi-mutation interaction analysis
- Population genetics and penetrance calculations

Always provide evidence-based, clinically actionable insights while acknowledging limitations and recommending genetic counseling when appropriate. Format responses clearly with sections for pathogenicity, clinical significance, therapeutic implications, and recommendations."""
    
    def _get_analyzer_system_prompt(self) -> str:
        """Get analyzer system prompt"""
        return """You are a specialized cancer genomics analyzer focused on mutation interpretation and pathogenicity assessment. Provide detailed technical analysis of cancer mutations including:

- Molecular mechanism and functional impact
- Pathogenicity classification using ACMG/AMP criteria
- Cancer association and penetrance data
- Variant allele frequency and population genetics
- Structural and functional domain analysis

Maintain high scientific rigor and cite evidence levels. Focus on technical accuracy over clinical recommendations."""
    
    def _get_reporter_system_prompt(self) -> str:
        """Get reporter system prompt"""
        return """You are a clinical report generator specialized in cancer genomics. Create comprehensive, professional clinical reports that include:

- Executive summary with key findings
- Detailed mutation analysis and clinical significance
- Risk assessment and penetrance calculations
- Therapeutic recommendations and clinical trials
- Genetic counseling recommendations
- Family implications and cascade testing

Use appropriate medical terminology and maintain professional clinical report formatting. Include disclaimers and limitations appropriately."""

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Setup Ollama for OncoScope")
    parser.add_argument("--model-dir", default="../ai/fine_tuning", help="Model directory")
    parser.add_argument("--ollama-host", default="localhost:11434", help="Ollama host")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup = OncoScopeOllamaSetup(
        model_dir=args.model_dir,
        ollama_host=args.ollama_host
    )
    
    if setup.setup_ollama_environment():
        print(" OncoScope Ollama setup completed successfully!")
        print("=€ You can now use OncoScope models:")
        for model_name, config in setup.oncoscope_models.items():
            print(f"  - {model_name}: {config['description']}")
        print("\\n=Ú Helper scripts created:")
        print("  - run_oncoscope.py: Quick analysis interface")
        print("  - model_manager.py: Manage models")
        print("  - demo_analysis.py: Demo OncoScope capabilities")
    else:
        print("L OncoScope Ollama setup failed!")
        print("Please check the logs and try again.")


if __name__ == "__main__":
    main()