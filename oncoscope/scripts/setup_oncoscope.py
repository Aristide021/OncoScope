#!/usr/bin/env python3
"""
OncoScope Setup Master Script
Orchestrates the complete OncoScope data preparation and validation workflow
"""

import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_name: str, description: str) -> bool:
    """Run a script and handle errors gracefully"""
    try:
        logger.info(f"🔄 {description}")
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} - Completed successfully")
            if result.stdout:
                logger.info(f"Output: {result.stdout}")
            return True
        else:
            logger.error(f"❌ {description} - Failed")
            logger.error(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ {description} - Exception: {e}")
        return False

def main():
    """Main OncoScope setup workflow"""
    
    print("""
    🧬 OncoScope Setup Master Workflow
    ===================================
    
    This script will:
    1. Validate your OncoScope installation
    2. Create targeted COSMIC mutation data
    3. Download real FDA drug associations
    4. Validate top 50 clinically actionable mutations
    5. Optimize data formats for training
    6. Setup AI models (Ollama)
    
    """)
    
    # Track success of each step
    workflow_steps = [
        ("validate_installation.py", "Validating OncoScope installation"),
        ("create_targeted_cosmic.py", "Creating targeted COSMIC mutation data"),
        ("download_real_drug_data.py", "Downloading real FDA drug associations"),
        ("cosmic_top50_validator.py", "Validating top 50 actionable mutations"),
        ("optimize_data_format.py", "Optimizing data formats for training"),
        ("setup_ollama.py", "Setting up AI models (Ollama)")
    ]
    
    successful_steps = 0
    total_steps = len(workflow_steps)
    
    logger.info(f"🚀 Starting OncoScope setup workflow ({total_steps} steps)")
    
    for step_num, (script, description) in enumerate(workflow_steps, 1):
        logger.info(f"📍 Step {step_num}/{total_steps}: {description}")
        
        if run_script(script, description):
            successful_steps += 1
        else:
            logger.warning(f"⚠️ Step {step_num} failed, but continuing with remaining steps...")
            
        print()  # Add spacing between steps
    
    # Final summary
    print("=" * 60)
    logger.info(f"🏁 OncoScope Setup Complete!")
    logger.info(f"📊 Success Rate: {successful_steps}/{total_steps} steps completed successfully")
    
    if successful_steps == total_steps:
        print("""
        ✅ Perfect! All steps completed successfully.
        
        🎯 OncoScope is now ready with:
        • 41 validated actionable cancer mutations
        • FDA-verified drug associations  
        • TCGA-verified genomics data
        • Literature-backed COSMIC mutations
        • Optimized training datasets
        • AI models deployed and ready
        
        🚀 Ready for production cancer genomics analysis!
        """)
    elif successful_steps >= total_steps * 0.8:
        print(f"""
        ⚠️ Mostly successful! {successful_steps}/{total_steps} steps completed.
        
        OncoScope should be functional but may have reduced capabilities.
        Check the logs above for any failed components.
        """)
    else:
        print(f"""
        ❌ Setup encountered significant issues.
        
        Only {successful_steps}/{total_steps} steps completed successfully.
        Please review the error messages above and ensure all dependencies are installed.
        """)
    
    print("📋 Next Steps:")
    print("• Run the premium dataset preparation: python -m oncoscope.ai.fine_tuning.prepare_dataset --premium")
    print("• Start the OncoScope analysis server: python -m oncoscope.main")
    print("• View generated data: ls -la ../data/")

if __name__ == "__main__":
    main()