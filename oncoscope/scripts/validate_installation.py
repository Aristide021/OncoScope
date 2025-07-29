"""
OncoScope Installation Validator
Comprehensive validation of OncoScope installation and dependencies
"""

import os
import sys
import json
import subprocess
import importlib
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OncoScopeValidator:
    """Validate OncoScope installation and configuration"""
    
    def __init__(self, oncoscope_root: str = ".."):
        """Initialize validator"""
        self.oncoscope_root = Path(__file__).parent / oncoscope_root
        self.validation_results = {}
        
        # Expected directory structure
        self.expected_structure = {
            'backend': ['risk_calculator.py', 'report_generator.py', 'clustering_engine.py'],
            'ai/inference': ['prompts.py'],
            'ai/fine_tuning': ['train_cancer_model.py', 'prepare_dataset.py'],
            'data': ['clinvar_variants.json', 'cosmic_mutations.json', 'drug_associations.json', 'cancer_types.json'],
            'scripts': ['create_demo_data.py', 'download_cosmic.py', 'setup_ollama.py', 'validate_installation.py']
        }
        
        # Required Python packages
        self.required_packages = {
            'core': ['numpy', 'pandas', 'scipy', 'scikit-learn'],
            'ml': ['torch', 'transformers'],
            'optional': ['unsloth', 'datasets', 'trl'],
            'bio': ['pysam', 'cyvcf2'],
            'web': ['requests', 'flask'],
            'viz': ['matplotlib', 'seaborn'],
            'utils': ['pyyaml', 'tqdm', 'click']
        }
        
        # System requirements
        self.system_requirements = {
            'python_version': (3, 8),
            'min_memory_gb': 8,
            'recommended_memory_gb': 16
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("= Starting OncoScope installation validation...")
        
        validation_steps = [
            ('System Requirements', self._validate_system_requirements),
            ('Directory Structure', self._validate_directory_structure),
            ('Core Files', self._validate_core_files),
            ('Data Files', self._validate_data_files),
            ('Python Dependencies', self._validate_python_dependencies),
            ('Import Tests', self._validate_imports),
            ('Configuration', self._validate_configuration),
            ('External Services', self._validate_external_services),
            ('Functional Tests', self._validate_functionality)
        ]
        
        overall_status = True
        
        for step_name, validator_func in validation_steps:
            logger.info(f"Validating: {step_name}")
            try:
                result = validator_func()
                self.validation_results[step_name] = result
                
                if result['status'] == 'PASS':
                    logger.info(f" {step_name}: PASSED")
                elif result['status'] == 'WARN':
                    logger.warning(f"  {step_name}: WARNING - {result.get('message', '')}")
                else:
                    logger.error(f"L {step_name}: FAILED - {result.get('message', '')}")
                    overall_status = False
                    
            except Exception as e:
                logger.error(f"L {step_name}: ERROR - {e}")
                self.validation_results[step_name] = {'status': 'ERROR', 'message': str(e)}
                overall_status = False
        
        # Generate summary
        summary = self._generate_validation_summary(overall_status)
        self.validation_results['summary'] = summary
        
        # Save validation report
        self._save_validation_report()
        
        return self.validation_results
    
    def _validate_system_requirements(self) -> Dict[str, Any]:
        """Validate system requirements"""
        issues = []
        
        # Python version check
        current_python = sys.version_info
        required_python = self.system_requirements['python_version']
        
        if current_python < required_python:
            issues.append(f"Python {required_python[0]}.{required_python[1]}+ required, found {current_python[0]}.{current_python[1]}")
        
        # Memory check (rough estimate)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb < self.system_requirements['min_memory_gb']:
                issues.append(f"Minimum {self.system_requirements['min_memory_gb']}GB RAM required, found {memory_gb:.1f}GB")
            elif memory_gb < self.system_requirements['recommended_memory_gb']:
                issues.append(f"Recommended {self.system_requirements['recommended_memory_gb']}GB RAM, found {memory_gb:.1f}GB")
        except ImportError:
            issues.append("Cannot check memory requirements (psutil not available)")
        
        # Platform check
        platform_info = {
            'platform': sys.platform,
            'python_version': f"{current_python[0]}.{current_python[1]}.{current_python[2]}",
            'architecture': 'x64' if sys.maxsize > 2**32 else 'x32'
        }
        
        status = 'FAIL' if issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'System requirements met',
            'details': platform_info,
            'issues': issues
        }
    
    def _validate_directory_structure(self) -> Dict[str, Any]:
        """Validate directory structure"""
        missing_dirs = []
        missing_files = []
        
        for directory, files in self.expected_structure.items():
            dir_path = self.oncoscope_root / directory
            
            if not dir_path.exists():
                missing_dirs.append(directory)
                continue
            
            for file in files:
                file_path = dir_path / file
                if not file_path.exists():
                    missing_files.append(f"{directory}/{file}")
        
        issues = []
        if missing_dirs:
            issues.extend([f"Missing directory: {d}" for d in missing_dirs])
        if missing_files:
            issues.extend([f"Missing file: {f}" for f in missing_files])
        
        status = 'FAIL' if issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'Directory structure is correct',
            'missing_directories': missing_dirs,
            'missing_files': missing_files
        }
    
    def _validate_core_files(self) -> Dict[str, Any]:
        """Validate core Python files"""
        core_files_status = {}
        issues = []
        
        core_files = [
            'backend/risk_calculator.py',
            'backend/report_generator.py', 
            'backend/clustering_engine.py',
            'ai/inference/prompts.py',
            'ai/fine_tuning/train_cancer_model.py',
            'ai/fine_tuning/prepare_dataset.py'
        ]
        
        for file_path in core_files:
            full_path = self.oncoscope_root / file_path
            
            if not full_path.exists():
                core_files_status[file_path] = 'MISSING'
                issues.append(f"Missing: {file_path}")
                continue
            
            # Check if file is not empty
            if full_path.stat().st_size == 0:
                core_files_status[file_path] = 'EMPTY'
                issues.append(f"Empty file: {file_path}")
                continue
            
            # Try to parse as Python (basic syntax check)
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                compile(content, str(full_path), 'exec')
                core_files_status[file_path] = 'VALID'
            except SyntaxError as e:
                core_files_status[file_path] = 'SYNTAX_ERROR'
                issues.append(f"Syntax error in {file_path}: {e}")
            except Exception as e:
                core_files_status[file_path] = 'ERROR'
                issues.append(f"Error reading {file_path}: {e}")
        
        status = 'FAIL' if issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'All core files are valid',
            'file_status': core_files_status
        }
    
    def _validate_data_files(self) -> Dict[str, Any]:
        """Validate data files"""
        data_files_status = {}
        issues = []
        
        data_files = [
            'clinvar_variants.json',
            'cosmic_mutations.json',
            'drug_associations.json',
            'cancer_types.json'
        ]
        
        for file_name in data_files:
            file_path = self.oncoscope_root / 'data' / file_name
            
            if not file_path.exists():
                data_files_status[file_name] = 'MISSING'
                issues.append(f"Missing data file: {file_name}")
                continue
            
            # Check if file is not empty
            if file_path.stat().st_size == 0:
                data_files_status[file_name] = 'EMPTY'
                issues.append(f"Empty data file: {file_name}")
                continue
            
            # Try to parse as JSON
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Basic validation
                if not data:
                    data_files_status[file_name] = 'EMPTY_JSON'
                    issues.append(f"Empty JSON data: {file_name}")
                else:
                    data_files_status[file_name] = 'VALID'
                    
                    # File-specific validation
                    if file_name == 'clinvar_variants.json':
                        self._validate_clinvar_data(data, issues)
                    elif file_name == 'cosmic_mutations.json':
                        self._validate_cosmic_data(data, issues)
                        
            except json.JSONDecodeError as e:
                data_files_status[file_name] = 'INVALID_JSON'
                issues.append(f"Invalid JSON in {file_name}: {e}")
            except Exception as e:
                data_files_status[file_name] = 'ERROR'
                issues.append(f"Error reading {file_name}: {e}")
        
        status = 'FAIL' if issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'All data files are valid',
            'file_status': data_files_status
        }
    
    def _validate_python_dependencies(self) -> Dict[str, Any]:
        """Validate Python package dependencies"""
        package_status = {}
        missing_core = []
        missing_optional = []
        
        for category, packages in self.required_packages.items():
            for package in packages:
                try:
                    importlib.import_module(package)
                    package_status[package] = 'INSTALLED'
                except ImportError:
                    package_status[package] = 'MISSING'
                    
                    if category in ['core', 'ml']:
                        missing_core.append(package)
                    else:
                        missing_optional.append(package)
        
        issues = []
        if missing_core:
            issues.extend([f"Missing core package: {p}" for p in missing_core])
        
        status = 'FAIL' if missing_core else ('WARN' if missing_optional else 'PASS')
        message = '; '.join(issues) if issues else 'All required packages installed'
        
        if missing_optional:
            message += f" (Optional packages missing: {', '.join(missing_optional)})"
        
        return {
            'status': status,
            'message': message,
            'package_status': package_status,
            'missing_core': missing_core,
            'missing_optional': missing_optional
        }
    
    def _validate_imports(self) -> Dict[str, Any]:
        """Validate imports of OncoScope modules"""
        import_results = {}
        issues = []
        
        # Add OncoScope root to path temporarily
        sys.path.insert(0, str(self.oncoscope_root))
        
        modules_to_test = [
            'oncoscope.backend.risk_calculator',
            'oncoscope.backend.report_generator',
            'oncoscope.backend.clustering_engine',
            'oncoscope.ai.inference.prompts'
        ]
        
        for module_name in modules_to_test:
            try:
                module = importlib.import_module(module_name)
                import_results[module_name] = 'SUCCESS'
                
                # Basic validation - check for expected classes/functions
                if 'risk_calculator' in module_name:
                    if not hasattr(module, 'CancerRiskCalculator'):
                        issues.append(f"{module_name}: Missing CancerRiskCalculator class")
                elif 'report_generator' in module_name:
                    if not hasattr(module, 'ClinicalReportGenerator'):
                        issues.append(f"{module_name}: Missing ClinicalReportGenerator class")
                elif 'clustering_engine' in module_name:
                    if not hasattr(module, 'CancerMutationClusterer'):
                        issues.append(f"{module_name}: Missing CancerMutationClusterer class")
                elif 'prompts' in module_name:
                    if not hasattr(module, 'GenomicAnalysisPrompts'):
                        issues.append(f"{module_name}: Missing GenomicAnalysisPrompts class")
                        
            except ImportError as e:
                import_results[module_name] = f'IMPORT_ERROR: {e}'
                issues.append(f"Cannot import {module_name}: {e}")
            except Exception as e:
                import_results[module_name] = f'ERROR: {e}'
                issues.append(f"Error testing {module_name}: {e}")
        
        # Remove from path
        sys.path.remove(str(self.oncoscope_root))
        
        status = 'FAIL' if issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'All modules import successfully',
            'import_results': import_results
        }
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files"""
        config_issues = []
        
        # Check for configuration files
        config_files = ['config.yaml', 'settings.json', '.env']
        found_configs = []
        
        for config_file in config_files:
            config_path = self.oncoscope_root / config_file
            if config_path.exists():
                found_configs.append(config_file)
        
        # This is optional, so just warn if none found
        if not found_configs:
            config_issues.append("No configuration files found (optional)")
        
        status = 'WARN' if config_issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(config_issues) if config_issues else 'Configuration OK',
            'found_configs': found_configs
        }
    
    def _validate_external_services(self) -> Dict[str, Any]:
        """Validate external service connectivity"""
        service_status = {}
        issues = []
        
        # Check Ollama
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                service_status['ollama'] = 'RUNNING'
            else:
                service_status['ollama'] = f'ERROR_{response.status_code}'
                issues.append(f"Ollama service error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            service_status['ollama'] = 'NOT_RUNNING'
            issues.append("Ollama service not running")
        except Exception as e:
            service_status['ollama'] = f'ERROR: {e}'
            issues.append(f"Ollama check failed: {e}")
        
        # Check internet connectivity (for updates)
        try:
            response = requests.get('https://httpbin.org/status/200', timeout=5)
            service_status['internet'] = 'CONNECTED' if response.status_code == 200 else 'LIMITED'
        except:
            service_status['internet'] = 'NO_CONNECTION'
            issues.append("No internet connection (may affect updates)")
        
        # Services are optional for basic functionality
        status = 'WARN' if issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'External services OK',
            'service_status': service_status
        }
    
    def _validate_functionality(self) -> Dict[str, Any]:
        """Validate basic functionality"""
        functionality_tests = {}
        issues = []
        
        # Add OncoScope to path
        sys.path.insert(0, str(self.oncoscope_root))
        
        try:
            # Test risk calculator
            from oncoscope.backend.risk_calculator import CancerRiskCalculator
            calculator = CancerRiskCalculator()
            
            test_mutation = {
                'gene': 'BRCA1',
                'variant': 'c.68_69delAG',
                'pathogenicity_score': 0.95
            }
            
            # This should not raise an exception
            risk_result = calculator.calculate_basic_risk(test_mutation)
            functionality_tests['risk_calculator'] = 'WORKING'
            
        except Exception as e:
            functionality_tests['risk_calculator'] = f'ERROR: {e}'
            issues.append(f"Risk calculator test failed: {e}")
        
        try:
            # Test report generator
            from oncoscope.backend.report_generator import ClinicalReportGenerator
            generator = ClinicalReportGenerator()
            
            # Basic instantiation test
            functionality_tests['report_generator'] = 'WORKING'
            
        except Exception as e:
            functionality_tests['report_generator'] = f'ERROR: {e}'
            issues.append(f"Report generator test failed: {e}")
        
        # Remove from path
        sys.path.remove(str(self.oncoscope_root))
        
        status = 'FAIL' if issues else 'PASS'
        return {
            'status': status,
            'message': '; '.join(issues) if issues else 'Basic functionality working',
            'test_results': functionality_tests
        }
    
    def _validate_clinvar_data(self, data: Dict, issues: List[str]) -> None:
        """Validate ClinVar data structure"""
        if not isinstance(data, dict):
            issues.append("ClinVar data should be a dictionary")
            return
        
        required_genes = ['BRCA1', 'BRCA2', 'TP53']
        for gene in required_genes:
            if gene not in data:
                issues.append(f"Missing {gene} in ClinVar data")
    
    def _validate_cosmic_data(self, data: Dict, issues: List[str]) -> None:
        """Validate COSMIC data structure"""
        if not isinstance(data, dict):
            issues.append("COSMIC data should be a dictionary")
            return
        
        required_genes = ['KRAS', 'EGFR', 'TP53']
        for gene in required_genes:
            if gene not in data:
                issues.append(f"Missing {gene} in COSMIC data")
    
    def _generate_validation_summary(self, overall_status: bool) -> Dict[str, Any]:
        """Generate validation summary"""
        passed = sum(1 for r in self.validation_results.values() if r.get('status') == 'PASS')
        warned = sum(1 for r in self.validation_results.values() if r.get('status') == 'WARN')
        failed = sum(1 for r in self.validation_results.values() if r.get('status') in ['FAIL', 'ERROR'])
        
        return {
            'overall_status': 'PASS' if overall_status else 'FAIL',
            'total_checks': len(self.validation_results),
            'passed': passed,
            'warned': warned,
            'failed': failed,
            'timestamp': str(os.path.getctime(__file__))
        }
    
    def _save_validation_report(self) -> None:
        """Save validation report"""
        report_file = self.oncoscope_root / 'validation_report.json'
        
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        logger.info(f"Validation report saved to {report_file}")
    
    def print_summary(self) -> None:
        """Print validation summary"""
        summary = self.validation_results.get('summary', {})
        
        print("\n" + "="*60)
        print("= OncoScope Installation Validation Summary")
        print("="*60)
        
        overall = summary.get('overall_status', 'UNKNOWN')
        if overall == 'PASS':
            print(" Overall Status: PASSED")
        else:
            print("L Overall Status: FAILED")
        
        print(f"=Ê Total Checks: {summary.get('total_checks', 0)}")
        print(f" Passed: {summary.get('passed', 0)}")
        print(f"  Warnings: {summary.get('warned', 0)}")
        print(f"L Failed: {summary.get('failed', 0)}")
        
        print("\n=Ë Detailed Results:")
        print("-" * 40)
        
        for check_name, result in self.validation_results.items():
            if check_name == 'summary':
                continue
                
            status = result.get('status', 'UNKNOWN')
            message = result.get('message', '')
            
            if status == 'PASS':
                print(f" {check_name}")
            elif status == 'WARN':
                print(f"  {check_name}: {message}")
            else:
                print(f"L {check_name}: {message}")
        
        print("\n" + "="*60)
        
        if overall == 'PASS':
            print("<‰ OncoScope is ready to use!")
        else:
            print("=' Please address the failed checks before using OncoScope.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Validate OncoScope installation")
    parser.add_argument("--oncoscope-root", default="..", help="OncoScope root directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--report-only", action="store_true", help="Only generate report, no console output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    validator = OncoScopeValidator(oncoscope_root=args.oncoscope_root)
    results = validator.run_full_validation()
    
    if not args.report_only:
        validator.print_summary()
    
    # Exit with appropriate code
    overall_status = results.get('summary', {}).get('overall_status')
    sys.exit(0 if overall_status == 'PASS' else 1)


if __name__ == "__main__":
    main()