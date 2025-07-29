#!/usr/bin/env python3
"""
OncoScope - Privacy-First Cancer Genomics Analysis Platform
Setup configuration for package installation and distribution
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = ""
if (this_directory / "README.md").exists():
    long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="oncoscope",
    version="1.0.0",
    author="OncoScope Team",
    author_email="contact@oncoscope.ai",
    description="Privacy-first cancer genomics analysis powered by AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oncoscope/oncoscope",
    
    packages=find_packages(),
    include_package_data=True,
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.4.14",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "oncoscope=oncoscope.backend.main:main",
            "oncoscope-server=oncoscope.backend.main:main",
            "oncoscope-validate=oncoscope.scripts.validate_installation:main",
        ],
    },
    
    package_data={
        "oncoscope": [
            "data/*.json",
            "frontend/*.html",
            "frontend/*.css",
            "frontend/*.js",
            "ai/fine_tuning/*.json",
            "ai/fine_tuning/*.txt",
        ],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/oncoscope/oncoscope/issues",
        "Documentation": "https://docs.oncoscope.ai",
        "Source": "https://github.com/oncoscope/oncoscope",
    },
    
    keywords="cancer genomics bioinformatics ai machine-learning healthcare privacy",
    zip_safe=False,
)