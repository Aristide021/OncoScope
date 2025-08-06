#!/usr/bin/env python3
"""
Convert Mermaid diagrams to SVG and PNG formats
Requires: mmdc (mermaid-cli) to be installed
Install with: npm install -g @mermaid-js/mermaid-cli
"""

import os
import subprocess
import glob

def convert_mermaid_to_formats():
    """Convert all .mmd files to SVG and PNG"""
    
    # Get all mermaid files
    mmd_files = glob.glob("*.mmd")
    
    if not mmd_files:
        print("No .mmd files found in current directory")
        return
    
    print(f"Found {len(mmd_files)} Mermaid diagrams to convert")
    
    for mmd_file in mmd_files:
        base_name = os.path.splitext(mmd_file)[0]
        
        # Convert to SVG
        svg_file = f"{base_name}.svg"
        print(f"Converting {mmd_file} to {svg_file}...")
        try:
            subprocess.run([
                "mmdc",
                "-i", mmd_file,
                "-o", svg_file,
                "-t", "dark",
                "-b", "#1a1a1a"
            ], check=True)
            print(f"✓ Created {svg_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create {svg_file}: {e}")
        except FileNotFoundError:
            print("✗ mmdc not found. Install with: npm install -g @mermaid-js/mermaid-cli")
            return
        
        # Convert to PNG
        png_file = f"{base_name}.png"
        print(f"Converting {mmd_file} to {png_file}...")
        try:
            subprocess.run([
                "mmdc",
                "-i", mmd_file,
                "-o", png_file,
                "-t", "dark",
                "-b", "#1a1a1a",
                "-w", "2048",
                "-H", "1536"
            ], check=True)
            print(f"✓ Created {png_file}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create {png_file}: {e}")
    
    print("\nConversion complete!")
    
    # List all generated files
    print("\nGenerated files:")
    for pattern in ["*.mmd", "*.svg", "*.png"]:
        files = glob.glob(pattern)
        if files:
            print(f"\n{pattern}:")
            for f in sorted(files):
                print(f"  - {f}")

if __name__ == "__main__":
    # Change to diagrams directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    convert_mermaid_to_formats()