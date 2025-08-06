# OncoScope Architecture Diagrams

This directory contains architectural diagrams for the OncoScope precision oncology platform.

## Diagrams

### 1. System Architecture Overview (`01_system_architecture`)
Shows the complete system architecture with four main layers:
- Frontend Layer (Electron)
- Backend Layer (FastAPI)
- Data Layer (SQLite)
- AI Layer (Ollama/Gemma 3n)

### 2. Data Flow Pipeline (`02_data_flow_pipeline`)
Illustrates the complete data processing pipeline from user input to clinical output:
- Input Stage
- Processing Stage
- Analysis Stage
- Output Stage

### 3. AI Model Architecture (`03_ai_model_architecture`)
Details the Gemma 3n integration and fine-tuning approach:
- Input processing
- 2B parameter model architecture
- LoRA fine-tuning
- Structured output generation

### 4. Technology Stack (`04_technology_stack`)
Comprehensive view of all technologies used:
- Presentation Layer
- Application Layer
- Business Logic
- Data Access
- AI Integration
- Infrastructure

### 5. Microservices Communication (`05_microservices_communication`)
Shows how different services interact:
- API Gateway pattern
- Service communication
- External service integration

### 6. Deployment Architecture (`06_deployment_architecture`)
End-to-end deployment process:
- Development environment
- Build process
- Distribution formats
- End-user setup

### 7. Mutation Analysis Pipeline (`07_mutation_analysis_pipeline`)
Detailed mutation processing workflow:
- Input parsing
- Database enrichment
- AI enhancement
- Risk assessment
- Clinical output

### 8. Security Architecture (`08_security_architecture`)
Privacy and security measures:
- Input validation
- Privacy protection
- Access control
- Data security

## File Formats

Each diagram is available in three formats:
- `.mmd` - Mermaid source files (version controlled)
- `.svg` - Scalable vector graphics (for presentations)
- `.png` - Raster images (for documentation)

## Generating Diagrams

To convert Mermaid files to SVG/PNG:

```bash
# Install mermaid-cli if not already installed
npm install -g @mermaid-js/mermaid-cli

# Run the conversion script
python convert_diagrams.py
```

## Using in Presentations

### For Slides
- Use SVG files for best quality at any size
- PNG files work well for PowerPoint/Keynote

### For Documentation
- Embed Mermaid source directly in Markdown
- Use PNG files for README files

### For Web
- SVG files are recommended for responsive design
- Can be styled with CSS

## Color Scheme

The diagrams use a consistent color palette:
- Primary: `#0066ff` (OncoScope Blue)
- Secondary: `#FF6B6B` (Alert Red)
- Success: `#4ECDC4` (Teal)
- Warning: `#F6AE2D` (Orange)
- Dark: `#2F4858` (Dark Blue)

## Editing

To edit diagrams:
1. Modify the `.mmd` files
2. Run `python convert_diagrams.py` to regenerate SVG/PNG
3. Commit all changes

## License

These diagrams are part of the OncoScope project and subject to the same license terms.