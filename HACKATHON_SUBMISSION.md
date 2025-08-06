# OncoScope: Privacy-First Cancer Genomics Analysis with Gemma 3n

## Abstract

OncoScope addresses a critical bottleneck in cancer care: the slow, costly, and privacy-compromising process of genomic analysis. We have developed a privacy-first, on-device desktop application that empowers clinicians to perform sophisticated cancer mutation analysis in 20-40 seconds, not weeks. Our solution leverages a Gemma 3n E4B model, which we transformed into a domain-specific expert using Unsloth for hyper-efficient fine-tuning on a curated dataset of 5,998 cancer genomics examples from ClinVar, COSMIC Top 50, and expert-curated sources. The entire AI engine runs locally via Ollama, guaranteeing that sensitive patient genetic data never leaves the clinician's device. The result is a powerful, interactive tool that provides immediate, actionable insights, demonstrating a tangible path toward making personalized medicine more accessible, secure, and immediate for patients everywhere.

## The Challenge: Bridging the Gap in Precision Oncology

Precision oncology promises to tailor cancer treatment to a patient's unique genetic profile. However, the practical application of this promise faces three significant barriers:

### 1. Time
Analyzing a patient's genomic data can take days or weeks, involving complex bioinformatics pipelines and multiple specialists. This delay can be critical for patients with aggressive cancers.

### 2. Privacy & Security
Genetic data is the most personal information a patient has. Uploading this data to third-party cloud services for analysis introduces significant privacy risks and compliance hurdles (e.g., HIPAA).

### 3. Cost & Accessibility
Cloud-based AI analysis and specialized bioinformatics software can be prohibitively expensive, limiting access for smaller clinics, researchers, or healthcare systems in low-resource settings.

**Our mission was to solve these challenges by leveraging the unique on-device capabilities of Google's Gemma 3n model.**

## Our Solution: OncoScope - An On-Device Genomics Platform

OncoScope is a fully-functional, cross-platform desktop application that brings the power of a genomic analysis lab directly to a clinician's or researcher's computer.

### The User Experience

The user experience is simple and powerful:

1. The user opens the OncoScope application
2. They can either upload a standard mutation data file (e.g., VCF, CSV) or paste a list of mutations directly into the interface
3. Optionally, they can add non-identifying patient context, such as age or cancer type, for a more personalized analysis
4. Upon clicking "Analyze," the backend server, running locally, processes the request
5. The fine-tuned Gemma 3n model, served by Ollama, performs a deep analysis of the mutations
6. Within 20-40 seconds, a rich, interactive report is displayed in the UI, detailing each mutation's pathogenicity, associated cancer risks, and potential targeted therapies, along with an overall risk assessment

**Crucially, this entire workflow happens 100% offline and on-device.**

## System Architecture

OncoScope is engineered as a robust, full-stack application with a clear separation of concerns. This architecture ensures stability, scalability, and a seamless user experience.

### Frontend (Electron App)

- **Technology**: Built with standard HTML, CSS, and JavaScript, packaged in Electron v25.3.0
- **Role**: Provides the cross-platform graphical user interface (GUI). It handles user input, renders the interactive results dashboard, and communicates with the local backend API
- **Justification**: Electron was chosen to create a native-like desktop experience without the overhead of platform-specific codebases. It allows for a modern, web-based UI while having the necessary access to the local system to interact with the backend

### Backend (FastAPI Server)

- **Technology**: A high-performance asynchronous Python server built with FastAPI v0.104.1
- **Role**: Acts as the central nervous system of the application. It exposes a REST API that the frontend consumes. It is responsible for parsing requests, orchestrating the analysis workflow, calling the Ollama client, and structuring the final report
- **Justification**: FastAPI's async capabilities are perfect for handling potentially long-running AI inference tasks without blocking the application. Its Pydantic integration ensures that all data flowing through the system is strongly typed and validated

### AI Core (Gemma 3n + Ollama)

- **Technology**: The fine-tuned Gemma 3n E4B model, served locally and managed by Ollama
- **Role**: This is the analytical engine. The `ollama_client.py` in our backend communicates with the Ollama server, sending structured prompts and receiving detailed JSON-formatted genomic analysis in return
- **Justification**: This is the cornerstone of our privacy-first architecture. Ollama provides an incredibly simple and robust way to manage and serve LLMs locally, making it the perfect choice for deploying our specialized Gemma 3n model

### Data Flow

```
User Action (Electron UI) → HTTP Request → FastAPI Backend → 
ollama_client.py → Ollama Server (Gemma 3n) → JSON Response → 
FastAPI Backend → HTTP Response → Rendered Report (Electron UI)
```

## The AI Engine: Specializing Gemma 3n for Genomic Analysis

Our primary technical achievement was transforming the general-purpose Gemma 3n model into a specialized expert for cancer genomics.

### A. Why Gemma 3n?

Gemma 3n was the ideal foundation for this project due to its design philosophy:

- **On-Device Performance**: Its MatFormer architecture is optimized for high performance on consumer hardware, which is essential for a desktop application
- **Privacy-First**: The ability to run locally is not an afterthought; it's a core feature, which perfectly aligns with our project's primary goal
- **State-of-the-Art Architecture**: As a next-generation model with Multi-Query Attention (MQA), it provided a powerful and efficient base for fine-tuning

### B. Fine-Tuning with Unsloth

To achieve expert-level performance, we fine-tuned Gemma 3n E4B using Unsloth. This process is detailed in our `ai/fine_tuning/train_cancer_model.py` script.

- **The Dataset**: We curated a mixed dataset of 5,998 high-quality examples (5,398 training + 600 evaluation), combining data from:
  - ClinVar targeted variants
  - COSMIC Top 50 validated mutations
  - Expert-curated clinical examples
  
- **The Process**: We leveraged Unsloth's FastModel for memory-efficient LoRA (Low-Rank Adaptation) fine-tuning with rank 32. By using 4-bit quantization and Unsloth's optimizations, we were able to fine-tune the model on an NVIDIA A100 GPU with 42.5GB VRAM

- **The Outcome**: The result is a model that not only understands the syntax of genomics but also the semantics of clinical relevance, consistently providing accurate, structured, and reliable output

### C. Local Deployment with Ollama

The fine-tuned model is deployed locally using Ollama:

- **Integration**: Our `backend/mutation_analyzer.py` uses a dedicated `ai/inference/ollama_client.py` to interact with the Ollama API
- **Robustness**: We created a custom Modelfile to configure the model's system prompt, parameters (like temperature), and to point to the GGUF-quantized model weights generated by Unsloth
- **Core Functionality**: This local deployment is what enables the core promise of OncoScope: powerful AI analysis with zero data transmission to the cloud

### D. Advanced Prompt Engineering

As seen in `ai/inference/prompts.py`, we developed a sophisticated prompt library. We instruct the model to analyze mutations in the context of existing diagnoses and provide its response in a strictly-defined JSON schema. This structured prompting is key to getting reliable, parsable data from the LLM on every run.

## Overcoming Technical Challenges

### Challenge 1: VRAM Limitations during Fine-Tuning

- **Problem**: Fine-tuning even a 4B effective parameter model can be memory-intensive
- **Solution**: Unsloth's 4-bit quantization and optimized training kernels significantly reduced VRAM usage, making the process feasible on available hardware

### Challenge 2: Ensuring Reliable, Structured JSON Output

- **Problem**: LLMs can sometimes fail to adhere strictly to JSON formats, adding conversational text that breaks parsing
- **Solution**: We implemented a three-part strategy:
  1. **Strong Prompting**: Explicitly instructing the model in the prompt to only return a JSON object
  2. **Ollama's JSON Mode**: Using the `format: "json"` parameter in our API calls to Ollama
  3. **Robust Parsing**: Our backend code includes logic to clean and parse the response, with fallbacks in case of failure

### Challenge 3: Full-Stack Integration & Performance

- **Problem**: Orchestrating the frontend, backend, and the Ollama server to start and communicate correctly, while ensuring the analysis felt responsive
- **Solution**: We used health checks from the frontend to the backend to ensure the system was ready before allowing analysis. The asynchronous nature of FastAPI and the high performance of the Gemma 3n model meant that even complex analyses of multiple mutations complete in 20-40 seconds, providing a responsive user experience

## Key Features Demonstrated

- ✅ **Multi-Mutation Analysis**: Analyzes interactions between multiple genetic variants
- ✅ **Context-Aware Results**: Considers patient's existing diagnosis, not just mutations
- ✅ **FDA Drug Matching**: Links mutations to approved targeted therapies
- ✅ **Risk Visualization**: Interactive charts showing mutation impact
- ✅ **Educational Loading**: Cancer facts displayed during analysis
- ✅ **Professional UI/UX**: Polished interface suitable for clinical use

## Performance Metrics

| Metric | Value |
|--------|-------|
| Analysis Time | 20-40 seconds |
| Dataset Size | 5,998 examples |
| Training Hardware | NVIDIA A100 (42.5GB VRAM) |
| LoRA Rank | 32 |
| Learning Rate | 1e-4 |
| Model Size | Gemma 3n E4B (4B effective parameters) |

## Conclusion & Future Vision

OncoScope is a working, end-to-end proof-of-concept that demonstrates the immense potential of on-device AI to solve meaningful, real-world problems. By leveraging the power of Gemma 3n, the efficiency of Unsloth, and the simplicity of Ollama, we have built a tool that is not just technically innovative but also impactful, secure, and accessible.

This project is more than a hackathon submission; it is the foundation for a new paradigm in clinical decision support. Our future vision includes:

### Immediate Next Steps
- **Q4 Quantization**: Further optimize for performance and portability
- **EHR Integration**: Build plugins for electronic health record systems
- **Clinical Validation**: Partner with cancer centers for real-world testing

### Long-term Vision
- **Expanding Multimodality**: Incorporating Gemma 3n's image understanding capabilities to analyze pathology slide images alongside genetic data
- **Mobile Deployment**: Adapting the architecture to run on tablets and mobile devices for even greater accessibility
- **Global Impact**: Making precision oncology accessible to every hospital, not just major cancer centers

**We believe OncoScope is a compelling demonstration of the Gemma 3n Impact Challenge's core mission: building real products for a better world.**

---

### 🔗 Links & Resources

- **Repository**: [GitHub - OncoScope](https://github.com/yourusername/oncoscope)
- **Demo Video**: [Watch on YouTube](https://youtube.com/demo)
- **Contact**: aristide021@gmail.com

### 🏆 Built with ❤️ and Gemma 3n for the Google Gemma 3n Impact Challenge 2025