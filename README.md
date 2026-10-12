# Agentic Auditor

![License: MIT](https://img.shields.io/badge/License-MIT-yellow)
![MLX](https://img.shields.io/badge/Local_Compute-Apple_MLX-black)
![LangGraph](https://img.shields.io/badge/Agentic-LangGraph-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/Security-Docker_Sandbox-2496ED)
![LangSmith](https://img.shields.io/badge/LLMOps-LangSmith-white)


## Introduction

This repository demonstrates a complete, end-to-end **Agentic MLOps** lifecycle for financial auditing. It moves beyond standard RAG by introducing distributed data engineering, synthetic dataset generation, local LLM fine-tuning, and a strict multi-agent architecture to prevent mathematical hallucinations.

**Core Capabilities**:
- **Distributed Data Engineering:** Uses **PySpark** to process raw financial text. Implements custom context-aware chunking (Header Injection) to preserve tabular structures before saving to Parquet files and ChromaDB.
- **Agentic Distillation:** Leverages gpt-oss (Teacher Model via Groq) to autonomously generate an instruction-tuning dataset containing complex Chain-of-Thought reasoning and Python code execution paths.
- **Efficient Fine-Tuning:** Uses Apple's **MLX** framework to natively fine-tune a **Qwen2.5-Coder-3B** model in 4-bit precision on Apple Silicon Unified Memory.
- **Hybrid Edge-Cloud Orchestration:** Uses **LangGraph** to route traffic. A cloud model (Groq/Llama) acts as a strict Router/Evaluator to prevent hallucinations, while the locally fine-tuned MLX model handles deterministic Python code generation.
- **Enterprise Governance & Security:** Implements a strict **Human-in-the-Loop (HITL)** approval workflow via LangGraph memory checkpointers, and executes code safely within an ephemeral, network-isolated **Docker Sandbox** to prevent RCE vulnerabilities.

## Key Engineering Highlights
*   **Self-Querying RAG:** Implemented Metadata filtering (Ticker extraction) pre-retrieval to avoid "context poisoning" across different companies' financial reports.
*   **Strict Persona Alignment:** The Evaluator node strictly checks if the context contains both the requested *entity* and *metric*. If missing, it overrides the LLM's generic fallback with a strict corporate apology, preventing out-of-domain hallucinations.
*   **Human-in-the-Loop (HITL) Checkpoints:** Leveraging LangGraph's `interrupt_before`, the DAG pauses execution before any critical Python code is run. A human operator must explicitly review and approve the logic via the Streamlit UI, ensuring absolute compliance and auditability.
*   **Secure Ephemeral Sandbox:** The local MLX model's Python code is physically executed inside a restricted, offline Docker container (`python:3.11-slim`), completely neutralizing the risk of malicious OS operations.
*   **LLMOps & Observability:** Native integration with **LangSmith** enables distributed tracing of the entire Agentic DAG, monitoring token consumption, execution latency, and step-by-step logic routing.

## Compound AI System Architecture

The agent follows a strictly routed Directed Acyclic Graph (DAG) to optimize execution costs and guarantee data accuracy.

```mermaid
graph TD
    %% Styling
    classDef start_end fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:white,font-weight:bold;
    classDef cloud_llm fill:#3498db,stroke:#2980b9,stroke-width:2px,color:white;
    classDef local_llm fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:white;
    classDef db fill:#f39c12,stroke:#d35400,stroke-width:2px,color:white;
    classDef tool fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:white;
    classDef human fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:black,font-weight:bold;

    %% Nodes
    START((START)):::start_end
    END((END)):::start_end
    
    router[Router<br><i>Groq Llama 3.1</i>]:::cloud_llm
    retrieve[(ChromaDB<br><i>Self-Querying</i>)]:::db
    evaluator[Evaluator<br><i>Groq Llama 3.1</i>]:::cloud_llm
    coder[MLX Coder<br><i>Local Qwen 3B</i>]:::local_llm
    human_review{HITL Review<br><i>Streamlit UI</i>}:::human
    sandbox[Docker Sandbox<br><i>Ephemeral VM</i>]:::tool
    synthesizer[Synthesizer<br><i>Groq Llama 3.1</i>]:::cloud_llm

    %% Edges
    START --> router
    
    router -- "general_chat" --> synthesizer
    router -- "needs_financial_data" --> retrieve
    
    retrieve --> evaluator
    
    evaluator -- "needs_math" --> coder
    evaluator -- "can_answer_directly" --> synthesizer
    evaluator -- "insufficient_data" --> synthesizer
    
    coder --> |"Generates Python"| human_review
    human_review -- "Approve" --> sandbox
    human_review -- "Reject" --> synthesizer
    sandbox --> |"Execution Result"| synthesizer
    
    synthesizer --> END
```

## Prerequisites
- Apple Silicon Mac with at least 16GB RAM for MLX local training.
- Python 3.11
- Java (Required for PySpark)
- **Docker Desktop** (Running, for secure Sandbox execution)
- Valid API Keys for **Groq** and **LangSmith**

## Setup environment
```bash
git clone [https://github.com/marcolacagnina/Agentic-Auditor.git](https://github.com/marcolacagnina/Agentic-Auditor.git)
cd Agentic-Auditor/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env file with your credentials
echo "GROQ_API_KEY=your_groq_key_here" > .env
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)" >> .env
echo "LANGCHAIN_API_KEY=your_langsmith_key_here" >> .env
echo "LANGCHAIN_PROJECT=Agentic_Auditor" >> .env
```

## Usage Pipeline
This project is designed to be executed sequentially to mimic a real MLOps lifecycle.

1. Data Engineering (PySpark)
Process the raw data and build the Vector DB:
```bash
python -m src.spark_pipeline.ingest_spark
python -m src.spark_pipeline.load_chroma
```

2. Model Factory (Synthetic Data & MLX Training)
Generate the Distillation dataset and fine-tune the local coder model:
```bash
python -m src.dataset_gen.generate_dataset
python -m src.model_training.prepare_lora_dataset

# Train the model locally using MLX (Qwen 3B Coder)
mlx_lm.lora \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --train \
  --data data/processed \
  --iters 100 \
  --batch-size 1 \
  --num-layers 16 \
  --learning-rate 2e-5 \
  --max-seq-length 4096

# Test the newly trained weights
python -m src.model_training.test_lora
```

3. Production UI
Run the Agentic Workflow via Streamlit (Native execution recommended for MLX unified memory access):
```bash
streamlit run app.py
```

## Project Structure 
```bash
agentic-mlops-finance/
├── data/
│   ├── raw/                 # Raw financial TXT reports
│   ├── processed/           # Parquet chunks and JSONL datasets
├── adapters/                # MLX LoRA trained weights (generated locally)
├── src/
│   ├── spark_pipeline/      # PySpark ingestion & custom parsing
│   ├── dataset_gen/         # Synthetic Agentic Distillation logic
│   ├── model_training/      # MLX formatting and Code-Execution tests
│   └── production/
│       ├── graph.py         # LangGraph Compound AI logic
│       └── tools.py         # ChromaDB Self-Querying & Sandbox tools
├── chroma_db/               # Persistent Vector Store
└── app.py                   # Streamlit Frontend UI
```
