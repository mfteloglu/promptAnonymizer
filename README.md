# Prompt Anonymizer Thesis Project

This repository contains the implementation and evaluation code for a thesis project focused on **Purpose-Aware Sensitive Information Anonymization** for Large Language Models (LLMs).

The system is designed to detect and anonymize sensitive information (PII and sensitive contexts) in user prompts before they are sent to LLM providers, while preserving the utility of the prompt based on the user's specific intent (e.g., "Summarization" vs. "Question Answering").

## Repository Structure

The project is organized into several modular components:

### 📦 `promptAnonymizer/`
The core Python package containing the main logic for the system.
- **`analyzer/`**: Modules for analyzing input text, including PII detection agents, purpose extraction, and context analysis.
- **`anonymizer/`**: Implementation of anonymization techniques (Presidio, FPE, Context-aware).
- **`pipeline/`**: The orchestration layer built with LangGraph that manages the workflow.
- **`reidentifier/`**: Agents designed to simulate re-identification attacks for safety verification.

### 📊 `promptAnonymizer-evaluation/`
A suite of scripts and tools used to quantitatively evaluate the system.
- Benchmarks for PII detection accuracy.
- Utility preservation metrics (BERTScore, BLEU, ROUGE).
- Memory usage analysis.

### 🚀 `promptAnonymizer-server/`
A lightweight FastAPI backend that exposes the anonymization pipeline via a REST API. This allows external applications (like the browser extension) to consume the anonymization service.

### 🧩 `promptAnonymizer-browser-extension/`
A client-side Chrome/browser extension that intercepts prompts on web interfaces (e.g., ChatGPT), sanitizes them using the local API, and allows the user to review changes before submission.

### 📂 `datasets/`
Contains the datasets used for testing and evaluation, including synthetic PII datasets and CSV files containing experimental results.

## Getting Started

To use the core package:

```bash
cd promptAnonymizer
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

To run the server:

```bash
cd promptAnonymizer-server
pip install -r requirements.txt
uvicorn main:app --reload
```

For more detailed instructions, please refer to the `README.md` files within each subdirectory.
