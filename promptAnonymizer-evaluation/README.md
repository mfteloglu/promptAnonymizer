# Prompt Anonymizer Evaluation Scripts

This folder contains scripts to evaluate various aspects of the anonymization pipeline, including PII detection accuracy, utility preservation, and masking rates.

## Setup

Ensure you have the `promptAnonymizer` package installed in your environment (see `../promptAnonymizer/README.md`).

Install additional evaluation dependencies:
```bash
pip install pandas tqdm tiktoken langchain-community langchain-openai
```

## Scripts

### 1. `evaluate_pii_detection.py`
Evaluates the accuracy of PII detection (Presidio + LLM) against a labeled dataset.

**Usage:**
```bash
python3 evaluate_pii_detection.py --limit 100 --model_name llama3.2
```
- `--limit`: Number of samples to process.
- `--model_name`: Local LLM model to use.

### 2. `evaluate_utility_preservation.py`
Measures how well the anonymized text preserves the utility for downstream tasks (e.g., QA, Summarization) using an LLM-as-a-Judge approach (GPT-4o).

**Usage:**
```bash
python3 evaluate_utility_preservation.py --limit 50
```
*Note: Requires `OPENAI_API_KEY` environment variable.*

### 3. `evaluate_purpose_impact.py`
Compares the utility of "Purpose Aware" anonymization vs. "Anonymize All" strategy.

**Usage:**
```bash
python3 evaluate_purpose_impact.py --limit 50
```

### 4. `evaluate_masking_vs_fpe_*.py`
A set of scripts to compare Masking vs. Format Preserving Encryption (FPE) using different NLP metrics.
- `evaluate_masking_vs_fpe_bertscore.py`: Uses BERTScore.
- `evaluate_masking_vs_fpe_bleu.py`: Uses BLEU score.
- `evaluate_masking_vs_fpe_rouge.py`: Uses ROUGE score.

**Usage:**
```bash
python3 evaluate_masking_vs_fpe_bertscore.py --limit 50
```

### 5. `evaluate_memory_usage.py`
Monitors the CPU and RAM usage of the local Ollama instance while running different models.

**Usage:**
```bash
python3 evaluate_memory_usage.py
```

## Data
The scripts expect a dataset file (e.g., `combined_dataset.csv`) in the `../datasets/` folder.
