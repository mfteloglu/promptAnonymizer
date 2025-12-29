# Prompt Anonymizer

## Purpose-Aware Sensitive Information Pipeline

This project now implements a unified pipeline to detect and anonymize two categories of sensitivity:
- PII detection and anonymization
- Context-aware sensitive info detection and anonymization (when personal/sensitive without explicit PII)

An initial Purpose Detection step determines whether the sensitive information is relevant to the user's goal. If sensitive content is relevant to the purpose, it is not anonymized.

### High-Level Flow
1. Purpose Detection: classify prompt intent (e.g., `SUMMARIZATION`, `QUESTION_ANSWERING`).
2. PII Detection → Purpose-Aware Anonymization: detect PII and anonymize only entities not relevant to the purpose.
3. Context Detection → Purpose-Aware Anonymization: detect context sensitivity (e.g., `HEALTH_DATA`, `FINANCIAL_DATA`); anonymize the text only if detected PII is sensitive AND not relevant to the purpose.

### Key Modules
- `pipeline/langgraph_app.py`: Orchestrates the full pipeline (purpose → PII → context → re-identification).
- `analyzer/purpose/agent`: LLM-based purpose classifier (Ollama local model).
- `analyzer/purpose/detector`: Purpose constants and PII relevance mapping (`PURPOSE_RELEVANT_PII`).
- `pipeline/config.py`: Context relevance mapping per purpose (`PURPOSE_RELEVANT_CONTEXT_TYPES`).
- `anonymizer/presidio`: PII detection via Presidio (+ optional LLM) and purpose-aware anonymization.
- `analyzer/context/agent`: LLM-based sensitive-context classifier returning `{is_sensitive, context_type, ...}`.
- `anonymizer/context`: LLM-based context anonymizer with placeholders, used only when context is sensitive and not relevant.
- `reidentifier/agent`: LLM-based agent that attempts to re-identify anonymized entities to ensure safety.

### Relevance Rules
- PII relevance per purpose is configured in `analyzer/purpose/detector/main.py` (`PURPOSE_RELEVANT_PII`).
- Context-type relevance per purpose is configured in `pipeline/config.py` (`PURPOSE_RELEVANT_CONTEXT_TYPES`).

Examples:
- `QUESTION_ANSWERING`: keep relevant entities (e.g., `LOCATION` for weather questions) → no anonymization for these.
- `SUMMARIZATION`: generally anonymize all PII unless critical to the summary.

### Installation & Build

This project uses `setuptools` and `pip`.

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```
   *Dependencies include: `requests`, `presidio-analyzer`, `presidio-anonymizer`, `spacy`, `langgraph`, `langchain-core`, `faker`.*

3. Download Spacy model:
   ```bash
   python3 -m spacy download en_core_web_lg
   ```

### Quick Start (LangGraph via main.py)
```bash
# From the thesis directory
python3 main.py
```
This runs a demo script that processes a sample text.

### Using LangGraph in Code
```python
from pipeline.langgraph_app import run_pipeline
result = run_pipeline(
    text="Contact me at john@example.com; server IP 10.0.0.5; need troubleshooting",
    use_reidentification=True
)
print(result["final_text"])      # purpose-aware anonymized text
```

### LangGraph Pipeline
Builds the flow: **Purpose → PII → Context → Re-identification**.
- **Purpose**: Determines the user's intent.
- **PII**: Anonymizes PII based on purpose relevance.
- **Context**: Checks for sensitive context.
- **Re-identification**: (Optional) Attempts to break the anonymization. If successful (breach detected), it loops back to the PII node to re-anonymize with stricter settings or different methods.

```python
from pipeline.langgraph_app import run_pipeline
result = run_pipeline("Error on server at IP 10.0.0.1. Email me john@example.com")
print(result["final_text"])  # Keeps relevant entities based on purpose
```

CLI-style smoke test:
```bash
python3 - <<'PY'
from pipeline.langgraph_app import run_pipeline
print(run_pipeline("Error on server at IP 10.0.0.1. Email me john@example.com"))
PY
```

### Tests
- PII purpose rules: `test_purpose_detection.py` (e.g., keep LOCATION for weather questions; anonymize names in general chat).

### Configuration and Extensibility
- Add or adjust allowed purposes in `analyzer/purpose/detector/main.py` (`ALL_PURPOSES`).
- Adjust PII relevance per purpose in `PURPOSE_RELEVANT_PII`.
- Adjust context relevance per purpose in `pipeline/config.py` (`PURPOSE_RELEVANT_CONTEXT_TYPES`).

### Dependencies
Defined in `pyproject.toml` (`requests`, `presidio-analyzer`, `presidio-anonymizer`, `spacy`). Local LLM expected at `http://localhost:11434` (Ollama) with model name `llama3.2` (configurable in code).

