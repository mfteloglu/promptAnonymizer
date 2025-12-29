# Prompt Anonymizer API

A tiny FastAPI server exposing a single endpoint that uses the `promptAnonymizer` Python package's `run_pipeline(text)` to anonymize input text.

## Setup

Create and activate a virtual environment, install `promptAnonymizer` in editable mode, then install API dependencies.

```zsh
# From the repo root
cd promptAnonymizer
python3 -m venv .venv
source .venv/bin/activate

# Install the promptAnonymizer package (editable)
pip install -e .

# Install API dependencies
cd ../promptAnonymizer-server
pip install -r requirements.txt
```

## Run

```zsh
# With the virtual environment still active
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

or

```zsh
python3 ./main.py
```

## Test

```zsh
curl -s -X POST \
  http://127.0.0.1:8000/anonymize_text \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Error on server at IP 192.168.1.10. Email me at user@example.com",
    "use_reidentification": true
  }' | jq
```

The response is the JSON result returned by `run_pipeline(text)`.

### Request Parameters

- `text` (string): The input text to anonymize.
- `use_llm_agent_for_pii` (bool, default=True): Whether to use the LLM agent for PII detection alongside Presidio.
- `anonymization_method` (string, default="pseudonymization"): Method to use ("pseudonymization", "masking", "fpe").
- `use_reidentification` (bool, default=False): Whether to run the re-identification agent to verify safety.
- `ollama_port` (int, default=11434): Port for the local Ollama instance.
- `model` (string, default="llama3.2"): The LLM model to use.
