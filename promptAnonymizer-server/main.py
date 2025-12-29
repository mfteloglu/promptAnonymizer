from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Try to import the pipeline function in a robust way.
# Preference: import from local repo source (pipeline/). Fallback to promptAnonymizer package if available.
try:
    from pipeline.langgraph_app import run_pipeline  # type: ignore  # when running in the repo
except ModuleNotFoundError:
    # Add ../promptAnonymizer to sys.path so that `pipeline/` becomes importable
    repo_prompt_anonymizer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "promptAnonymizer"))
    if os.path.isdir(repo_prompt_anonymizer_dir) and repo_prompt_anonymizer_dir not in sys.path:
        sys.path.insert(0, repo_prompt_anonymizer_dir)
    try:
        from pipeline.langgraph_app import run_pipeline  # type: ignore  # try again after adjusting path
    except ModuleNotFoundError:
        # Final fallback: if someone has packaged pipeline under promptAnonymizer (less likely)
        from promptAnonymizer.pipeline.langgraph_app import run_pipeline  # type: ignore


class AnonymizeRequest(BaseModel):
    text: str
    use_llm_agent_for_pii: bool = True
    anonymization_method: str = "pseudonymization"
    use_reidentification: bool = False
    ollama_port: int = 11434
    model: str = "llama3.2"


app = FastAPI(title="Prompt Anonymizer API", version="0.1.0")

@app.post("/anonymize_text")
def anonymize_text(payload: AnonymizeRequest):
    print(f"Received request to anonymize text of length {len(payload.text)}")
    try:
        result = run_pipeline(
            text=payload.text,
            model=payload.model,
            use_llm_agent_for_pii=payload.use_llm_agent_for_pii,
            anonymization_method=payload.anonymization_method,
            use_reidentification=payload.use_reidentification,
            ollama_port=payload.ollama_port
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional: allow `python main.py` for local dev
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
