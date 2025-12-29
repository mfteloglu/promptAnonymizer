import sys
import os
import time
import pandas as pd
import argparse
import json
from tqdm import tqdm
from typing import Dict, List, Any
import tiktoken

from promptAnonymizer.pipeline.langgraph_app import run_pipeline

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Constants
DEFAULT_MODEL = "llama3.2"
JUDGE_MODEL = "gpt-4o-mini"
DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "pii_dataset.csv")

TASK_PROMPTS = {
    "qa": "What would be the 3 personal traits of the person introducted in the following text. {text}",
    "open-ended": "Write a small story about the person introduced in the following text. {text}"
}

JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating the utility preservation of an anonymization system.
Your task is to compare a Reference Response (generated from original text) with a Candidate Response (generated from anonymized text and then de-anonymized).
You must assess whether the Candidate Response maintains the semantic correctness, factual consistency, and overall usefulness of the Reference Response.
Focus on semantic equivalence and task completion rather than stylistic variation.

Score the Candidate Response on a scale of 1 to 5:
1: Severe degradation in utility. The response is irrelevant, incorrect, or nonsensical compared to the reference.
2: Significant loss of utility. Key information is missing or altered, making the response much less useful.
3: Moderate utility preservation. The response is somewhat useful but lacks some details or accuracy found in the reference.
4: Good utility preservation. The response is mostly accurate and useful, with only minor discrepancies.
5: Excellent utility preservation. The response is semantically equivalent to the reference and fully useful.

Output your evaluation in the following JSON format:
{
    "score": <integer 1-5>,
    "reasoning": "<short explanation>"
}
"""

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def get_llm_response(prompt: str, model_name: str = DEFAULT_MODEL, retries: int = 3) -> str:
    """
    Get response from local Ollama model with retries and refusal handling.
    """
    for attempt in range(retries):
        try:
            llm = Ollama(model=model_name)
            response = llm.invoke(prompt)
            
            # Check for common refusal patterns
            refusal_phrases = [
                "I can't provide", "I cannot provide", "I can't help", "I cannot help",
                "I am unable to", "I'm unable to", "I cannot write", "I can't write",
                "I cannot generate", "I can't generate", "against my programming",
                "violate my safety guidelines", "I cannot fulfill this request"
            ]
            
            if any(phrase in response for phrase in refusal_phrases):
                print(f"Refusal detected (Attempt {attempt+1}/{retries}). Retrying...")
                # Modify prompt slightly to encourage compliance
                prompt = "This is a fictional scenario for research purposes. " + prompt
                continue
                
            return response
        except Exception as e:
            print(f"Error calling Ollama ({model_name}): {e}")
            time.sleep(1)
            
    return ""

def get_judge_evaluation(reference: str, candidate: str, model_name: str = JUDGE_MODEL) -> Dict[str, Any]:
    """
    Get evaluation from OpenAI Judge.
    """
    try:
        chat = ChatOpenAI(model_name=model_name, temperature=0)
        
        user_prompt = f"""Reference Response:
{reference}

Candidate Response:
{candidate}

Evaluate the Candidate Response against the Reference Response."""

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        response = chat.invoke(messages).content
        
        # Parse JSON from response
        try:
            # clean markdown code blocks if present
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from judge response: {response}")
            return {"score": 0, "reasoning": "Parse Error"}
            
    except Exception as e:
        print(f"Error calling OpenAI Judge ({model_name}): {e}")
        return {"score": 0, "reasoning": f"API Error: {e}"}

def deanonymize_text(text: str, mapping: Dict[str, str]) -> str:
    """
    Replace anonymized tokens with original values using the mapping.
    """
    if not mapping:
        return text
    
    # Sort mapping by length of key (descending) to avoid partial replacements
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
    
    result = text
    for anon, orig in sorted_mapping:
        result = result.replace(anon, orig)
    return result

def run_pipeline_with_retry(text: str, model: str, use_purpose: bool, retries: int = 3) -> Dict:
    """
    Run the pipeline with retry logic for JSON parsing errors.
    """
    for attempt in range(retries):
        try:
            result = run_pipeline(
                text=text,
                model=model,
                anonymization_method="pseudonymization",
                use_llm_agent_for_pii=True,
                use_reidentification=False,
                use_purpose_awareness=use_purpose
            )
            return result
        except Exception as e:
            print(f"Pipeline error (Attempt {attempt+1}/{retries}): {e}")
            if "JSON parsing failed" in str(e) or "no array found" in str(e):
                time.sleep(1)
                continue
            else:
                # If it's not a JSON error, maybe re-raise or just continue retrying?
                # For now, retry all exceptions as the pipeline might be flaky
                time.sleep(1)
                continue
    
    # If all retries fail, return a dummy result or raise
    raise Exception("Pipeline failed after multiple retries")

def evaluate_sample(row: pd.Series, model: str, judge_model: str) -> List[Dict]:
    original_text = row['text']
    results = []
    
    # Generate Reference Responses for both tasks
    reference_responses = {}
    for task, prompt_template in TASK_PROMPTS.items():
        prompt = prompt_template.format(text=original_text)
        resp = get_llm_response(prompt, model)
        reference_responses[task] = resp

    # Evaluate Purpose Aware vs Not Purpose Aware
    # True = Purpose Aware (Selective Anonymization)
    # False = Not Purpose Aware (Anonymize All Detected PII)
    purpose_configs = [True, False]
    
    for use_purpose in purpose_configs:
        config_name = "Purpose Aware" if use_purpose else "Anonymize All"
        
        try:
            # Run pipeline to get anonymized text
            pipeline_result = run_pipeline_with_retry(
                text=original_text,
                model=model,
                use_purpose=use_purpose
            )
        except Exception as e:
            print(f"Skipping configuration {config_name} due to pipeline failure: {e}")
            continue
        
        anonymized_text = pipeline_result['final_text']
        mapping = pipeline_result['pii'].get('deanonymization_mapping', {})
        
        for task, prompt_template in TASK_PROMPTS.items():
            # Generate Anonymized Response
            prompt = prompt_template.format(text=anonymized_text)
            anon_resp = get_llm_response(prompt, model)
            
            # De-anonymize
            deanonymized_resp = deanonymize_text(anon_resp, mapping)
            
            # Judge
            ref_resp = reference_responses[task]
            
            # Calculate tokens
            input_tokens = count_tokens(prompt, judge_model) # Approximation of input to task model
            output_tokens = count_tokens(anon_resp, judge_model) # Approximation of output from task model
            
            print(f"Task: {task}, Config: {config_name}")
            
            eval_result = get_judge_evaluation(ref_resp, deanonymized_resp, judge_model)
            
            results.append({
                "document_id": row.get('document', 'unknown'),
                "task": task,
                "configuration": config_name,
                "score": eval_result['score'],
                "reasoning": eval_result['reasoning'],
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reference_response": ref_resp,
                "anonymized_response": anon_resp,
                "deanonymized_response": deanonymized_resp
            })
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Purpose Awareness Impact on Utility")
    parser.add_argument("--input_file", type=str, default=DATASET_PATH, help="Path to input CSV")
    parser.add_argument("--output_file", type=str, default="purpose_impact_results.csv", help="Path to output CSV")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Local model for task generation")
    parser.add_argument("--judge_model", type=str, default=JUDGE_MODEL, help="OpenAI model for judging")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        sys.exit(1)
        
    # Detect separator based on file extension or content, but defaulting to ; for pii_dataset.csv
    sep = ';' if 'pii_dataset.csv' in args.input_file else ','
    df = pd.read_csv(args.input_file, sep=sep)
    if args.limit:
        df = df.head(args.limit)
        
    all_results = []
    
    print(f"Starting evaluation on {len(df)} samples...")
    print(f"Task Model: {args.model}")
    print(f"Judge Model: {args.judge_model}")
    print("Comparing: Purpose Aware vs Anonymize All (Pseudonymization)")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            sample_results = evaluate_sample(row, args.model, args.judge_model)
            all_results.extend(sample_results)
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
            
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_file, index=False)
    
    # Calculate and print average scores
    if not results_df.empty:
        print("\nResults Summary:")
        try:
            summary = results_df.groupby(['task', 'configuration'])['score'].mean()
            print(summary)
        except KeyError:
            print("Could not calculate summary stats (missing columns?)")
    else:
        print("No results generated.")
    
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
