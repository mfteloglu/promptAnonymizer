import sys
import os
import time
import pandas as pd
import argparse
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from typing import Dict, List

from promptAnonymizer.pipeline.langgraph_app import run_pipeline
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

def get_llm_response(prompt: str, model_type: str = "local", model_name: str = "llama3.2") -> str:
    """
    Get response from LLM (Ollama or OpenAI).
    """
    try:
        if model_type == "local":
            llm = Ollama(model=model_name)
            return llm.invoke(prompt)
        elif model_type == "openai":
            # Assumes OPENAI_API_KEY is set in environment
            chat = ChatOpenAI(model_name=model_name, temperature=0)
            messages = [HumanMessage(content=prompt)]
            return chat.invoke(messages).content
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

def deanonymize_text(text: str, mapping: Dict[str, str]) -> str:
    """
    Replace anonymized tokens with original values using the mapping.
    """
    if not mapping:
        return text
        
    # Sort mapping by length of key (descending) to avoid partial replacements
    # e.g. replace "John Smith" before "John"
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    
    result = text
    for key in sorted_keys:
        if key in result:
            result = result.replace(key, mapping[key])
            
    return result

def evaluate_summarization(df: pd.DataFrame, limit: int = None, model_type: str = "local", model_name: str = "llama3.2"):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    results = []
    
    count = 0
    for _, row in tqdm(df.iterrows(), total=min(len(df), limit) if limit else len(df)):
        if limit and count >= limit:
            break
            
        original_text = row['text']
        # Use the prompt from the dataset or a standard one?
        # User said: "Get the prompts with Summarize this article in one text"
        # But the dataset has "Summarize the following text into 3 texts." etc.
        # I will use a standard instruction + text.
        instruction = "Summarize the following text in one paragraph:"
        
        # 1. Generate Reference Summary (Gold Standard) from Original Text
        # We use the SAME LLM to generate the reference, assuming it's the "upper bound" of quality.
        ref_prompt = f"{instruction}\n\n{original_text}"
        ref_summary = get_llm_response(ref_prompt, model_type, model_name)
        
        if not ref_summary:
            continue

        # 2. Experiment 1: Masking
        start_time = time.time()
        res_mask = run_pipeline(original_text, anonymization_method="masking", use_llm_agent_for_pii=True, model=model_name)
        time_mask = time.time() - start_time
        masked_text = res_mask['final_text']
        mapping_mask = res_mask['pii'].get('deanonymization_mapping', {})
        
        mask_prompt = f"{instruction}\n\n{masked_text}"
        mask_summary = get_llm_response(mask_prompt, model_type, model_name)
        
        # Deanonymize
        deanonymized_mask_summary = deanonymize_text(mask_summary, mapping_mask)
        
        # 3. Experiment 2: FPE
        start_time = time.time()
        res_fpe = run_pipeline(original_text, anonymization_method="fpe", use_llm_agent_for_pii=True, model=model_name)
        time_fpe = time.time() - start_time
        fpe_text = res_fpe['final_text']
        mapping_fpe = res_fpe['pii'].get('deanonymization_mapping', {})
        
        fpe_prompt = f"{instruction}\n\n{fpe_text}"
        fpe_summary = get_llm_response(fpe_prompt, model_type, model_name)
        
        # Deanonymize
        deanonymized_fpe_summary = deanonymize_text(fpe_summary, mapping_fpe)

        # 4. Experiment 3: Pseudonymization
        start_time = time.time()
        res_pseudo = run_pipeline(original_text, anonymization_method="pseudonymization", use_llm_agent_for_pii=True, model=model_name)
        time_pseudo = time.time() - start_time
        pseudo_text = res_pseudo['final_text']
        mapping_pseudo = res_pseudo['pii'].get('deanonymization_mapping', {})
        
        pseudo_prompt = f"{instruction}\n\n{pseudo_text}"
        pseudo_summary = get_llm_response(pseudo_prompt, model_type, model_name)
        
        # Deanonymize
        deanonymized_pseudo_summary = deanonymize_text(pseudo_summary, mapping_pseudo)
        
        # Calculate Scores
        scores_mask = scorer.score(ref_summary, deanonymized_mask_summary)
        scores_fpe = scorer.score(ref_summary, deanonymized_fpe_summary)
        scores_pseudo = scorer.score(ref_summary, deanonymized_pseudo_summary)
        
        results.append({
            "original_text": original_text,
            "ref_summary": ref_summary,
            "masked_text": masked_text,
            "mask_summary": mask_summary,
            "deanonymized_mask_summary": deanonymized_mask_summary,
            "time_mask": time_mask,
            "fpe_text": fpe_text,
            "fpe_summary": fpe_summary,
            "deanonymized_fpe_summary": deanonymized_fpe_summary,
            "time_fpe": time_fpe,
            "pseudo_text": pseudo_text,
            "pseudo_summary": pseudo_summary,
            "deanonymized_pseudo_summary": deanonymized_pseudo_summary,
            "time_pseudo": time_pseudo,
            "rouge1_mask": scores_mask['rouge1'].fmeasure,
            "rouge2_mask": scores_mask['rouge2'].fmeasure,
            "rougeL_mask": scores_mask['rougeL'].fmeasure,
            "rouge1_fpe": scores_fpe['rouge1'].fmeasure,
            "rouge2_fpe": scores_fpe['rouge2'].fmeasure,
            "rougeL_fpe": scores_fpe['rougeL'].fmeasure,
            "rouge1_pseudo": scores_pseudo['rouge1'].fmeasure,
            "rouge2_pseudo": scores_pseudo['rouge2'].fmeasure,
            "rougeL_pseudo": scores_pseudo['rougeL'].fmeasure,
        })
        
        count += 1
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Masking vs FPE for Summarization")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to evaluate")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV")
    parser.add_argument("--model_type", type=str, default="local", choices=["local", "openai"], help="LLM type")
    parser.add_argument("--model_name", type=str, default="llama3.2", help="Model name (e.g. llama3.2, gpt-4)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = args.dataset or os.path.join(base_dir, "datasets", "combined_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    # Filter for Summarization
    if 'prompt_category' in df.columns:
        df = df[df['prompt_category'] == 'Summarization']
        print(f"Filtered to {len(df)} summarization samples.")
    else:
        print("Warning: 'prompt_category' column not found. Using all samples.")
    
    results_df = evaluate_summarization(df, limit=args.limit, model_type=args.model_type, model_name=args.model_name)
    
    # Print Average Scores
    print("\n=== Average ROUGE Scores ===")
    print(f"Masking (Deanonymized) vs Reference:")
    print(f"  ROUGE-1: {results_df['rouge1_mask'].mean():.4f}")
    print(f"  ROUGE-2: {results_df['rouge2_mask'].mean():.4f}")
    print(f"  ROUGE-L: {results_df['rougeL_mask'].mean():.4f}")
    print(f"  Avg Time: {results_df['time_mask'].mean():.4f}s")
    
    print(f"\nFPE (Deanonymized) vs Reference:")
    print(f"  ROUGE-1: {results_df['rouge1_fpe'].mean():.4f}")
    print(f"  ROUGE-2: {results_df['rouge2_fpe'].mean():.4f}")
    print(f"  ROUGE-L: {results_df['rougeL_fpe'].mean():.4f}")
    print(f"  Avg Time: {results_df['time_fpe'].mean():.4f}s")

    print(f"\nPseudonymization (Deanonymized) vs Reference:")
    print(f"  ROUGE-1: {results_df['rouge1_pseudo'].mean():.4f}")
    print(f"  ROUGE-2: {results_df['rouge2_pseudo'].mean():.4f}")
    print(f"  ROUGE-L: {results_df['rougeL_pseudo'].mean():.4f}")
    print(f"  Avg Time: {results_df['time_pseudo'].mean():.4f}s")
    
    # Add summary row
    summary_row = {
        "original_text": "AVERAGE",
        "rouge1_mask": results_df['rouge1_mask'].mean(),
        "rouge2_mask": results_df['rouge2_mask'].mean(),
        "rougeL_mask": results_df['rougeL_mask'].mean(),
        "time_mask": results_df['time_mask'].mean(),
        "rouge1_fpe": results_df['rouge1_fpe'].mean(),
        "rouge2_fpe": results_df['rouge2_fpe'].mean(),
        "rougeL_fpe": results_df['rougeL_fpe'].mean(),
        "time_fpe": results_df['time_fpe'].mean(),
        "rouge1_pseudo": results_df['rouge1_pseudo'].mean(),
        "rouge2_pseudo": results_df['rouge2_pseudo'].mean(),
        "rougeL_pseudo": results_df['rougeL_pseudo'].mean(),
        "time_pseudo": results_df['time_pseudo'].mean(),
    }
    results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save results
    output_path = os.path.join(os.path.dirname(dataset_path), "masking_vs_fpe_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
