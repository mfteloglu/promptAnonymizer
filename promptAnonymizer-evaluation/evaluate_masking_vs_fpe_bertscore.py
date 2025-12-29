import sys
import os
import time
import pandas as pd
import argparse
import json
from tqdm import tqdm
from typing import Dict, List
import torch

from promptAnonymizer.pipeline.langgraph_app import run_pipeline
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage

try:
    from bert_score import score
except ImportError:
    print("bert_score not installed. Please run: pip install bert-score")
    sys.exit(1)

def get_llm_response(prompt: str, model_type: str = "local", model_name: str = "llama3.2") -> str:
    """
    Get response from LLM (Ollama or OpenAI).
    """
    try:
        if model_type == "local":
            # Updated import for newer langchain versions if needed, but keeping consistent with existing code
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
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    
    result = text
    for key in sorted_keys:
        if key in result:
            result = result.replace(key, mapping[key])
            
    return result

def evaluate_paraphrasing(df: pd.DataFrame, limit: int = None, model_type: str = "local", model_name: str = "llama3.2"):
    results = []
    
    # Lists to store texts for batch BERTScore calculation
    refs = []
    cands_mask = []
    cands_fpe = []
    cands_pseudo = []
    
    instruction = "Rewrite this text to make it concise."
    
    print(f"Starting evaluation with instruction: '{instruction}'")
    
    count = 0
    for _, row in tqdm(df.iterrows(), total=min(len(df), limit) if limit else len(df)):
        if limit and count >= limit:
            break
            
        original_text = row['text']
        
        # 1. Generate Reference Paraphrase (Gold Standard) from Original Text
        ref_prompt = f"{instruction}\n\n{original_text}"
        ref_response = get_llm_response(ref_prompt, model_type, model_name)
        
        if not ref_response:
            continue

        # 2. Experiment 1: Masking
        start_time = time.time()
        res_mask = run_pipeline(original_text, anonymization_method="masking", use_llm_agent_for_pii=True, model=model_name)
        time_mask = time.time() - start_time
        masked_text = res_mask['final_text']
        mapping_mask = res_mask['pii'].get('deanonymization_mapping', {})
        
        mask_prompt = f"{instruction}\n\n{masked_text}"
        mask_response = get_llm_response(mask_prompt, model_type, model_name)
        
        # Deanonymize
        deanonymized_mask_response = deanonymize_text(mask_response, mapping_mask)
        
        # 3. Experiment 2: FPE
        start_time = time.time()
        res_fpe = run_pipeline(original_text, anonymization_method="fpe", use_llm_agent_for_pii=True, model=model_name)
        time_fpe = time.time() - start_time
        fpe_text = res_fpe['final_text']
        mapping_fpe = res_fpe['pii'].get('deanonymization_mapping', {})
        
        fpe_prompt = f"{instruction}\n\n{fpe_text}"
        fpe_response = get_llm_response(fpe_prompt, model_type, model_name)
        
        # Deanonymize
        deanonymized_fpe_response = deanonymize_text(fpe_response, mapping_fpe)

        # 4. Experiment 3: Pseudonymization
        start_time = time.time()
        res_pseudo = run_pipeline(original_text, anonymization_method="pseudonymization", use_llm_agent_for_pii=True, model=model_name)
        time_pseudo = time.time() - start_time
        pseudo_text = res_pseudo['final_text']
        mapping_pseudo = res_pseudo['pii'].get('deanonymization_mapping', {})
        
        pseudo_prompt = f"{instruction}\n\n{pseudo_text}"
        pseudo_response = get_llm_response(pseudo_prompt, model_type, model_name)
        
        # Deanonymize
        deanonymized_pseudo_response = deanonymize_text(pseudo_response, mapping_pseudo)
        
        # Store for BERTScore
        refs.append(ref_response)
        cands_mask.append(deanonymized_mask_response)
        cands_fpe.append(deanonymized_fpe_response)
        cands_pseudo.append(deanonymized_pseudo_response)
        
        results.append({
            "original_text": original_text,
            "ref_response": ref_response,
            "masked_text": masked_text,
            "mask_response": mask_response,
            "deanonymized_mask_response": deanonymized_mask_response,
            "time_mask": time_mask,
            "fpe_text": fpe_text,
            "fpe_response": fpe_response,
            "deanonymized_fpe_response": deanonymized_fpe_response,
            "time_fpe": time_fpe,
            "pseudo_text": pseudo_text,
            "pseudo_response": pseudo_response,
            "deanonymized_pseudo_response": deanonymized_pseudo_response,
            "time_pseudo": time_pseudo,
        })
        
        count += 1
    
    if not results:
        print("No results generated.")
        return pd.DataFrame()

    print("\nCalculating BERTScore (this may take a moment)...")
    
    # Calculate BERTScore for Masking
    P_mask, R_mask, F1_mask = score(cands_mask, refs, lang="en", verbose=True)
    
    # Calculate BERTScore for FPE
    P_fpe, R_fpe, F1_fpe = score(cands_fpe, refs, lang="en", verbose=True)

    # Calculate BERTScore for Pseudonymization
    P_pseudo, R_pseudo, F1_pseudo = score(cands_pseudo, refs, lang="en", verbose=True)
    
    # Add scores to results
    for i, res in enumerate(results):
        res['bert_precision_mask'] = P_mask[i].item()
        res['bert_recall_mask'] = R_mask[i].item()
        res['bert_f1_mask'] = F1_mask[i].item()
        
        res['bert_precision_fpe'] = P_fpe[i].item()
        res['bert_recall_fpe'] = R_fpe[i].item()
        res['bert_f1_fpe'] = F1_fpe[i].item()

        res['bert_precision_pseudo'] = P_pseudo[i].item()
        res['bert_recall_pseudo'] = R_pseudo[i].item()
        res['bert_f1_pseudo'] = F1_pseudo[i].item()
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Masking vs FPE for Paraphrasing (BERTScore)")
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
    
    # Filter for Paraphrasing/Rewriting
    if 'prompt_category' in df.columns:
        # Try exact match first, then partial if needed
        target_category = 'Paraphrasing / Rewriting'
        df = df[df['prompt_category'] == target_category]
        print(f"Filtered to {len(df)} '{target_category}' samples.")
    else:
        print("Warning: 'prompt_category' column not found. Using all samples.")
        
    # Further filter for specific instruction if needed, but user asked to use the instruction "Rewrite this text to make it concise."
    # We will use that instruction for generation.
    
    results_df = evaluate_paraphrasing(df, limit=args.limit, model_type=args.model_type, model_name=args.model_name)
    
    if results_df.empty:
        return

    # Print Average Scores
    print("\n=== Average BERTScore ===")
    print(f"Masking (Deanonymized) vs Reference:")
    print(f"  Precision: {results_df['bert_precision_mask'].mean():.4f}")
    print(f"  Recall:    {results_df['bert_recall_mask'].mean():.4f}")
    print(f"  F1:        {results_df['bert_f1_mask'].mean():.4f}")
    print(f"  Avg Time:  {results_df['time_mask'].mean():.4f}s")
    
    print(f"\nFPE (Deanonymized) vs Reference:")
    print(f"  Precision: {results_df['bert_precision_fpe'].mean():.4f}")
    print(f"  Recall:    {results_df['bert_recall_fpe'].mean():.4f}")
    print(f"  F1:        {results_df['bert_f1_fpe'].mean():.4f}")
    print(f"  Avg Time:  {results_df['time_fpe'].mean():.4f}s")

    print(f"\nPseudonymization (Deanonymized) vs Reference:")
    print(f"  Precision: {results_df['bert_precision_pseudo'].mean():.4f}")
    print(f"  Recall:    {results_df['bert_recall_pseudo'].mean():.4f}")
    print(f"  F1:        {results_df['bert_f1_pseudo'].mean():.4f}")
    print(f"  Avg Time:  {results_df['time_pseudo'].mean():.4f}s")
    
    # Add summary row
    summary_row = {
        "original_text": "AVERAGE",
        "bert_precision_mask": results_df['bert_precision_mask'].mean(),
        "bert_recall_mask": results_df['bert_recall_mask'].mean(),
        "bert_f1_mask": results_df['bert_f1_mask'].mean(),
        "time_mask": results_df['time_mask'].mean(),
        "bert_precision_fpe": results_df['bert_precision_fpe'].mean(),
        "bert_recall_fpe": results_df['bert_recall_fpe'].mean(),
        "bert_f1_fpe": results_df['bert_f1_fpe'].mean(),
        "time_fpe": results_df['time_fpe'].mean(),
        "bert_precision_pseudo": results_df['bert_precision_pseudo'].mean(),
        "bert_recall_pseudo": results_df['bert_recall_pseudo'].mean(),
        "bert_f1_pseudo": results_df['bert_f1_pseudo'].mean(),
        "time_pseudo": results_df['time_pseudo'].mean(),
    }
    results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save results
    output_path = os.path.join(os.path.dirname(dataset_path), "masking_vs_fpe_bertscore_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
