import sys
import os
import time
import pandas as pd
import argparse
import json
from tqdm import tqdm
import sacrebleu
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
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    
    result = text
    for key in sorted_keys:
        if key in result:
            result = result.replace(key, mapping[key])
            
    return result

def evaluate_translation(df: pd.DataFrame, limit: int = None, model_type: str = "local", model_name: str = "llama3.2"):
    results = []
    
    count = 0
    for _, row in tqdm(df.iterrows(), total=min(len(df), limit) if limit else len(df)):
        if limit and count >= limit:
            break
            
        original_text = row['text']
        # Use the specific prompt requested
        instruction = "The text may contain masked placeholders like <PERSON>, <EMAIL_ADDRESS>, etc. Please preserve these placeholders exactly as they are in the translation:"
        
        # 1. Generate Reference Translation (Gold Standard) from Original Text
        ref_prompt = f"{original_text}"
        ref_translation = get_llm_response(ref_prompt, model_type, model_name)
        
        if not ref_translation:
            continue

        clean_instruction = "Translate the following text from English to German."

        # 2. Experiment 1: Masking
        start_time = time.time()
        res_mask = run_pipeline(original_text, anonymization_method="masking", use_llm_agent_for_pii=True, model=model_name)
        time_mask = time.time() - start_time
        masked_text = res_mask['final_text']
        mapping_mask = res_mask['pii'].get('deanonymization_mapping', {})
        
        if "\n" in masked_text:
            _, content_mask = masked_text.split("\n", 1)
            masked_text_fixed = f"{clean_instruction}\n{content_mask}"
        else:
            masked_text_fixed = f"{clean_instruction}\n\n{masked_text}"

        mask_prompt = f"{instruction}\n\n{masked_text_fixed}"
        mask_translation = get_llm_response(mask_prompt, model_type, model_name)
        deanonymized_mask_translation = deanonymize_text(mask_translation, mapping_mask)
        
        # 3. Experiment 2: FPE
        start_time = time.time()
        res_fpe = run_pipeline(original_text, anonymization_method="fpe", use_llm_agent_for_pii=True, model=model_name)
        time_fpe = time.time() - start_time
        fpe_text = res_fpe['final_text']
        mapping_fpe = res_fpe['pii'].get('deanonymization_mapping', {})
        
        if "\n" in fpe_text:
            _, content_fpe = fpe_text.split("\n", 1)
            fpe_text_fixed = f"{clean_instruction}\n{content_fpe}"
        else:
            fpe_text_fixed = f"{clean_instruction}\n\n{fpe_text}"
        
        fpe_prompt = f"{fpe_text_fixed}"
        fpe_translation = get_llm_response(fpe_prompt, model_type, model_name)
        deanonymized_fpe_translation = deanonymize_text(fpe_translation, mapping_fpe)

        # 4. Experiment 3: Pseudonymization
        start_time = time.time()
        res_pseudo = run_pipeline(original_text, anonymization_method="pseudonymization", use_llm_agent_for_pii=True, model=model_name)
        time_pseudo = time.time() - start_time
        pseudo_text = res_pseudo['final_text']
        mapping_pseudo = res_pseudo['pii'].get('deanonymization_mapping', {})
        
        if "\n" in pseudo_text:
            _, content_pseudo = pseudo_text.split("\n", 1)
            pseudo_text_fixed = f"{clean_instruction}\n{content_pseudo}"
        else:
            pseudo_text_fixed = f"{clean_instruction}\n\n{pseudo_text}"
        
        pseudo_prompt = f"{instruction}\n\n{pseudo_text_fixed}"
        pseudo_translation = get_llm_response(pseudo_prompt, model_type, model_name)
        deanonymized_pseudo_translation = deanonymize_text(pseudo_translation, mapping_pseudo)
        
        # Calculate BLEU Scores
        bleu_mask = sacrebleu.sentence_bleu(deanonymized_mask_translation, [ref_translation])
        bleu_fpe = sacrebleu.sentence_bleu(deanonymized_fpe_translation, [ref_translation])
        bleu_pseudo = sacrebleu.sentence_bleu(deanonymized_pseudo_translation, [ref_translation])
        
        print(f"\n--- Sample {count+1} ---")
        print(f"Original: {original_text[:100]}...")
        print(f"Ref Translation: {ref_translation[:100]}...")
        print(f"Masked Text: {masked_text[:100]}...")
        print(f"Mask Translation: {mask_translation[:100]}...")
        print(f"Deanonymized Mask: {deanonymized_mask_translation[:100]}...")
        print(f"FPE Text: {fpe_text[:100]}...")
        print(f"FPE Translation: {fpe_translation[:100]}...")
        print(f"Deanonymized FPE: {deanonymized_fpe_translation[:100]}...")
        print(f"Pseudo Text: {pseudo_text[:100]}...")
        print(f"Pseudo Translation: {pseudo_translation[:100]}...")
        print(f"Deanonymized Pseudo: {deanonymized_pseudo_translation[:100]}...")
        print(f"BLEU Mask: {bleu_mask.score:.2f}")
        print(f"BLEU FPE: {bleu_fpe.score:.2f}")
        print(f"BLEU Pseudo: {bleu_pseudo.score:.2f}")
        
        results.append({
            "original_text": original_text,
            "ref_translation": ref_translation,
            "masked_text": masked_text,
            "mask_translation": mask_translation,
            "deanonymized_mask_translation": deanonymized_mask_translation,
            "time_mask": time_mask,
            "fpe_text": fpe_text,
            "fpe_translation": fpe_translation,
            "deanonymized_fpe_translation": deanonymized_fpe_translation,
            "time_fpe": time_fpe,
            "pseudo_text": pseudo_text,
            "pseudo_translation": pseudo_translation,
            "deanonymized_pseudo_translation": deanonymized_pseudo_translation,
            "time_pseudo": time_pseudo,
            "bleu_mask": bleu_mask.score,
            "bleu_fpe": bleu_fpe.score,
            "bleu_pseudo": bleu_pseudo.score,
        })
        
        count += 1
        
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Masking vs FPE for Translation (BLEU)")
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
    
    # Filter for Translation and specific prompt
    if 'prompt_category' in df.columns and 'source_prompt' in df.columns:
        # Filter for Translation category
        df = df[df['prompt_category'] == 'Translation']
        # Filter for specific prompt text (English to German)
        # Using str.contains to be safe against minor variations or whitespace
        df = df[df['source_prompt'].str.contains("Translate the following text from English to German", case=False, na=False)]
        print(f"Filtered to {len(df)} English-to-German translation samples.")
    else:
        print("Warning: 'prompt_category' or 'source_prompt' column not found. Using all samples.")
    
    if len(df) == 0:
        print("No samples found matching criteria.")
        return

    results_df = evaluate_translation(df, limit=args.limit, model_type=args.model_type, model_name=args.model_name)
    
    # Print Average Scores
    print("\n=== Average BLEU Scores ===")
    print(f"Masking (Deanonymized) vs Reference:")
    print(f"  BLEU: {results_df['bleu_mask'].mean():.4f}")
    print(f"  Avg Time: {results_df['time_mask'].mean():.4f}s")
    
    print(f"\nFPE (Deanonymized) vs Reference:")
    print(f"  BLEU: {results_df['bleu_fpe'].mean():.4f}")
    print(f"  Avg Time: {results_df['time_fpe'].mean():.4f}s")

    print(f"\nPseudonymization (Deanonymized) vs Reference:")
    print(f"  BLEU: {results_df['bleu_pseudo'].mean():.4f}")
    print(f"  Avg Time: {results_df['time_pseudo'].mean():.4f}s")
    
    # Add summary row
    summary_row = {
        "original_text": "AVERAGE",
        "bleu_mask": results_df['bleu_mask'].mean(),
        "time_mask": results_df['time_mask'].mean(),
        "bleu_fpe": results_df['bleu_fpe'].mean(),
        "time_fpe": results_df['time_fpe'].mean(),
        "bleu_pseudo": results_df['bleu_pseudo'].mean(),
        "time_pseudo": results_df['time_pseudo'].mean(),
    }
    results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save results
    output_path = os.path.join(os.path.dirname(dataset_path), "masking_vs_fpe_bleu_results.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to {output_path}")

if __name__ == "__main__":
    main()
