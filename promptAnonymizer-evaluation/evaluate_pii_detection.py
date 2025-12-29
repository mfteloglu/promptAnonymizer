import sys
import os
import time
import pandas as pd
import ast
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from tqdm import tqdm

from promptAnonymizer.pipeline.langgraph_app import run_pipeline

# Mapping from Dataset Labels to System Entity Types
LABEL_MAPPING = {
    "NAME_STUDENT": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
    "PHONE_NUM": "PHONE_NUMBER",
    "STREET_ADDRESS": "LOCATION",
    "URL_PERSONAL": "URL",
    "URL": "URL",
    # "USERNAME": "PERSON", # Optional: map username to person if desired
}

# Types to include in the evaluation (Overall and By Type)
TARGET_TYPES = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "URL"}

def reconstruct_text_and_extract_gt(row) -> Tuple[str, Set[Tuple[str, int, int]]]:
    """
    Reconstructs text from tokens and extracts ground truth entities based on labels.
    Returns:
        text: Reconstructed string.
        entities: Set of (entity_type, start, end) tuples.
    """
    try:
        tokens = ast.literal_eval(row['tokens'])
        trailing_whitespace = ast.literal_eval(row['trailing_whitespace'])
        labels = ast.literal_eval(row['labels'])
    except (ValueError, SyntaxError):
        # Fallback if they are already lists (depending on pandas loading)
        tokens = row['tokens'] if isinstance(row['tokens'], list) else []
        trailing_whitespace = row['trailing_whitespace'] if isinstance(row['trailing_whitespace'], list) else []
        labels = row['labels'] if isinstance(row['labels'], list) else []

    text = ""
    entities = set()
    
    current_entity_type = None
    current_start = None
    
    cursor = 0
    
    for token, ws, label in zip(tokens, trailing_whitespace, labels):
        start = cursor
        end = start + len(token)
        
        # Reconstruct text
        text += token
        if ws:
            text += " "
        cursor = len(text)
        
        # Process BIO labels
        if label == 'O':
            if current_entity_type:
                # End of entity
                if current_entity_type in LABEL_MAPPING:
                    entities.add((LABEL_MAPPING[current_entity_type], current_start, current_end))
                current_entity_type = None
                current_start = None
        else:
            prefix, label_type = label.split('-', 1)
            
            if prefix == 'B':
                if current_entity_type:
                    # End previous entity
                    if current_entity_type in LABEL_MAPPING:
                        entities.add((LABEL_MAPPING[current_entity_type], current_start, current_end))
                
                current_entity_type = label_type
                current_start = start
                current_end = end
            elif prefix == 'I':
                if current_entity_type == label_type:
                    # Continue entity
                    current_end = end
                else:
                    # Mismatch or new entity without B- tag (shouldn't happen in valid BIO)
                    # Treat as new B- if type differs, or ignore if no current type
                    if current_entity_type:
                         if current_entity_type in LABEL_MAPPING:
                            entities.add((LABEL_MAPPING[current_entity_type], current_start, current_end))
                    
                    current_entity_type = label_type
                    current_start = start
                    current_end = end

    # Capture last entity
    if current_entity_type and current_entity_type in LABEL_MAPPING:
        entities.add((LABEL_MAPPING[current_entity_type], current_start, current_end))
        
    return text, entities

def extract_system_entities(pipeline_result: Dict) -> Set[Tuple[str, int, int]]:
    """
    Extracts detected entities from the pipeline result.
    Combines kept and anonymized entities.
    """
    entities = set()
    pii_data = pipeline_result.get("pii", {})
    
    # Combine all detected entities
    all_detected = pii_data.get("kept_entities", []) + pii_data.get("anonymized_entities", [])
    
    for ent in all_detected:
        entities.add((ent["entity_type"], ent["start"], ent["end"]))
        
    return entities

def calculate_metrics(gt_entities: Set[Tuple[str, int, int]], pred_entities: Set[Tuple[str, int, int]]) -> Dict[str, int]:
    """
    Calculates TP, FP, FN using overlap matching.
    A GT entity is a TP if it overlaps with any Pred entity of the same type.
    A Pred entity is a FP if it does not overlap with any GT entity of the same type.
    """
    tp = 0
    fn = 0
    fp = 0
    
    # Convert to lists for iteration
    gt_list = list(gt_entities)
    pred_list = list(pred_entities)
    
    # Track matched predictions to avoid double counting FPs (though a pred matching multiple GTs is still 1 pred)
    # Actually, for Precision/Recall:
    # TP: Number of GT entities that are correctly detected (overlap).
    # FN: Number of GT entities not detected.
    # FP: Number of Pred entities that do not match any GT.
    
    matched_preds = set()
    
    for gt in gt_list:
        gt_type, gt_start, gt_end = gt
        found_match = False
        for i, pred in enumerate(pred_list):
            pred_type, pred_start, pred_end = pred
            
            if gt_type == pred_type:
                # Check overlap
                # Overlap if start < end and end > start
                if max(gt_start, pred_start) < min(gt_end, pred_end):
                    found_match = True
                    matched_preds.add(i)
                    # We don't break here if we want to find all matches, but for TP count of GT, one match is enough.
                    break
        
        if found_match:
            tp += 1
        else:
            fn += 1
            
    fp = len(pred_list) - len(matched_preds)
    
    return {"TP": tp, "FP": fp, "FN": fn}

def evaluate_configuration(df: pd.DataFrame, use_llm_agent: bool, limit: int = None, model_name: str = "llama3.2") -> Tuple[Dict, pd.DataFrame]:
    print(f"Evaluating with use_llm_agent_for_pii={use_llm_agent}, model={model_name}...")
    
    metrics_by_type = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    total_metrics = {"TP": 0, "FP": 0, "FN": 0}
    detailed_results = []
    
    count = 0
    for _, row in tqdm(df.iterrows(), total=min(len(df), limit) if limit else len(df)):
        if limit and count >= limit:
            break
            
        text, gt_entities = reconstruct_text_and_extract_gt(row)
        
        try:
            # Run pipeline
            start_time = time.time()
            result = run_pipeline(
                text=text,
                use_llm_agent_for_pii=use_llm_agent,
                anonymization_method="pseudonymization",
                model=model_name # Ensure consistent model
            )
            duration = time.time() - start_time
            
            pred_entities = extract_system_entities(result)
            
            # Filter entities to only include TARGET_TYPES
            gt_entities_filtered = {e for e in gt_entities if e[0] in TARGET_TYPES}
            pred_entities_filtered = {e for e in pred_entities if e[0] in TARGET_TYPES}
            
            # Calculate global metrics
            row_metrics = calculate_metrics(gt_entities_filtered, pred_entities_filtered)
            for k, v in row_metrics.items():
                total_metrics[k] += v
            
            # Calculate per-type metrics
            # We need to split GT and Pred by type
            gt_by_type = defaultdict(set)
            for e in gt_entities_filtered:
                gt_by_type[e[0]].add(e)
                
            pred_by_type = defaultdict(set)
            for e in pred_entities_filtered:
                pred_by_type[e[0]].add(e)
                
            all_types = set(gt_by_type.keys()) | set(pred_by_type.keys())
            
            for t in all_types:
                m = calculate_metrics(gt_by_type[t], pred_by_type[t])
                for k, v in m.items():
                    metrics_by_type[t][k] += v
            
            # Store detailed result
            detailed_results.append({
                "text": text,
                "gt_entities": str(list(gt_entities_filtered)),
                "pred_entities": str(list(pred_entities_filtered)),
                "TP": row_metrics["TP"],
                "FP": row_metrics["FP"],
                "FN": row_metrics["FN"],
                "time": duration,
                "pii_node_time": result.get("pii_node_execution_time", 0)
            })
                    
        except Exception as e:
            print(f"Error processing row {count}: {e}")
            
        count += 1
        
    return {"total": total_metrics, "by_type": dict(metrics_by_type)}, pd.DataFrame(detailed_results)

def print_results(results: Dict, df: pd.DataFrame, title: str):
    print(f"\n{'='*20} {title} {'='*20}")
    
    def get_f1(m):
        p = m["TP"] / (m["TP"] + m["FP"]) if (m["TP"] + m["FP"]) > 0 else 0
        r = m["TP"] / (m["TP"] + m["FN"]) if (m["TP"] + m["FN"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f1

    # Total
    p, r, f1 = get_f1(results["total"])
    avg_time = df['time'].mean() if not df.empty else 0
    avg_pii_time = df['pii_node_time'].mean() if not df.empty and 'pii_node_time' in df.columns else 0
    
    print(f"OVERALL:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall:    {r:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Avg Time:  {avg_time:.4f}s")
    print(f"  Avg PII Time: {avg_pii_time:.4f}s")
    print(f"  Counts:    TP={results['total']['TP']}, FP={results['total']['FP']}, FN={results['total']['FN']}")
    
    print("\nBY TYPE:")
    for t, m in sorted(results["by_type"].items()):
        p, r, f1 = get_f1(m)
        print(f"  {t}:")
        print(f"    Precision: {p:.4f}")
        print(f"    Recall:    {r:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
        print(f"    Counts:    TP={m['TP']}, FP={m['FP']}, FN={m['FN']}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate PII Detection System")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to evaluate (default: all)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset CSV")
    parser.add_argument("--model_name", type=str, default="llama3.2", help="Model name (e.g. llama3.2, gpt-4)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = args.dataset or os.path.join(base_dir, "datasets", "combined_dataset.csv")
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    limit = args.limit
    if limit:
        print(f"Limiting evaluation to first {limit} rows.")
    
    # Evaluate Presidio Only
    results_presidio, df_presidio = evaluate_configuration(df, use_llm_agent=False, limit=limit, model_name=args.model_name)
    print_results(results_presidio, df_presidio, "Presidio Only (Baseline)")
    
    # Calculate overall metrics for summary
    tp = results_presidio['total']['TP']
    fp = results_presidio['total']['FP']
    fn = results_presidio['total']['FN']
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    avg_time = df_presidio['time'].mean() if not df_presidio.empty else 0
    avg_pii_time = df_presidio['pii_node_time'].mean() if not df_presidio.empty and 'pii_node_time' in df_presidio.columns else 0
    
    # Add per-row metrics
    df_presidio['Precision'] = df_presidio.apply(lambda row: row['TP'] / (row['TP'] + row['FP']) if (row['TP'] + row['FP']) > 0 else 0, axis=1)
    df_presidio['Recall'] = df_presidio.apply(lambda row: row['TP'] / (row['TP'] + row['FN']) if (row['TP'] + row['FN']) > 0 else 0, axis=1)
    df_presidio['F1'] = df_presidio.apply(lambda row: 2 * row['Precision'] * row['Recall'] / (row['Precision'] + row['Recall']) if (row['Precision'] + row['Recall']) > 0 else 0, axis=1)

    # Append summary row
    summary_row = {
        "text": "OVERALL AVERAGE / TOTAL",
        "gt_entities": "",
        "pred_entities": "",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "time": avg_time,
        "pii_node_time": avg_pii_time,
        "Precision": p,
        "Recall": r,
        "F1": f1
    }
    df_presidio = pd.concat([df_presidio, pd.DataFrame([summary_row])], ignore_index=True)

    # Append per-type summary rows
    for t, m in sorted(results_presidio["by_type"].items()):
        tp_t = m['TP']
        fp_t = m['FP']
        fn_t = m['FN']
        p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) > 0 else 0
        
        type_row = {
            "text": f"OVERALL - {t}",
            "gt_entities": "",
            "pred_entities": "",
            "TP": tp_t,
            "FP": fp_t,
            "FN": fn_t,
            "time": 0,
            "pii_node_time": 0,
            "Precision": p_t,
            "Recall": r_t,
            "F1": f1_t
        }
        df_presidio = pd.concat([df_presidio, pd.DataFrame([type_row])], ignore_index=True)

    # Sanitize model name for filename
    sanitized_model_name = args.model_name.replace(":", "").replace(".", "")

    # Save Presidio results
    output_path_presidio = os.path.join(os.path.dirname(dataset_path), f"pii_detection_presidio_results_{sanitized_model_name}.csv")
    df_presidio.to_csv(output_path_presidio, index=False)
    print(f"Detailed results saved to {output_path_presidio}")
    
    # Evaluate Presidio + Agent
    results_agent, df_agent = evaluate_configuration(df, use_llm_agent=True, limit=limit, model_name=args.model_name)
    print_results(results_agent, df_agent, "Presidio + Local Agent")
    
    # Calculate overall metrics for summary
    tp = results_agent['total']['TP']
    fp = results_agent['total']['FP']
    fn = results_agent['total']['FN']
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    avg_time = df_agent['time'].mean() if not df_agent.empty else 0
    avg_pii_time = df_agent['pii_node_time'].mean() if not df_agent.empty and 'pii_node_time' in df_agent.columns else 0
    
    # Add per-row metrics
    df_agent['Precision'] = df_agent.apply(lambda row: row['TP'] / (row['TP'] + row['FP']) if (row['TP'] + row['FP']) > 0 else 0, axis=1)
    df_agent['Recall'] = df_agent.apply(lambda row: row['TP'] / (row['TP'] + row['FN']) if (row['TP'] + row['FN']) > 0 else 0, axis=1)
    df_agent['F1'] = df_agent.apply(lambda row: 2 * row['Precision'] * row['Recall'] / (row['Precision'] + row['Recall']) if (row['Precision'] + row['Recall']) > 0 else 0, axis=1)

    # Append summary row
    summary_row = {
        "text": "OVERALL AVERAGE / TOTAL",
        "gt_entities": "",
        "pred_entities": "",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "time": avg_time,
        "pii_node_time": avg_pii_time,
        "Precision": p,
        "Recall": r,
        "F1": f1
    }
    df_agent = pd.concat([df_agent, pd.DataFrame([summary_row])], ignore_index=True)

    # Append per-type summary rows
    for t, m in sorted(results_agent["by_type"].items()):
        tp_t = m['TP']
        fp_t = m['FP']
        fn_t = m['FN']
        p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) > 0 else 0
        
        type_row = {
            "text": f"OVERALL - {t}",
            "gt_entities": "",
            "pred_entities": "",
            "TP": tp_t,
            "FP": fp_t,
            "FN": fn_t,
            "time": 0,
            "pii_node_time": 0,
            "Precision": p_t,
            "Recall": r_t,
            "F1": f1_t
        }
        df_agent = pd.concat([df_agent, pd.DataFrame([type_row])], ignore_index=True)
    
    # Save Agent results'time'].mean() if not df_agent.empty else 0
    avg_pii_time = df_agent['pii_node_time'].mean() if not df_agent.empty and 'pii_node_time' in df_agent.columns else 0
    
    # Add per-row metrics
    df_agent['Precision'] = df_agent.apply(lambda row: row['TP'] / (row['TP'] + row['FP']) if (row['TP'] + row['FP']) > 0 else 0, axis=1)
    df_agent['Recall'] = df_agent.apply(lambda row: row['TP'] / (row['TP'] + row['FN']) if (row['TP'] + row['FN']) > 0 else 0, axis=1)
    df_agent['F1'] = df_agent.apply(lambda row: 2 * row['Precision'] * row['Recall'] / (row['Precision'] + row['Recall']) if (row['Precision'] + row['Recall']) > 0 else 0, axis=1)

    # Append summary row
    summary_row = {
        "text": "OVERALL AVERAGE / TOTAL",
        "gt_entities": "",
        "pred_entities": "",
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "time": avg_time,
        "pii_node_time": avg_pii_time,
        "Precision": p,
        "Recall": r,
        "F1": f1
    }
    df_agent = pd.concat([df_agent, pd.DataFrame([summary_row])], ignore_index=True)
    
    # Save Agent results
    output_path_agent = os.path.join(os.path.dirname(dataset_path), f"pii_detection_agent_results_{sanitized_model_name}.csv")
    df_agent.to_csv(output_path_agent, index=False)
    print(f"Detailed results saved to {output_path_agent}")

if __name__ == "__main__":
    main()
