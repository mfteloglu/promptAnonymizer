import pandas as pd
import os

def generate_combined_dataset():
    # File paths
    prompts_path = 'writing_tools_prompts.csv'
    pii_path = 'pii_dataset.csv'
    output_path = 'combined_dataset.csv'

    # Check if files exist
    if not os.path.exists(prompts_path) or not os.path.exists(pii_path):
        print("Error: Input files not found.")
        return

    # Read the CSV files
    try:
        df_prompts = pd.read_csv(prompts_path)
        # pii_dataset.csv uses semicolons as separators based on inspection
        df_pii = pd.read_csv(pii_path, sep=';', on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    # Verify we have enough data
    num_prompts = len(df_prompts)
    required_texts = num_prompts * 10
    
    if len(df_pii) < required_texts:
        print(f"Error: Not enough texts in pii_dataset. Need {required_texts}, but have {len(df_pii)}.")
        return

    print(f"Found {num_prompts} prompts.")
    print(f"Found {len(df_pii)} texts in pii_dataset.")
    print(f"Generating {required_texts} combined entries...")

    # Shuffle pii dataset to get random texts
    # We take the first 'required_texts' rows after shuffling
    df_pii_sampled = df_pii.sample(n=required_texts, random_state=42).reset_index(drop=True)

    new_rows = []

    # Iterate through each prompt
    for prompt_idx, prompt_row in df_prompts.iterrows():
        prompt_text = prompt_row['Prompt']
        
        # Get the slice of 10 texts for this prompt
        start_idx = prompt_idx * 10
        end_idx = start_idx + 10
        
        subset_pii = df_pii_sampled.iloc[start_idx:end_idx].copy()
        
        for _, pii_row in subset_pii.iterrows():
            # Create a copy of the row to modify
            new_row = pii_row.to_dict()
            
            # Combine prompt and text
            # Adding a newline or space separator as appropriate. 
            # User example: "rewrite this text to be more concise + textfrompii_dataset"
            # I will add a newline for better separation, or just a space if preferred. 
            # Usually prompts like "Rewrite this:" are followed by the text.
            # I'll use a newline to be safe and clear, or just concatenation as requested.
            # User said "concat the text", I'll add a space separator.
            
            original_text = str(new_row.get('text', ''))
            combined_text = f"{prompt_text}\n\n{original_text}"
            
            new_row['text'] = combined_text
            
            # Optionally add the prompt metadata if useful, though user didn't explicitly ask for it as a column, 
            # they said "keep the other columns" (implying pii columns). 
            # I'll add the prompt info just in case it's needed for tracking.
            new_row['source_prompt'] = prompt_text
            if 'WritingToolCategory' in prompt_row:
                new_row['prompt_category'] = prompt_row['WritingToolCategory']
            
            new_rows.append(new_row)

    # Create new DataFrame
    df_combined = pd.DataFrame(new_rows)

    # Save to CSV
    df_combined.to_csv(output_path, index=False)
    print(f"Successfully created {output_path} with {len(df_combined)} rows.")

if __name__ == "__main__":
    generate_combined_dataset()
