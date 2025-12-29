import pandas as pd
import ast
from collections import Counter

def count_piis(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    total_piis = 0
    pii_counts = Counter()

    if 'labels' not in df.columns:
        print("Error: 'labels' column not found in the dataset.")
        return

    print(f"Total number of entries in the dataset: {len(df)}")

    for index, row in df.iterrows():
        labels_str = row['labels']
        try:
            # The labels are stored as a string representation of a list
            labels = ast.literal_eval(labels_str)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse labels for row {index}")
            continue
        
        if not isinstance(labels, list):
            continue

        for label in labels:
            if label.startswith('B-'):
                total_piis += 1
                # Extract the PII type (e.g., NAME_STUDENT from B-NAME_STUDENT)
                pii_type = label[2:]
                pii_counts[pii_type] += 1

    print(f"Total number of PII entities found: {total_piis}")
    print("\nBreakdown by PII type:")
    for pii_type, count in pii_counts.most_common():
        print(f"{pii_type}: {count}")

if __name__ == "__main__":
    dataset_path = "combined_dataset.csv"
    count_piis(dataset_path)
