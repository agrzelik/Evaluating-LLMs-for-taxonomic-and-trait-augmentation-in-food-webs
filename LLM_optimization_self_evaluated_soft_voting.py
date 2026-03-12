import pandas as pd
import os
import ast
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process food web taxonomic data.')
    parser.add_argument('--version', type=str, required=True, help='Version string, e.g. v1')
    return parser.parse_args()

args = parse_args()
version = args.version

model_config = {
    'chatgpt': 'Chatgpt',
    'claude': 'Claude',
    'gemini': 'Gemini',
    'qwen': 'Qwen'
}

processed_dfs = []
models = list(model_config.keys())

# Load and rename columns
for model_name, folder in model_config.items():
    path = f'LLM features/Processed/{folder}/{version}/{folder.lower()}_Processed_Final.xlsx'
    df = pd.read_excel(path)
    
    cols_to_rename = {col: f"{col}_{model_name}" for col in df.columns if col not in ['file_name', 'node']}
    processed_dfs.append(df.rename(columns=cols_to_rename))

# Merge
final_df = processed_dfs[0]
for df in processed_dfs[1:]:
    final_df = pd.merge(final_df, df, on=['file_name', 'node'], how='outer')

# Helper function to safely convert confidence to float
def safe_float_convert(conf):
    """Convert confidence value to float, handling lists and strings."""
    if pd.isna(conf):
        return None
    
    try:
        return float(conf)
    except (ValueError, TypeError):
        # Handle string representations of lists like '[1.0, 1.0]'
        try:
            if isinstance(conf, str):
                # Try to parse as literal
                parsed = ast.literal_eval(conf)
                if isinstance(parsed, list):
                    # Take mean of list values
                    return sum(float(x) for x in parsed) / len(parsed)
                return float(parsed)
            elif isinstance(conf, list):
                # Already a list
                return sum(float(x) for x in conf) / len(conf)
        except:
            return None
    
    return None

# Function to get best value with consensus score
def get_best_value(row, target_col, conf_col_base):
    scores = {}
    for m in models:
        v_col = f"{target_col}_{m}"
        c_col = f"{conf_col_base}_{m}"
        
        if v_col in row and c_col in row:
            val = row[v_col]
            conf = row[c_col]
            
            if pd.notna(val) and str(val).lower() != 'nan':
                conf_float = safe_float_convert(conf)
                
                if conf_float is not None:
                    val_clean = str(val).strip()
                    scores[val_clean] = scores.get(val_clean, 0) + conf_float
    
    if not scores:
        return None, 0
    
    best_val = max(scores, key=scores.get)
    avg_score = scores[best_val] / len(models)
    return best_val, avg_score

taxonomic_levels = [
    ('kingdom', 'Kingdom_confidence'),
    ('phylum', 'Phylum_confidence'),
    ('class', 'Class_confidence'),
    ('order', 'Order_confidence'),
    ('family', 'Family_confidence'),
    ('genus', 'Genus_confidence'),
    ('species', 'Species_confidence')
]

# Apply consensus scoring for all taxonomic levels
for value_col, conf_col in taxonomic_levels:
    best_col = f'best_{value_col}'
    score_col = f'{value_col}_consensus_score'
    
    final_df[[best_col, score_col]] = final_df.apply(
        lambda r: pd.Series(get_best_value(r, value_col, conf_col)), axis=1
    )

# Display results
display_cols = ['file_name', 'node']
for value_col, _ in taxonomic_levels:
    display_cols.extend([f'best_{value_col}', f'{value_col}_consensus_score'])

final_cols = ['file_name', 'node', 'best_kingdom', 'kingdom_consensus_score', 'best_phylum', 'phylum_consensus_score',
        'best_class', 'class_consensus_score','best_order', 'order_consensus_score', 'best_family',
       'family_consensus_score', 'best_genus', 'genus_consensus_score', 'best_species', 'species_consensus_score']

end_df = final_df[final_cols]
output_dir = f'Ensembles/self_evaluated_soft_voting/{version}'
os.makedirs(output_dir, exist_ok=True)
end_df.to_csv(f'Ensembles/self_evaluated_soft_voting/{version}/consensus_results_{version}.csv', sep=';')
