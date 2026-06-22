import os
import numpy as np
import pandas as pd
import argparse
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score

# === CONFIGURATION ===
def parse_args():
    parser = argparse.ArgumentParser(description='Process food web taxonomic data.')
    parser.add_argument('--version', type=str, required=True, help='Version string, e.g. v1')
    parser.add_argument('--nanstrategy', type=str, required=True, help='1 for replacing all nan with empty strings and comparing all, 2 for remove observations missing in baseline, 3 for removing observations with nan for either llm or baseline')
    return parser.parse_args()

args = parse_args()
VERSION = args.version
nanstrategy = args.nanstrategy

INPUT_PATH = f'Comparisons/full_df_{VERSION}.csv'
OUTPUT_DIR = 'Comparisons'
N_BOOTSTRAP = 7000 #originally 10000, changed for RAM purposes

TAX_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

worms_models = ["chatgpt", "claude", "gemini", "qwen"]
scorers      = ["chatgpt", "claude", "gemini", "qwen", "all"]

MODELS = (
    ["chatgpt", "claude", "gemini", "qwen", "consensus",
     "arbitrated_gpt", "arbitrated_claude", "arbitrated_qwen", "arbitrated_gemini"]
    + [f"worms_llm_{m}" for m in worms_models]
    + [f"best_{s}" for s in scorers]
)

df = pd.read_csv(INPUT_PATH, sep=';')
print(f"Read {len(df)} rows, {len(df.columns)} columns")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# === METRICS ===
def new_normalize_series(value):
    if pd.isna(value):
        return ''
    elif (str(value).strip().lower() == ''):
        return ''
    else:
        return str(value).strip().lower()


def get_valid_indices(baseline_col, llm_col, nanstrategy_mode):
    baseline = baseline_col.apply(new_normalize_series)
    llm      = llm_col.apply(new_normalize_series)

    if nanstrategy_mode == '1':
        mask = pd.Series([True] * len(baseline), index=baseline.index)
    elif nanstrategy_mode == '2':
        mask = baseline != ''
    elif nanstrategy_mode == '3':
        mask = (baseline != '') & (llm != '')

    return mask

def compute_metrics(baseline_col, llm_col):
    """
    Returns (agreement=1-JSD, accuracy=exact_match%)
    only for records where baseline has a value.
    """
   
    baseline = baseline_col.apply(new_normalize_series)
    llm      = llm_col.apply(new_normalize_series)

    '''
    hallucination_mask = (baseline == '') & (llm != '')
    hallucination_rate = hallucination_mask.sum() / len(baseline == '')
    '''
    
    n = len(baseline)

    both_valid    = (baseline == llm).sum() / n                   # baseline == llm, answer or empty
    hallucination_rate = ((baseline == '') & (llm != '')).sum() / n    # baseline empty, LLM gives an overspecified answer
    missed_rate        = ((baseline != '') & (llm == '')).sum() / n    # baseline with answer, LLM empty
    incorrect_rate     = ((baseline != '') & (llm != '') & (baseline != llm)).sum() / n     #incorrect LLM answer
        
    if nanstrategy == '1':
        pass
    elif nanstrategy == '2': 
        valid = baseline != ''
        baseline = baseline[valid]
        llm      = llm[valid]
    elif nanstrategy == '3':
        valid = (baseline != '') & (llm != '')
        baseline = baseline[valid]
        llm      = llm[valid]

    # Agreement (1 - JSD)
    p = baseline.value_counts(normalize=True)
    q = llm.value_counts(normalize=True)
    idx = p.index.union(q.index)
    p = p.reindex(idx, fill_value=0)
    q = q.reindex(idx, fill_value=0)
    agreement = 1 - jensenshannon(p, q)

    if nanstrategy == '2':
        n = len(baseline)
        missed_rate        = ((baseline != '') & (llm == '')).sum() / n    # baseline with answer, LLM empty
        incorrect_rate     = ((baseline != '') & (llm != '') & (baseline != llm)).sum() / n     #incorrect LLM answer
        hallucination_rate = np.nan

    # Accuracy (sklearn)
    accuracy = accuracy_score(baseline,llm)

    return agreement, accuracy, hallucination_rate, missed_rate, incorrect_rate

# === BOOTSTRAP ===

def get_bootstrap_ci(df, models, tax_levels, n_iterations=1000):
    records = []

    for i in range(n_iterations):
        boot = df.sample(frac=1, replace=True)

        for model in models:
            for tax in tax_levels:
                b_col = f"{tax}_baseline"
                l_col = f"{tax}_{model}"

                if b_col not in boot.columns or l_col not in boot.columns:
                    continue

                agreement, accuracy, hallucination_rate, missed_rate, incorrect_rate = compute_metrics(boot[b_col], boot[l_col])
                records.append({
                    "boot_id":       i,
                    "model":         model,
                    "taxonomy_level": tax,
                    "agreement":     agreement,
                    "accuracy":      accuracy,
                    "hallucination_rate": hallucination_rate,
                    "missed_rate": missed_rate,
                    "incorrect_rate": incorrect_rate

                })

        if (i + 1) % 100 == 0:
            print(f"Bootstrap {i+1}/{n_iterations}")

    return pd.DataFrame(records)

# === METRICS ON THE FULL DATASET ===
results = []
per_column_masks = {}

for model in MODELS:
    for tax in TAX_LEVELS:
        b_col = f"{tax}_baseline"
        l_col = f"{tax}_{model}"
        if b_col not in df.columns or l_col not in df.columns:
            continue

        key = f"{tax}_{model}"
        per_column_masks[key] = get_valid_indices(df[b_col], df[l_col], nanstrategy)

        agreement, accuracy, hallucination_rate, missed_rate, incorrect_rate = compute_metrics(df[b_col], df[l_col])
        results.append({
            "model": model,
            "taxonomy_level": tax,
            "agreement": agreement,
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "missed_rate": missed_rate,
            "incorrect_rate": incorrect_rate
        })

df_full = pd.DataFrame(results)

# === SAVE MASK ===
mask_df = pd.DataFrame(per_column_masks, index=df.index)
mask_df['node'] = df['node'].values
mask_df['file_name'] = df['file_name'].values
mask_df.to_csv(f'{OUTPUT_DIR}/valid_masks_{VERSION}_nanstrategy_{nanstrategy}.csv', sep=';')

# === BOOTSTRAP CI ===
df_boot = get_bootstrap_ci(df, MODELS, TAX_LEVELS, n_iterations=N_BOOTSTRAP)
summary = df_boot.groupby(['model', 'taxonomy_level']).agg(
    agreement_ci_low  = ('agreement', lambda x: x.quantile(0.025)),
    agreement_ci_high = ('agreement', lambda x: x.quantile(0.975)),
    accuracy_ci_low   = ('accuracy',  lambda x: x.quantile(0.025)),
    accuracy_ci_high  = ('accuracy',  lambda x: x.quantile(0.975)),
    hallucination_rate_ci_low   = ('hallucination_rate',  lambda x: x.quantile(0.025)),
    hallucination_rate_ci_high  = ('hallucination_rate',  lambda x: x.quantile(0.975)),
    missed_rate_ci_low   = ('missed_rate',  lambda x: x.quantile(0.025)),
    missed_rate_ci_high  = ('missed_rate',  lambda x: x.quantile(0.975)),
    incorrect_rate_ci_low   = ('incorrect_rate',  lambda x: x.quantile(0.025)),
    incorrect_rate_ci_high  = ('incorrect_rate',  lambda x: x.quantile(0.975)),
    
).reset_index()

# === MERGE ===
df_final = pd.merge(df_full, summary, on=['model', 'taxonomy_level'], how='left')
df_final.to_csv(f'{OUTPUT_DIR}/llm_scores_{VERSION}_nanstrategy_{nanstrategy}.csv', sep=';', index=False)

