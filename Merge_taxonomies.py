import pandas as pd
import glob
import os
from functools import reduce
import numpy as np
import argparse

# === CONFIGURATION ===
MODELS = ['chatgpt', 'claude', 'gemini', 'qwen']
SCORERS = ['chatgpt', 'claude', 'gemini', 'qwen', 'all']
TAX_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def parse_args():
    parser = argparse.ArgumentParser(description='Process food web taxonomic data.')
    parser.add_argument('--version', type=str, required=True, help='Version string, e.g. v1')
    return parser.parse_args()

args = parse_args()
VERSION = args.version

PROCESSED_DIR = "LLM features/Processed"
BASELINE_PATH = "baseline/baseline_file_1.csv"
CONSENSUS_PATH = f"Ensembles/self_evaluated_soft_voting/{VERSION}/consensus_results_{VERSION}.csv"
ARBITRATED_PATH = f"Ensembles/black_box_approach/{VERSION}/arbitrated_taxonomy_all_models.xlsx"
WORMS_DIR = "LLM_worms_files"
OPTION3_DIR = "Ensembles/cross_evaluated_soft_voting"
OUTPUT_PATH = f"Comparisons/full_df_{VERSION}.csv"

TAX_RENAME = {
    'Kingdom': 'kingdom', 'Phylum': 'phylum', 'Class': 'class',
    'Order': 'order', 'Family': 'family', 'Genus': 'genus', 'Species': 'species'
}

def standardize_keys(df):
    df["node"] = df["node"].astype(str).str.strip().str.lower()
    df["file_name"] = df["file_name"].astype(str).str.strip().str.lower().str.replace('.scor', '', regex=False)
    return df

# === 1. Loading LLM files (one per model) ===
dfs = []
for model in MODELS:
    pattern = os.path.join(PROCESSED_DIR, f"{model}", f"{VERSION}", f"{model}_Processed_Final.xlsx")
    matches = glob.glob(pattern, recursive=False)
    if not matches:
        print(f"No file for model: {model}")
        continue

    path = matches[0]
    print(f"Read {model}: {path}")
    df = pd.read_excel(path)
    df = standardize_keys(df)

    rename_dict = {}
    cols_to_keep = ["node", "file_name", "IsAlive"]
    for col in df.columns:
        if col.lower().strip() in TAX_LEVELS:
            rename_dict[col] = f"{col.lower().strip()}_{model}"
            cols_to_keep.append(col)

    df = df[cols_to_keep].rename(columns=rename_dict)
    dfs.append(df)

df_tax = reduce(
    lambda left, right: pd.merge(left, right, on=["node", "file_name", "IsAlive"], how="outer"),
    dfs
)

# === 2. Self-evaluated soft voting ===
df_cons = pd.read_csv(CONSENSUS_PATH, sep=';')
df_cons = standardize_keys(df_cons)

cons_rename = {}
for col in df_cons.columns:
    if col.startswith("best_"):
        cons_rename[col] = f"{col.replace('best_', '')}_consensus"
    elif "_score" in col:
        cons_rename[col] = f"{col.replace('_consensus_score','').replace('_score','')}_score_consensus"

df_cons = df_cons.rename(columns=cons_rename)
cols_cons = ["node", "file_name"] + list(cons_rename.values())
df_cons = df_cons[[c for c in cols_cons if c in df_cons.columns]]

df_step1 = pd.merge(df_tax, df_cons, on=["node", "file_name"], how="outer")

# === 3. Baseline ===
baseline_df = pd.read_csv(BASELINE_PATH, sep=';')
baseline_df = baseline_df.rename(columns={'Node name': 'node'})
baseline_df = baseline_df.rename(columns={'FileName': 'file_name'})
baseline_df = standardize_keys(baseline_df)

baseline_rename = {k: f"{v}_baseline" for k, v in TAX_RENAME.items()}
cols_baseline = ["node", "file_name"] + [c for c in TAX_RENAME.keys() if c in baseline_df.columns] + ["IsAlive"]
baseline_df = baseline_df[cols_baseline].rename(columns=baseline_rename)


print(df_step1['IsAlive'].isna().sum())
print(baseline_df['IsAlive'].isna().sum())
print(df_step1['kingdom_consensus'].isna().sum())
print(baseline_df['kingdom_baseline'].isna().sum())
print(df_step1.shape)
print(baseline_df.shape)

df_step2 = pd.merge(df_step1, baseline_df, on=["node", "file_name", "IsAlive"], how="right")

# === 4. Black-box approach ===
df_arb = pd.read_excel(ARBITRATED_PATH)
df_arb = standardize_keys(df_arb)
df_step3 = pd.merge(df_step2, df_arb, on=["node", "file_name"], how="left")

# === 5. WoRMS+LLMs ===
def load_worms_llm(model):
    path = f"{WORMS_DIR}/processed_taxonomic_data_{VERSION}_{model}.csv"
    df = pd.read_csv(path, sep=';')
    df["node"] = df["OriginalName"].str.strip().str.lower()
    df["file_name"] = df["FileName"].str.strip().str.lower().str.replace('.scor', '', regex=False)
    rename = {k: f"{v}_worms_llm_{model}" for k, v in TAX_RENAME.items()}
    cols = ["node", "file_name"] + [c for c in TAX_RENAME.keys() if c in df.columns]
    return df[cols].rename(columns=rename).set_index(["node", "file_name"])

df_worms = pd.concat(
    [load_worms_llm(m) for m in MODELS], axis=1, join="outer"
).reset_index()

df_final_a = pd.merge(df_step3, df_worms, on=["node", "file_name"], how="left")

# === 6. Cross-model evaluated soft voting ===
def load_option3(scorer):
    path = f"{OPTION3_DIR}/{VERSION}/best_scoring_{scorer}.csv"
    df = pd.read_csv(path, sep=';')
    df = standardize_keys(df)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    tax_cols = [col for col in df.columns
                if any(f"{t}_best" in col for t in TAX_LEVELS) and "_score" not in col]
    return df[["node", "file_name"] + tax_cols].set_index(["node", "file_name"])

df_opt3 = pd.concat(
    [load_option3(s) for s in SCORERS], axis=1, join="outer"
).reset_index()

df_final = pd.merge(df_final_a, df_opt3, on=["node", "file_name"], how="left")
#df_final = df_final[df_final['IsAlive']==True]

# === Save ===
os.makedirs("Comparisons", exist_ok=True)
df_final.to_csv(OUTPUT_PATH, sep=';', index=False)
print(f"Saved: {OUTPUT_PATH} ({len(df_final)} rows, {len(df_final.columns)} columns)")