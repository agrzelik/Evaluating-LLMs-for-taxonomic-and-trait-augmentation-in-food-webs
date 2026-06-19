import pandas as pd
import glob
import os
from functools import reduce
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score

# === CONFIGURATION ===
MODELS = ['chatgpt']
SCORERS = ['chatgpt', 'claude', 'gemini', 'qwen', 'all']
TAX_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
VERSION_PROMPTING = 'gen1_0603'

PROCESSED_DIR = "LLM features/Processed"
BASELINE_PATH = "baseline/baseline_file_1.csv"
WORMS_DIR = "LLM_worms_files"
OPTION3_DIR = "Ensembles/cross_evaluated_soft_voting"
OUTPUT_PATH = f"Separate_session_prompting/SEPARATE_SESSION_full_df.csv"

TAX_RENAME = {
    'Kingdom': 'kingdom', 'Phylum': 'phylum', 'Class': 'class',
    'Order': 'order', 'Family': 'family', 'Genus': 'genus', 'Species': 'species'
}

TAX_RANK_ORDER = {tax: i for i, tax in enumerate(TAX_LEVELS)}

label_dict = {
    'accuracy_chatgpt': 'Classic prompting',
    'accuracy_chat_gpt_separate_session': 'Taxonomy pruning'
}

def standardize_keys(df):
    df["node"] = df["node"].astype(str).str.strip().str.lower()
    df["file_name"] = df["file_name"].astype(str).str.strip().str.lower().str.replace('.scor', '', regex=False)
    return df

# === 1. Loading separate session LLM file (one per model) ===
df_cons = pd.read_csv('Separate_session_prompting/Taxonomies_from_separate_session.csv')
df_cons = standardize_keys(df_cons)

# === 2. Classic prompting ===


for model in MODELS:
    pattern = os.path.join(PROCESSED_DIR, f"{model}", f"{VERSION_PROMPTING}", f"{model}_Processed_Final.xlsx")
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

    df_0 = df[cols_to_keep].rename(columns=rename_dict)

df1 = pd.merge(df_cons, df_0, on=["node", "file_name"], how="outer", suffixes=("_separate_session", "") )

# === 3. Baseline ===
baseline_df = pd.read_csv(BASELINE_PATH, sep=';')
baseline_df = baseline_df.rename(columns={'Node name': 'node'})
baseline_df = baseline_df.rename(columns={'FileName': 'file_name'})
baseline_df = standardize_keys(baseline_df)

baseline_rename = {k: f"{v}_baseline" for k, v in TAX_RENAME.items()}
cols_baseline = ["node", "file_name"] + [c for c in TAX_RENAME.keys() if c in baseline_df.columns] + ["IsAlive"]
baseline_df = baseline_df[cols_baseline].rename(columns=baseline_rename)

df_final = pd.merge(df1, baseline_df, on=["node", "file_name",], how="right")

# === Save ===
os.makedirs("Comparisons", exist_ok=True)
df_final.to_csv(OUTPUT_PATH, sep=';', index=False)
print(f"Saved: {OUTPUT_PATH} ({len(df_final)} rows, {len(df_final.columns)} columns)")

TAX_LEVELS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
VERSION = "SEPARATE_SESSION"

def normalize(value):
    if pd.isna(value) or str(value).strip() == '':
        return ''
    return str(value).strip().lower()

def get_deepest_rank(row, suffix):
    for tax in reversed(TAX_LEVELS):
        val = normalize(row.get(f"{tax}{suffix}", ''))
        if val != '':
            return tax
    return None

def compute_pruning_stats(df):
    mask = df.apply(lambda row: get_deepest_rank(row, '_baseline') is not None, axis=1)
    df_ann = df[mask].copy()
    df_ann['baseline_rank'] = df_ann.apply(lambda row: get_deepest_rank(row, '_baseline'), axis=1)
    df_ann['chat_rank']    = df_ann.apply(lambda row: get_deepest_rank(row, '_chatgpt_separate_session'), axis=1)

    def classify(row):
        j, c = row['baseline_rank'], row['chat_rank']
        if c is None: return 'over_pruning'
        if TAX_RANK_ORDER[c] < TAX_RANK_ORDER[j]: return 'over_pruning'
        if TAX_RANK_ORDER[c] == TAX_RANK_ORDER[j]: return 'correct_pruning'
        return 'under_pruning'

    df_ann['pruning_label'] = df_ann.apply(classify, axis=1)

    rows = []
    for rank in TAX_LEVELS:
        s = df_ann[df_ann['baseline_rank'] == rank]
        n = len(s)
        under   = (s['pruning_label'] == 'under_pruning').sum()
        correct = (s['pruning_label'] == 'correct_pruning').sum()
        over    = (s['pruning_label'] == 'over_pruning').sum()
        rows.append([rank.capitalize(), n,
                     f"{under} ({under/n*100:.0f}%)" if n else "0 (0%)",
                     f"{correct} ({correct/n*100:.0f}%)" if n else "0 (0%)",
                     f"{over} ({over/n*100:.0f}%)" if n else "0 (0%)"])

    n = len(df_ann)
    under   = (df_ann['pruning_label'] == 'under_pruning').sum()
    correct = (df_ann['pruning_label'] == 'correct_pruning').sum()
    over    = (df_ann['pruning_label'] == 'over_pruning').sum()
    rows.append(['Total', n,
                 f"{under} ({under/n*100:.0f}%)",
                 f"{correct} ({correct/n*100:.0f}%)",
                 f"{over} ({over/n*100:.0f}%)"])

    result_df = pd.DataFrame(rows, columns=['Rank', 'N', 'Under-pruning', 'Correct pruning', 'Over-pruning'])
    return result_df

def compute_metrics(baseline_col, llm_col, nanstrategy):
    baseline = baseline_col.apply(normalize)
    llm      = llm_col.apply(normalize)
    n        = len(baseline)

    hallucination_rate = ((baseline == '') & (llm != '')).sum() / n
    missed_rate        = ((baseline != '') & (llm == '')).sum() / n
    incorrect_rate     = ((baseline != '') & (llm != '') & (baseline != llm)).sum() / n

    if nanstrategy == '2':
        valid    = baseline != ''
        baseline = baseline[valid]
        llm      = llm[valid]
        hallucination_rate = np.nan

    elif nanstrategy == '3':
        valid    = (baseline != '') & (llm != '')
        baseline = baseline[valid]
        llm      = llm[valid]

    p   = baseline.value_counts(normalize=True)
    q   = llm.value_counts(normalize=True)
    idx = p.index.union(q.index)
    p   = p.reindex(idx, fill_value=0)
    q   = q.reindex(idx, fill_value=0)
    agreement = 1 - jensenshannon(p, q)
    accuracy  = accuracy_score(baseline, llm)

    return agreement, accuracy, hallucination_rate, missed_rate, incorrect_rate


def make_heatmap(df_final, nanstrategy, option, version):

    taxonomy_order = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    results = []
    for tax in TAX_LEVELS:
        b_col = f"{tax}_baseline"
        l_col = f"{tax}_chatgpt_separate_session"
        c_col = f"{tax}_chatgpt"
        if b_col not in df_final.columns or l_col not in df_final.columns:
            continue

        agreement_l, accuracy_l, hall_l, missed_l, incorrect_l = compute_metrics(df_final[b_col], df_final[l_col], nanstrategy)

        agreement_c, accuracy_c, hall_c, missed_c, incorrect_c = compute_metrics(df_final[b_col], df_final[c_col], nanstrategy)

        metrics_l = {
            'agreement': agreement_l,
            'accuracy': accuracy_l,
            'hallucination': hall_l,
            'missed': missed_l,
            'incorrect': incorrect_l,
        }

        metrics_c = {
            'agreement': agreement_c,
            'accuracy': accuracy_c,
            'hallucination': hall_c,
            'missed': missed_c,
            'incorrect': incorrect_c,
        }

        row = {'taxonomy_level': tax}

        for metric in metrics_l:
            row[f'{metric}_chat_gpt_separate_session'] = metrics_l[metric]
            row[f'{metric}_chatgpt'] = metrics_c[metric]
            #row[f'{metric}_diff'] = metrics_c[metric] - metrics_l[metric]

        results.append(row)
        df_scores = pd.DataFrame(results)

    # ── pivot ──────────────────────────────────────────────────────────────────
    heatmap_data = df_scores.set_index('taxonomy_level')[
        ['accuracy_chatgpt',
        'accuracy_chat_gpt_separate_session']
    ]

    heatmap_data = heatmap_data.reindex(taxonomy_order)

    heatmap_anno = heatmap_data.map(
        lambda v: f"{v*100:.0f}" if pd.notna(v) else "")

    #heatmap_data = df_scores.pivot(index='taxonomy_level', columns='model', values=option)
    #heatmap_data = heatmap_data.reindex(taxonomy_order)
    '''
    anno_df       = df_scores.copy()
    anno_df['anno'] = anno_df[option].apply(lambda v: f"{v*100:.0f}" if pd.notna(v) else "")
    heatmap_anno  = anno_df.pivot(index='taxonomy_level', columns='model', values='anno')
    heatmap_anno  = heatmap_anno.reindex(taxonomy_order)
    '''
    # ── plot ───────────────────────────────────────────────────────────────────
    cmap_dict = {
        'agreement':      'Blues',
        'accuracy':       'Blues',
        'hallucination_rate': 'Reds',
        'missed_rate':    'Oranges',
        'incorrect_rate': 'Purples',
    }
    title_dict = {
        'agreement':      'Agreement (1 − JSD)',
        'accuracy':       'Accuracy',
        'hallucination_rate': 'Overspecification rate',
        'missed_rate':    'Missed rate',
        'incorrect_rate': 'Incorrect rate',
    }

    heatmap_data = heatmap_data.rename(columns=label_dict)
    heatmap_anno = heatmap_anno.rename(columns=label_dict)


    fig, ax = plt.subplots(figsize=(4, 6))

    sns.heatmap(
        heatmap_data,
        annot=False,
        vmin=0, vmax=1,
        cmap=cmap_dict.get(option, 'Blues'),
        cbar=False,
        linewidths=0.5,
        ax=ax
    )

    for i, row_label in enumerate(heatmap_data.index):
        for j, col_label in enumerate(heatmap_data.columns):
            val  = heatmap_data.loc[row_label, col_label]
            anno = heatmap_anno.loc[row_label, col_label]
            if not anno:
                continue
            text_color = 'white' if (pd.notna(val) and val > 0.7) else 'black'
            ax.text(j + 0.5, i + 0.5, anno,
                    ha='center', va='center',
                    fontsize=16, fontweight='bold', color=text_color)
            
            
    #ax.set_title(f"{title_dict.get(option, option)}\n(nanstrategy {nanstrategy})",fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('')
    #ax.set_ylabel('Taxonomic level', fontsize=12)
    ax.set_ylabel('')
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    if nanstrategy=='1':
        ax.set_yticklabels([l.get_text().capitalize() for l in ax.get_yticklabels()], rotation=0)
    else:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.tight_layout()
    out = f'Figures/heatmap_{option}_{version}_nanstrategy{nanstrategy}.pdf'
    plt.savefig(out, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# === GENERATE ===
options = ['accuracy'] 

for nanstrategy in ['1', '2']:
    for option in options:
        if nanstrategy == '2' and option == 'hallucination_rate':
            continue
        make_heatmap(df_final, nanstrategy, option, VERSION)

pruning_table = compute_pruning_stats(df_final)
print(pruning_table.to_string(index=False))
pruning_table.to_csv("Separate_session_prompting/pruning_statistics.csv", sep=';', index=False)


print("Done!")