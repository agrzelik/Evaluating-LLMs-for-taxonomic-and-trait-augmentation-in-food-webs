"""
Pairwise comparison of versions across different models.

Folder structure:
    <base_dir>/<model>/<version>/file.xlsx

Usage:
    python Inter_model_comparison.py \
        --base-dir "LLM features/Processed" \
        --versions v1 v2 v3

The script will make all pairwise combinations across (model, version) pairs,
i.e. it compares e.g. gemini/v1 vs chatgpt/v1, gemini/v1 vs claude/v2, etc.
Results are saved as a detailed CSV per model-pair + a summary file.
"""

import os
import argparse
import itertools
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────

MODELS = ['gemini', 'chatgpt', 'claude', 'qwen']

EXCLUDE_COLUMNS = {
    'file_name', 'node', 'biomass', 'import', 'export', 'Import',
    'typical_length', 'metabolic_rate', 'feeding_time_preference',
    'generation_time', 'life_span', 'age_at_maturity', 'trophic_level',
    '_id', '_normalized_file', 'Species_confidence', 'Genus_confidence', 'Order_confidence',
    'Class_confidence', 'Family_confidence', 'Phylum_confidence', 'included_species_latin',
    'included_species_english', 'data_source_for_included_species',
    'representative_species', 'data_source_for_representative_species',
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_filename(filename):
    filename = str(filename).strip()
    for suffix in ('.scor', '.xlsx', '.csv', '_combined_augmented', '_orig'):
        filename = filename.replace(suffix, '')
    return filename.lower() + '.scor'


def normalize_value(value):
    if pd.isna(value) or str(value).strip().lower() == '':
        return ''
    return str(value).strip().lower()

def new_normalize_series(value):
    if pd.isna(value):
        return ''
    elif (str(value).strip().lower() == ''):
        return ''
    else:
        return str(value).strip().lower()


# ── Loading ────────────────────────────────────────────────────────────────────

def load_xlsx(label, folder_path):
    xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    if not xlsx_files:
        raise FileNotFoundError(f"No XLSX file in: {folder_path}")
    path = os.path.join(folder_path, xlsx_files[0])
    print(f"    [{label}] {path}")
    try:
        return pd.read_excel(path, engine='calamine')
    except Exception:
        return pd.read_excel(path, engine='openpyxl')


# ── Align ──────────────────────────────────────────────────────────────────────

def align(df1: pd.DataFrame, df2: pd.DataFrame):
    for df in (df1, df2):
        df['_normalized_file'] = df['file_name'].apply(normalize_filename)
        df['_id'] = df['_normalized_file'] + '_' + df['node'].str.lower().astype(str)
        df = df[df['IsAlive']==True]

    common = set(df1['_id']) & set(df2['_id'])
    if not common:
        return pd.DataFrame(), pd.DataFrame()

    a = df1[df1['_id'].isin(common)].sort_values('_id').reset_index(drop=True)
    b = df2[df2['_id'].isin(common)].sort_values('_id').reset_index(drop=True)
    return a, b


# ── Metrics ────────────────────────────────────────────────────────────────────

def agreement_for_column(s1, s2, col, mask_df, model1, model2, node_id):
    y1 = s1.apply(new_normalize_series)
    y2 = s2.apply(new_normalize_series)

    current_ids = node_id

    mask_col_1 = f"{col}_{model1}"
    mask_col_2 = f"{col}_{model2}"

    row_mask_1 = pd.Series(True, index=y1.index)
    row_mask_2 = pd.Series(True, index=y2.index)

    if mask_col_1 in mask_df.columns:
        id_to_mask = mask_df.set_index('_id')[mask_col_1]
        row_mask_1 = current_ids.map(id_to_mask).fillna(False).astype(bool)

    if mask_col_2 in mask_df.columns:
        id_to_mask = mask_df.set_index('_id')[mask_col_2]
        row_mask_2 = current_ids.map(id_to_mask).fillna(False).astype(bool)

    y1 = y1[row_mask_1]
    y2 = y2[row_mask_2]

    if len(y1) == 0:
        return {'accuracy': None, 'n_samples': 0}

    return {
        'accuracy': accuracy_score(y1, y2),
        'n_samples': len(y1),
    }

# ── Pairwise comparison ────────────────────────────────────────────────────────

def compare_pair(label1, label2, df1, df2, mask_df) -> pd.DataFrame:
    """
    label1 / label2: strings like "gemini/v1", "chatgpt/v2"
    """
    a, b = align(df1.copy(), df2.copy())

    if a.empty:
        print(f"  No common records: {label1} vs {label2}")
        return pd.DataFrame()

    print(f"  {label1} vs {label2} — {len(a)} common records")

    exclude = ['IsAlive','representative_species_english', 'representative_species_latin','latin_name','Kingdom_confidence',]
    cols = [c for c in a.columns if c not in EXCLUDE_COLUMNS and c not in exclude and c in b.columns]
    rows = []

    for col in tqdm(cols, desc="   columns", leave=False):
        m = agreement_for_column(a[col], b[col],col, mask_df, label1.split('/')[0], label2.split('/')[0], a['_id'])
        model1, version1 = label1.split('/', 1)
        model2, version2 = label2.split('/', 1)
        rows.append({
            'comparison': f"{label1} vs {label2}",
            'model1': model1,
            'version1': version1,
            'model2': model2,
            'version2': version2,
            'column': col,
            'accuracy': m['accuracy'],
            'n_samples': m['n_samples'],
        })

    result = pd.DataFrame(rows).dropna(subset=['accuracy'])
    print(f"  avg agreement: {result['accuracy'].mean():.4f}  ({len(result)} columns)")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Agreement across different models, pairwise between (model, version) pairs"
    )
    parser.add_argument('--base-dir', required=False, default="LLM features/Processed",
                        help='Parent folder with model subfolders, e.g., "LLM features/Processed"')
    parser.add_argument('--versions', nargs='+', required=True,
                        help='Names of version subfolders, e.g. v1 v2 v3')
    parser.add_argument('--models', nargs='+', default=MODELS,
                        help=f'Models to process (default: {MODELS})')
    parser.add_argument('--output-dir', default='Inter_Model_Divergence',
                        help='Results folder')
    parser.add_argument('--same-version-only', action='store_true',
                        help='Only compare pairs that share the same version (e.g. gemini/v1 vs chatgpt/v1)')
    parser.add_argument('--nanstrategy', default=2,
                        help='Nan cutting option')
    parser.add_argument('--option_comparison', default='inter',
                       help='inter or intra model')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load all available (model, version) datasets ──────────────────────────
    print("\nLoading datasets...")
    datasets = {}  # key: "model/version", value: DataFrame
    ns = args.nanstrategy
    option_comparison = args.option_comparison


    mask_df = pd.read_csv(f'Comparisons/valid_masks_gen1_0603_nanstrategy_{ns}.csv', sep=';')
    mask_df['_normalized_file'] = mask_df['file_name'].apply(normalize_filename)
    mask_df['_id'] = mask_df['_normalized_file'] + '_' + mask_df['node'].astype(str)


    for model in args.models:
        model_dir = os.path.join(args.base_dir, model)
        if not os.path.isdir(model_dir):
            print(f"  No folder: {model_dir} — skipping")
            continue
        for version in args.versions:
            folder = os.path.join(model_dir, version)
            if not os.path.isdir(folder):
                continue
            label = f"{model}/{version}"
            try:
                datasets[label] = load_xlsx(label, folder)
            except Exception as e:
                print(f"  Loading error {label}: {e}")

    if len(datasets) < 2:
        print(f"Need at least 2 datasets, found {len(datasets)}. Check paths.")
        return

    print(f"\nLoaded {len(datasets)} datasets: {list(datasets.keys())}")

    # ── Build pairs ───────────────────────────────────────────────────────────
    
    all_labels = list(datasets.keys())
    pairs = []

    for label1, label2 in itertools.combinations(all_labels, 2):
        model1 = label1.split('/')[0]
        model2 = label2.split('/')[0]
        version1 = label1.split('/')[1]
        version2 = label2.split('/')[1]

        # Skip intra-model pairs (those are handled by Intra_model_comparison.py)
        if model1 == model2:
            if option_comparison == 'intra':
                pairs.append((label1, label2))
            else: 
                continue

        # Optionally restrict to same-version pairs only
        if args.same_version_only and version1 != version2:
            continue

        if option_comparison == 'inter':
            pairs.append((label1, label2))

    print(f"\n{len(pairs)} inter-model pairs to compare.")
    

    # ── Run comparisons ───────────────────────────────────────────────────────
    all_results = []
    # Group pairs by (model1, model2) for per-pair-of-models output files
    pair_model_results: dict[str, list] = {}

    for label1, label2 in pairs:
        print(f"\n{'─'*60}")
        result = compare_pair(label1, label2, datasets[label1], datasets[label2], mask_df)
        if result.empty:
            continue

        all_results.append(result)

        model1 = label1.split('/')[0]
        model2 = label2.split('/')[0]
        # Canonical key (sorted so gemini_chatgpt == chatgpt_gemini)
        key = '_'.join(sorted([model1, model2]))
        pair_model_results.setdefault(key, []).append(result)

    if not all_results:
        print("\nNo results found. Check your paths and folder structure.")
        return

    # ── Save per model-pair CSVs ──────────────────────────────────────────────
    for key, results in pair_model_results.items():
        df = pd.concat(results, ignore_index=True)
        out_path = os.path.join(args.output_dir, f'{option_comparison}_agreement_{key}.csv')
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}  ({len(df)} rows)")

    # ── Save combined file ────────────────────────────────────────────────────
    final = pd.concat(all_results, ignore_index=True)
    out_all = os.path.join(args.output_dir, f'{option_comparison}_agreement_all_inter_model_nanstrategy_{ns}.csv')
    final.to_csv(out_all, index=False, sep=';')
    print(f"\nBatch file: {out_all}  ({len(final)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    summary = (
        final.groupby(['model1', 'version1', 'model2', 'version2'])['accuracy']
        .agg(mean_agreement='mean', n_columns='count')
        .round(4)
        .reset_index()
    )
    print(summary.to_string(index=False))

    # Also a condensed model-pair summary (averaged over versions)
    print(f"\n{'─'*60}")
    print("CONDENSED (averaged over all version combinations per model pair)")
    print(f"{'─'*60}")
    final['model_pair'] = final.apply(
        lambda r: ' vs '.join(sorted([r['model1'], r['model2']])), axis=1
    )
    condensed = (
        final.groupby('model_pair')['accuracy']
        .agg(mean_agreement='mean', n_columns='count')
        .round(4)
        .reset_index()
    )
    print(condensed.to_string(index=False))


if __name__ == '__main__':
    main()