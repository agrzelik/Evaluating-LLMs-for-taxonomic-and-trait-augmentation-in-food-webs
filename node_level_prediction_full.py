"""
node_level_prediction.py
========================
Node-level classification on marine food web networks.

Three tasks (all on the same node set – nodes matched in LLM features):
  1. trophic_category  – derived from trophic_level using Stergiou & Karpouzi (2002) bins
  2. functional_group  – directly from LLM feature file
  3. feeding_guild     – directly from LLM feature file

Two feature sets compared per task:
  - v1  : graph-structural features (topology + flows), NO trophic_level
  - f1  : LLM taxonomic/functional features, NO trophic_level, NO v1 columns

Baseline: majority classifier (most frequent class / total nodes)

Train/test split: Stratified Group K-Fold by network (file_name),
so nodes from the same network never appear in both train and test.

Output: one heatmap (rows = tasks, cols = majority/v1/f1), CSV summary.
"""

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import foodwebviz as fw
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.dummy import DummyClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH  = 'dataset_20260126_ecobase/processed/'
METADATA_PATH = 'dataset_20260126_ecobase/metadata_20260128.xlsx'
OUTPUT_PATH   = 'Node_classification'
FIG_OUTPUT = 'Figures'

LLM_MODELS = {
    'gemini':    'Input/functional_features_0803/Gemini_Processed_Final.xlsx',
}

TO_REMOVE = []#['Grand_Banks_782.scor']

# Trophic level bins - categories  (Stergiou & Karpouzi, 2002)
TROPHIC_BINS   = [0.0, 1.05, 2.1, 2.9, 3.7, 4.5, 99.0]
TROPHIC_LABELS = ['primary_producer', 'herbivore', 'omnivore_plant',
                  'omnivore_animal', 'carnivore', 'top_predator']

# Columns that are targets, never used as features
TARGET_COLS = frozenset([
    'trophic_level', 'trophic_category',
    'functional_group', 'feeding_guild',
])

# v1 feature set: graph topology
V1_FEATURES = [
    'topo_in_degree', 'topo_out_degree', 'topo_degree',
    'topo_betweenness', 'topo_clustering',
    'topo_neighbor_in_deg', 'topo_neighbor_out_deg',
    'export', 'import', 'biomass', 'respiration',
]

# Tasks: display_name - source column (None = derived)
TASKS = {
    'trophic_category': None,
    'functional_group': 'functional_group',
    'feeding_guild':    'feeding_guild',
}

MIN_CLASS_SAMPLES = 5
CV_FOLDS          = 5
N_TREES           = 200


# ============================================================
# SPECIES NAME NORMALIZATION
# ============================================================

def normalize_name(name: str) -> str:
    for ch in ' ()=/-+&;,<>X[]?.':
        name = str(name).replace(ch, '')
    return name.lower()


# ============================================================
# DATA LOADING
# ============================================================

def load_seasons(metadata_path, to_remove):
    meta = pd.read_excel(metadata_path)
    return [f"{n}.scor" for n in meta['Network name']
            if f"{n}.scor" not in to_remove]

def load_clusters(metadata_path, to_remove):
    meta = pd.read_excel(metadata_path)

    meta['file_name'] = meta['Network name'] + ".scor"
    meta = meta[~meta['file_name'].isin(to_remove)]

    return meta[['file_name', 'Cluster ID']]


def extract_topological_features(G):
    in_deg  = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    G_und   = G.to_undirected()
    btw     = nx.betweenness_centrality(G_und, normalized=True)
    clust   = nx.clustering(G_und)

    result = {}
    for node in G.nodes():
        neighbors = list(G.predecessors(node)) + list(G.successors(node))
        result[node] = {
            'topo_in_degree':        in_deg.get(node, 0),
            'topo_out_degree':       out_deg.get(node, 0),
            'topo_degree':           in_deg.get(node, 0) + out_deg.get(node, 0),
            'topo_betweenness':      btw.get(node, 0.0),
            'topo_clustering':       clust.get(node, 0.0),
            'topo_neighbor_in_deg':  np.mean([in_deg.get(n, 0)  for n in neighbors]) if neighbors else 0.0,
            'topo_neighbor_out_deg': np.mean([out_deg.get(n, 0) for n in neighbors]) if neighbors else 0.0,
        }
    return result


def collect_graph_node_features(seasons, dataset_path):
    records = []
    for season in seasons:
        try:
            foodweb = fw.read_from_SCOR(dataset_path + season)
            G = foodweb.get_graph()
        except Exception as e:
            print(f"[SKIP] {season}: {e}")
            continue

        topo         = extract_topological_features(G)
        network_name = season.replace('.scor', '')

        for node, attr in G.nodes(data=True):
            node_str = str(node).strip()
            row = {
                'file_name':    season,
                'network_name': network_name,
                'node_name':    node_str,
                'key_norm':     network_name + '_' + normalize_name(node_str),
            }
            row.update(attr)
            row.update(topo.get(node, {}))
            records.append(row)

    return pd.DataFrame(records)


def load_and_merge_llm(df_graph, llm_path):
    """
    Load LLM feature file and left-join onto graph node features via key_norm.
    Returns (merged_df, llm_feature_cols).
    """
    df_llm = pd.read_excel(llm_path)
    assert 'file_name' in df_llm.columns and 'node' in df_llm.columns

    df_llm['key_norm'] = (
        df_llm['file_name'].astype(str).str.strip()
                           .str.replace('.scor', '', regex=False)
        + '_' + df_llm['node'].apply(normalize_name)
    )

    meta_cols     = {'file_name', 'node', 'key_norm'}
    llm_feat_cols = ['activity_patterns', 'age_at_maturity_avg_years',
       'bioturbation_impact', 'body_form', 'bycatch_risk',
       'climate_sensitivity', 'competition_intensity', 'defense_mechanisms',
       'diet_breadth', 'ecosystem_role', 'fecundity',
       'feeding_guild', 'feeding_strategy', 'feeding_time_preference',
       'fishing_pressure', 'functional_group', 'generation_time_avg_years',
       'iucn_status', 'invasive_potential', 'larval_stage_type',
       'life_span_avg_years', 'locomotion_mode', 'metabolic_rate_avg',
       'metabolic_strategy', 'migration_pattern', 'osmotic_regulation',
       'parental_investment', 'predator_avoidance_strategy',
       'preferred_depth_zone', 'prey_capture_strategy',
       'reproductive_seasonality', 'reproductive_strategy',
       'salinity_tolerance', 'sensory_adaptations', 'sexual_dimorphism',
       'skin_covering', 'social_behavior', 'spawning_strategy',
       'substrate_association', 'symbiotic_associations',
       'temperature_preference', 'typical_length_avg_cm',
       'water_depth']

    df_merged = df_graph.merge(df_llm[['key_norm'] + llm_feat_cols + ['trophic_level']],
                               on='key_norm', how='left')

    n_matched = df_merged[llm_feat_cols].notna().all(axis=1).sum()
    print(f"LLM merge: {n_matched}/{len(df_merged)} nodes matched")
    return df_merged, llm_feat_cols


# ============================================================
# TARGET PREPARATION
# ============================================================

def make_trophic_category(df):
    if 'trophic_level' not in df.columns:
        raise ValueError("Column 'trophic_level' required for trophic_category target")
    tl   = pd.to_numeric(df['trophic_level'], errors='coerce')
    cats = pd.cut(tl, bins=TROPHIC_BINS, labels=TROPHIC_LABELS,
                  right=True, include_lowest=True).astype(str)
    df   = df.copy()
    df['trophic_category'] = cats
    return df[df['trophic_category'] != 'nan'].copy()


def prepare_labels(df, target_col, min_samples):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    df = df.copy()
    df['label'] = df[target_col].astype(str).str.strip()
    df['label'] = df['label'].str.split(',').str[0].str.strip()

    merge_map = {
        'omnivorous': 'omnivore', 
    }

    df['label'] = df['label'].replace(merge_map)
    df = df[df['label'].notna() & ~df['label'].isin(['nan', ''])]
    # Merge rare classes
    counts = df['label'].value_counts()
    rare   = set(counts[counts < min_samples].index)
    df['label'] = df['label'].apply(lambda x: 'other' if x in rare else x)
    return df


# ============================================================
# FEATURE MATRIX
# ============================================================

def build_feature_matrix(df, feature_cols):
    X = df[feature_cols].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = LabelEncoder().fit_transform(
                X[col].fillna('other').astype(str))
    return X.apply(pd.to_numeric, errors='coerce').fillna(0)


# ============================================================
# CLASSIFICATION
# ============================================================

def majority_accuracy(y):
    counts = pd.Series(y).value_counts()
    return float(counts.iloc[0]) / len(y)


def cross_validate(df, X, y, cv_folds):
    '''
    df["group"] = (
    df["file_name"].astype(str) + "_" +
    df["Cluster ID"].astype(str))
    '''
    #groups = df["group"].values
    groups = df['Cluster ID'].values
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y)

    n_splits = max(2, min(cv_folds, len(np.unique(groups)) // 2))
    sgkf     = StratifiedGroupKFold(n_splits=n_splits)

    scaler   = StandardScaler()

    clf = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    majority = DummyClassifier(strategy="most_frequent")

    accs = []
    baseline_accs = []

    for train_idx, test_idx in sgkf.split(X, y_enc, groups):

        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_te = scaler.transform(X.iloc[test_idx])

        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        mask = np.isin(y_te, np.unique(y_tr))
        if mask.sum() == 0:
            continue

        # Random Forest
        clf.fit(X_tr, y_tr)
        accs.append(
            accuracy_score(y_te[mask], clf.predict(X_te[mask]))
        )

        # Majority baseline
        majority.fit(X_tr, y_tr)
        baseline_accs.append(
            accuracy_score(y_te[mask], majority.predict(X_te[mask]))
        )

    return {
        'accuracy_mean': np.mean(accs) if accs else np.nan,
        'accuracy_std':  np.std(accs)  if accs else np.nan,
        'baseline_mean': np.mean(baseline_accs) if baseline_accs else np.nan,
        'baseline_std':  np.std(baseline_accs)  if baseline_accs else np.nan,
        'n_folds':       len(accs),
    }


# ============================================================
# SINGLE TASK
# ============================================================

def run_task(task_name, target_col, df, llm_feat_cols, model_name, min_samples, cv_folds):
    print(f"--- Task: {task_name} ---")

    # Build target
    try:
        if task_name == 'trophic_category':
            df = make_trophic_category(df)
        df_task = prepare_labels(df, target_col, min_samples)
    except ValueError as e:
        print(f"[SKIP] {e}")
        return {}

    y = df_task['label'].values
    print(f"Nodes: {len(df_task)}  |  Classes: {sorted(np.unique(y).tolist())}")
    print(f"Distribution: {dict(pd.Series(y).value_counts())}")

    results = {}

    # Majority baseline (from CV)
    avail_v1 = [c for c in V1_FEATURES if c in df_task.columns]
    X = build_feature_matrix(df_task, avail_v1)
    r = cross_validate(df_task, X, y, cv_folds)

    results['majority'] = {
        'accuracy_mean': r['baseline_mean'],
        'accuracy_std':  r['baseline_std'],
        'n_folds':       r['n_folds']
    }
    print(f"    majority  → {r['baseline_mean']:.3f} ± {r['baseline_std']:.3f}")

    # v1: graph topology + flows
    avail_v1 = [c for c in V1_FEATURES if c in df_task.columns]
    if avail_v1:
        X = build_feature_matrix(df_task, avail_v1)
        r = cross_validate(df_task, X, y, cv_folds)
        results['v1'] = r
        print(f"v1 - {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}  ({len(avail_v1)} features)")
    else:
        print(f"[SKIP v1] None of V1_FEATURES found in data")

    # f1: LLM taxonomic features (no target cols, no v1 cols)
    f1_cols = [c for c in llm_feat_cols
               if c not in TARGET_COLS and c not in V1_FEATURES and c != 'label']
    if f1_cols:
        X = build_feature_matrix(df_task, f1_cols)
        r = cross_validate(df_task, X, y, cv_folds)
        results[f'f1_{model_name}'] = r
        print(f"f1_{model_name} - {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}  ({len(f1_cols)} features)")
    else:
        print(f"[SKIP f1] No LLM features after filtering")

    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_heatmap(all_results, output_path):

    COL_RENAME = {
        'v1':        'Prediction without\n LLM-generated features',
        'f1_gemini': 'Prediction with \n LLM-generated features',
        'majority':  'Baseline prediction: \nmajority classifier',
    }
    TASK_RENAME = {
        'trophic_category': 'Trophic-level\ncategory',
        'functional_group': 'Functional\ngroup',
        'feeding_guild':    'Feeding\nguild',
    }

    col_order = ['majority', 'v1', 'f1_gemini']
    for res in all_results.values():
        for v in res:
            if v not in col_order:
                col_order.append(v)

    task_names = list(all_results.keys())
    data  = pd.DataFrame(index=task_names, columns=col_order, dtype=float)
    annot = pd.DataFrame(index=task_names, columns=col_order, dtype=str)

    for task, res in all_results.items():
        for ver in col_order:
            if ver in res:
                acc = res[ver]['accuracy_mean']
                std = res[ver]['accuracy_std']
                data.loc[task, ver]  = acc
                annot.loc[task, ver] = f"{acc*100:.1f}" if std == 0 else f"{acc*100:.1f}\n±{std*100:.1f}"
            else:
                data.loc[task, ver]  = np.nan
                annot.loc[task, ver] = '—'

    col_labels  = [COL_RENAME.get(c, c)  for c in col_order]
    task_labels = [TASK_RENAME.get(t, t) for t in task_names]

    fig, ax = plt.subplots(figsize=(max(6, len(col_order) * 6),
                                    max(3, len(task_names) * 1.6)))
    sns.heatmap(
    data.astype(float),
    cmap='Blues', vmin=0, vmax=1,
    linewidths=0.6, linecolor='white',
    cbar=False,
    ax=ax,)

    for i, task in enumerate(task_names):
        for j, ver in enumerate(col_order):

            value = data.loc[task, ver]
            text = annot.loc[task, ver]

            if pd.isna(value):
                ax.text(j + 0.5, i + 0.5, '—',
                        ha='center', va='center', fontsize=16, color='black')
                continue

            color = 'white' if value > 0.5 else 'black'

            parts = text.split('\n')
            mean = parts[0]
            sd = parts[1] if len(parts) > 1 else None

            ax.text(j + 0.5, i + 0.42, mean,ha='center', va='center',fontsize=30,color=color,fontweight = 'bold')

            if sd:
                ax.text(j + 0.5, i + 0.68, sd,
                    ha='center', va='center',
                    fontsize=20,color=color)


    for y in range(1, len(task_names)):
        ax.hlines(y, *ax.get_xlim(), colors='white', linewidth=8)

    ax.set_xlabel('', fontsize=14, labelpad=10)
    ax.set_ylabel('',        fontsize=14, labelpad=10)
    ax.set_yticklabels(task_labels, rotation=0, fontsize=28)
    ax.set_xticklabels(col_labels,  rotation=0, fontsize=28)

    plt.tight_layout()
    out = os.path.join(FIG_OUTPUT, 'node_classification_heatmap.pdf')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {out}")


# ============================================================
# MAIN
# ============================================================

def run():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("[1] Loading graph node features...")
    seasons  = load_seasons(METADATA_PATH, TO_REMOVE)
    df_clusters = load_clusters(METADATA_PATH, TO_REMOVE)
    df_graph_1 = collect_graph_node_features(seasons, DATASET_PATH)
    df_graph_2 = df_graph_1[df_graph_1['IsAlive']==True]
    df_graph = df_graph_2.merge(df_clusters, how='left', on='file_name')
    print(f"Total living nodes: {len(df_graph)}")

    all_results = {}

    for model_name, llm_path in LLM_MODELS.items():
        print(f"[2] LLM model: {model_name}")
        if not os.path.exists(llm_path):
            print(f"[SKIP] File not found: {llm_path}")
            continue

        df_combined, llm_feat_cols = load_and_merge_llm(df_graph.copy(), llm_path)

        # Keep only nodes with LLM features for a fair v1 vs f1 comparison
        has_llm = df_combined[llm_feat_cols].notna().any(axis=1)
        df_llm_nodes = df_combined[has_llm].copy()
        print(f"Nodes with LLM features: {len(df_llm_nodes)}")

        print("[3] Running tasks...")
        for task_name, target_col in TASKS.items():
            res = run_task(
                task_name    = task_name,
                target_col   = target_col or task_name,
                df           = df_llm_nodes.copy(),
                llm_feat_cols= llm_feat_cols,
                model_name   = model_name,
                min_samples  = MIN_CLASS_SAMPLES,
                cv_folds     = CV_FOLDS,
            )
            if res:
                all_results[task_name] = res

    if not all_results:
        print("No results to display.")
        return

    print("[4] Saving outputs...")
    plot_heatmap(all_results, OUTPUT_PATH)

    rows = [{'task': t, 'version': v, **m}
            for t, res in all_results.items()
            for v, m in res.items()]
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_PATH, 'node_classification_results.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")

    print(f"\n{'='*55}\n SUMMARY\n{'='*55}")
    print(df_out.to_string(index=False))

    for task, res in all_results.items():
        if 'v1' not in res:
            continue
        v1_acc = res['v1']['accuracy_mean']
        for ver, m in res.items():
            if not ver.startswith('f1'):
                continue
            delta = m['accuracy_mean'] - v1_acc
            tag   = "IMPROVEMENT" if delta > 0 else "NO IMPROVEMENT"
            print(f"  {task:22s} | {ver} vs v1: Δacc={delta:+.3f}  {tag}")

if __name__ == '__main__':
    run()