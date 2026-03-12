import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import argparse


# === CONFIGURATION ===

def parse_args():
    parser = argparse.ArgumentParser(description='Triangular plots parser')
    parser.add_argument('--versions', nargs='+', required=True,
                    help=f'Names of versions')
    parser.add_argument('--nanstrategy', default=2,
                        help='Nan cutting option')

    return parser.parse_args()

args = parse_args()
VERSIONS = args.versions
nanstrategy = args.nanstrategy


INPUT_FILE = f'Inter_Model_Divergence/inter_agreement_all_inter_model_nanstrategy_{nanstrategy}.csv'
OUTPUT_DIR  = 'Figures'
MODELS      = ['chatgpt', 'claude', 'gemini', 'qwen']
TAX_LEVELS  = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
norm = mcolors.Normalize(vmin=0, vmax=1)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# === MATRIX AND DIAGRAM ===
def correlation_matrix(level_to_compare, diagonal_values=None):
    #agreement_cols = [f'{level_to_compare}_claude',f'{level_to_compare}_gemini',f'{level_to_compare}_chatgpt',f'{level_to_compare}_qwen']
    df_inter = pd.read_csv(INPUT_FILE, sep=';')
    #df = ( df_inter.groupby(["model1", "model2", "column"])["accuracy"] .mean() .reset_index())

    df_inter["model_a"] = np.minimum(df_inter["model1"], df_inter["model2"])
    df_inter["model_b"] = np.maximum(df_inter["model1"], df_inter["model2"])
    # średnia accuracy niezależnie od kolejności 
    models_result = ( df_inter.groupby(["model_a", "model_b", "column"], as_index=False)["accuracy"] .mean() .rename(columns={"accuracy": "accuracy_mean"}))
    models_result = models_result[models_result['column'].isin(TAX_LEVELS)].reset_index(drop=True)

    agreement_matrix = pd.DataFrame(index=MODELS, columns=MODELS)

    for i in MODELS:
        for j in MODELS:
            subset = models_result[
                (models_result['model_a'] == i) &
                (models_result['model_b'] == j) &
                (models_result['column'] == level_to_compare)]
            if not subset.empty:
                agreement_matrix.loc[i, j] = subset['accuracy_mean'].iloc[0]*100
                
    agreement_matrix = agreement_matrix.T
    
    nice_labels = []
    for col in agreement_matrix.columns:
        name = col.replace(f'{level_to_compare}_', '')
        if name == "chatgpt":
            name = "ChatGPT"
        else:
            name = name.capitalize()
            
        nice_labels.append(name)

    mask = np.triu(np.ones(agreement_matrix.shape, dtype=bool))
    plot_matrix = agreement_matrix.copy().astype(float)
    plot_matrix[mask] = np.nan

    plot_matrix = agreement_matrix.copy().astype(float)
    mask = np.triu(np.ones(plot_matrix.shape, dtype=bool))
    plot_matrix.values[mask] = np.nan

    cmap = plt.cm.Blues.copy()
    cmap.set_bad('white')

    plt.figure(figsize=(8,8))
    im = plt.imshow(plot_matrix.values, vmin=0, vmax=100, cmap=cmap)
    #plt.suptitle(f"{level_to_compare.capitalize()}", fontsize=40, y=0.98)

    # lower triangle
    for i in range(len(agreement_matrix.columns)):
        for j in range(len(agreement_matrix.columns)):
            if i > j:
                val = agreement_matrix.iloc[i, j]
                color = "white" if val > 50 else "black"
                plt.text(j, i, f"{val:.0f}", ha="center", va="center", color=color, fontsize=35, fontweight = 'bold')

    # diagonal
    ax = plt.gca()
    if diagonal_values:
        for i, col in enumerate(agreement_matrix.columns):
            model_name = col.replace(f'{level_to_compare}_', '')
            diagonal_values = agreement_dict.get(level_to_compare, {})
            val = diagonal_values.get(model_name, None)

            if val is not None:
                color = cmap(norm(val))

                # 🔲 the entire cell square
                rect = patches.Rectangle(
                    (i - 0.5, i - 0.5),   # lower left corner
                    1,                   # width
                    1,                   # hight
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=2
                )
                ax.add_patch(rect)

                # text on top
                text_color = "white" if val > 0.4 else "black"

                plt.text(
                    i, i,
                    f"{val*100:.0f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=35,
                    zorder=3,
                    fontweight = 'bold'
                )
    #plt.colorbar(label="Agreement")
    plt.xticks(range(len(agreement_matrix.columns)), nice_labels, rotation=45, ha="right", fontsize=30)
    plt.yticks(range(len(agreement_matrix.columns)), nice_labels, fontsize=30)
    plt.title(f"{level_to_compare.capitalize()}", fontsize=50)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{level_to_compare}_accuracy_matrix_nan_strategy_{nanstrategy}.pdf')

# === RUN ===
df_intra = pd.read_csv(f'Inter_Model_Divergence/intra_agreement_all_inter_model_nanstrategy_{nanstrategy}.csv', sep=';')

df_intra = (
    df_intra.groupby(["column", "model2"], as_index=False)
      .agg({"accuracy": "mean"})
)

agreement_dict = (
    df_intra
        .set_index(["column", "model2"])["accuracy"]
        .unstack("model2")
        .to_dict(orient="index")
)

for level in TAX_LEVELS:
    print(f"Plotting: {level}")
    correlation_matrix(level, diagonal_values=True)