import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# ============================================================================
# HEATMAP with confidence intervals
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Process food web taxonomic data.')
    parser.add_argument('--version', type=str, required=True, help='Version string, e.g. v1')
    parser.add_argument('--nanstrategy', type=str, required=True, help='1 for replacing all nan with empty strings and comparing all, 2 for remove observations missing in baseline, 3 for removing observations with nan for either llm or baseline')
    return parser.parse_args()

args = parse_args()
version = args.version
nanstrategy = args.nanstrategy


df_stats = pd.read_csv(f'Comparisons/llm_scores_{version}_nanstrategy_{nanstrategy}.csv', sep=';')
options = ['agreement', 'accuracy', 'hallucination_rate', 'missed_rate', 'incorrect_rate']

# ============================================================================
# SHARED HELPERS
# ============================================================================

taxonomy_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

rename_dict_full = {
    "claude": "Claude",
    "gemini": "Gemini",
    "chatgpt": "ChatGPT",
    "qwen": "Qwen",
    "arbitrated_gpt": " ChatGPT",
    "arbitrated_claude": " Claude",
    "arbitrated_qwen": " Qwen",
    "arbitrated_gemini": " Gemini",
    "LLM optimization (1)": "All models",
    "worms_llm_claude": "WoRMS + Claude",
    "worms_llm_gemini": "WoRMS + Gemini",
    "worms_llm_chatgpt": "WoRMS + ChatGPT",
    "worms_llm_qwen": "WoRMS + Qwen",
    "best_claude": "  Claude",
    "best_gemini": "  Gemini",
    "best_chatgpt": "  ChatGPT",
    "best_qwen": "  Qwen",
    "best_all": "  All models"
}

column_groups = [
    ("WoRMS + LLM", ["WoRMS + ChatGPT", "WoRMS + Claude", "WoRMS + Gemini", "WoRMS + Qwen"]),
    ("Direct prompting", ["ChatGPT", "Claude", "Gemini", "Qwen"]),
    ("Ensemble:\n self-evaluated\n soft voting", ["All models"]),
    ("Ensemble: cross-model evaluated soft voting", ["  ChatGPT", "  Claude", "  Gemini", "  Qwen", "  All models"]),
    ("Ensemble: LLM as a judge", [" ChatGPT", " Claude", " Gemini", " Qwen"]),
]


def build_heatmap_data(df_stats, option):
    heatmap_data = df_stats.pivot(index="taxonomy_level", columns="model", values=option)
    heatmap_data = heatmap_data.rename(columns={'consensus': 'LLM optimization (1)'})
    heatmap_data = heatmap_data.reindex(taxonomy_order)
    heatmap_data = heatmap_data.rename(columns=rename_dict_full)

    new_column_order = [
        "chatgpt", "claude", "gemini", "qwen",
        "arbitrated_gpt", "arbitrated_claude", "arbitrated_gemini", "arbitrated_qwen",
        "LLM optimization (1)",
        "worms_llm_chatgpt", "worms_llm_claude", "worms_llm_gemini", "worms_llm_qwen",
        "best_chatgpt", "best_claude", "best_gemini", "best_qwen", "best_all"
    ]
    new_column_order_readable = [rename_dict_full.get(c, c) for c in new_column_order]
    available_cols = [c for c in new_column_order_readable if c in heatmap_data.columns]
    heatmap_data = heatmap_data[available_cols]

    SEPARATOR = "   "
    ordered_cols = []
    for i, (group_label, cols) in enumerate(column_groups):
        existing = [c for c in cols if c in heatmap_data.columns]
        if not existing:
            continue
        if ordered_cols:
            sep = SEPARATOR * (i + 1)
            heatmap_data[sep] = np.nan
            ordered_cols.append(sep)
        ordered_cols.extend(existing)

    return heatmap_data[ordered_cols], ordered_cols


def draw_group_brackets(ax, ordered_cols, labels, GROUP_LABEL_Y=1.02, GROUP_BRACKET_Y=1.01, fontsize=16):
    n_cols = len(ordered_cols)
    col_positions = {col: i + 0.5 for i, col in enumerate(ordered_cols)}
    for group_label, cols in column_groups:
        existing = [c for c in cols if c in col_positions]
        if not existing:
            continue
        x_start_data = col_positions[existing[0]] - 0.5
        x_end_data   = col_positions[existing[-1]] + 0.5
        x_mid_frac   = (x_start_data + x_end_data) / 2 / n_cols
        x_start_frac = x_start_data / n_cols
        x_end_frac   = x_end_data   / n_cols
        if labels:
            ax.text(x_mid_frac, GROUP_LABEL_Y, group_label,
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=fontsize, fontweight="bold", clip_on=False)
            ax.annotate("",
                        xy=(x_end_frac, GROUP_BRACKET_Y),
                        xycoords="axes fraction",
                        xytext=(x_start_frac, GROUP_BRACKET_Y),
                        textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-", color="gray", lw=1.5),
                        annotation_clip=False)


# ============================================================================
# 1. CI HEATMAP — full with annotations (agreement, accuracy)
# ============================================================================

def ci_heatmap(df_stats, option):
    heatmap_data, ordered_cols = build_heatmap_data(df_stats, option)

    mean_col    = option
    ci_low_col  = f"{option}_ci_low"
    ci_high_col = f"{option}_ci_high"

    def format_anno(row):
        return (
            f"{row[mean_col]*100:.0f}\n"
            f"[{row[ci_low_col]*100:.0f}, {row[ci_high_col]*100:.0f}]"
        )

    anno_data = df_stats.copy()
    anno_data['anno_text'] = anno_data.apply(format_anno, axis=1)
    heatmap_anno = anno_data.pivot(index="taxonomy_level", columns="model", values="anno_text")
    heatmap_anno = heatmap_anno.rename(columns={'consensus': 'LLM optimization (1)'})
    heatmap_anno = heatmap_anno.reindex(taxonomy_order)
    heatmap_anno = heatmap_anno.rename(columns=rename_dict_full)

    SEPARATOR = "   "
    anno_ordered = []
    for i, (_, cols) in enumerate(column_groups):
        existing = [c for c in cols if c in heatmap_anno.columns]
        if not existing:
            continue
        if anno_ordered:
            sep = SEPARATOR * (i + 1)
            heatmap_anno[sep] = ""
            anno_ordered.append(sep)
        anno_ordered.extend(existing)
    heatmap_anno = heatmap_anno[[c for c in anno_ordered if c in heatmap_anno.columns]]

    fig, ax = plt.subplots(figsize=(30, 15))
    sns.heatmap(heatmap_data, annot=False, fmt="", vmin=0, vmax=1,
                cmap="Blues", cbar=False, linewidths=0.5, ax=ax)

    for i, row_label in enumerate(heatmap_data.index):
        for j, col_label in enumerate(heatmap_data.columns):
            anno = heatmap_anno.loc[row_label, col_label] if col_label in heatmap_anno.columns else ""
            if not anno or str(anno).strip() == "":
                continue
            parts = str(anno).split("\n")
            main_val = parts[0] if len(parts) > 0 else ""
            ci_val   = parts[1] if len(parts) > 1 else ""
            cell_val = heatmap_data.loc[row_label, col_label]
            text_color = "white" if (pd.notna(cell_val) and cell_val > 0.7) else "black"
            ax.text(j + 0.5, i + 0.38, main_val,
                    ha="center", va="center", fontsize=16,
                    color=text_color, fontweight="bold")
            ax.text(j + 0.5, i + 0.66, ci_val,
                    ha="center", va="center", fontsize=12, color=text_color)

    ax.set_xlabel("Prompting strategy", fontsize=16)
    ax.set_ylabel("Taxonomic level", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels([label.get_text().capitalize() for label in ax.get_yticklabels()])
    for tick, col in zip(ax.get_xticklabels(), heatmap_data.columns):
        if col.strip() == "":
            tick.set_visible(False)

    draw_group_brackets(ax, ordered_cols, True)
    plt.subplots_adjust(top=0.82, bottom=0.25)
    path = f"Figures/{version}/"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/llm_{option}_heatmap_ci_{version}_nanstrategy_{nanstrategy}.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()
    print(f"CI heatmap saved: {option}")


# ============================================================================
# 2. SMALL HEATMAP — color pattern only, no labels or annotations
# ============================================================================

def small_ci_heatmap(df_stats, option):
    heatmap_data, ordered_cols = build_heatmap_data(df_stats, option)

    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(heatmap_data, annot=False, fmt="", vmin=0, vmax=1,
                cmap="Blues", cbar=False, linewidths=0.5, ax=ax)

    # main value only — no CI, larger font to fit small figure
    for i, row_label in enumerate(heatmap_data.index):
        for j, col_label in enumerate(heatmap_data.columns):
            cell_val = heatmap_data.loc[row_label, col_label]
            if pd.isna(cell_val):
                continue
            # get raw value from df_stats for this model/tax combo
            col_clean = col_label.strip()
            # find matching row in df_stats by reverse-looking up model name
            try:
                val = cell_val
                text_color = "white" if val > 0.7 else "#888888"
                ax.text(j + 0.5, i + 0.5, f"{val*100:.0f}",
                        ha="center", va="center",
                        fontsize=18, fontweight="bold", color=text_color)
            except Exception:
                continue



    # group brackets — slightly lower than full heatmap to fit compact figure
    draw_group_brackets(ax, ordered_cols, False,GROUP_LABEL_Y=1.04, GROUP_BRACKET_Y=1.02, fontsize=10)

    # no axis labels, no tick labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(left=False, bottom=False)

    plt.subplots_adjust(top=0.78)
    path = f"Figures/{version}/"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/small_{option}_heatmap_ci_{version}_nanstrategy_{nanstrategy}.pdf',
                format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Small heatmap saved: {option}")


# ============================================================================
# RUN
# ============================================================================

small_options  = ['hallucination_rate', 'missed_rate', 'incorrect_rate']
full_options   = ['agreement', 'accuracy']

for option in full_options:
    ci_heatmap(df_stats, option)

for option in small_options:
    small_ci_heatmap(df_stats, option)
