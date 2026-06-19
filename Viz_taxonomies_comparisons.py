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

def ci_heatmap(df_stats, option):
    heatmap_data = df_stats.pivot(
        index="taxonomy_level",
        columns="model",
        values= option
    )

    mean_col = f"{option}"
    ci_low_col = f"{option}_ci_low"
    ci_high_col = f"{option}_ci_high"

    def format_anno(row):
        return (
            f"{row[mean_col]*100:.0f}\n"
            f"[{row[ci_low_col]*100:.0f}, {row[ci_high_col]*100:.0f}]"
        )

    anno_data = df_stats.copy()
    anno_data['anno_text'] = anno_data.apply(format_anno, axis=1)
    heatmap_anno = anno_data.pivot(
        index="taxonomy_level",
        columns="model",
        values="anno_text"
    )

    rename_dict = {'consensus': 'LLM optimization (1)'}
    heatmap_data = heatmap_data.rename(columns=rename_dict)
    heatmap_anno = heatmap_anno.rename(columns=rename_dict)

    taxonomy_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    heatmap_data = heatmap_data.reindex(taxonomy_order)
    heatmap_anno = heatmap_anno.reindex(taxonomy_order)

    new_column_order = [
    "chatgpt", "claude", "gemini", "qwen",       
    "arbitrated_gpt", "arbitrated_claude", "arbitrated_gemini", "arbitrated_qwen",
    "LLM optimization (1)",
    "worms_llm_chatgpt", "worms_llm_claude", "worms_llm_gemini", "worms_llm_qwen",
    "best_chatgpt", "best_claude", "best_gemini", 
    "best_qwen", "best_all"
    ]

    rename_dict = {
        # --- Basic LLM ---
        "claude": "Claude",
        "gemini": "Gemini",
        "chatgpt": "ChatGPT",
        "qwen": "Qwen",

        # --- Self-evaluated soft voting ---
        "arbitrated_gpt": " ChatGPT",
        "arbitrated_claude": " Claude",
        "arbitrated_qwen": " Qwen",
        "arbitrated_gemini": " Gemini",

        # --- Black box ---
        "LLM optimization (1)": "All models",

        # --- WoRMS + LLM ---
        "worms_llm_claude": "WoRMS + Claude",
        "worms_llm_gemini": "WoRMS + Gemini",
        "worms_llm_chatgpt": "WoRMS + ChatGPT",
        "worms_llm_qwen": "WoRMS + Qwen",

        # --- Cross-model evaluated soft voting ---
        "best_claude": "  Claude",
        "best_gemini": "  Gemini",
        "best_chatgpt": "  ChatGPT",
        "best_qwen": "  Qwen",
        "best_all": "  All models"
    }

    heatmap_data = heatmap_data.rename(columns=rename_dict)
    heatmap_anno = heatmap_anno.rename(columns=rename_dict)

    new_column_order_readable = [rename_dict.get(c, c) for c in new_column_order]

    available_cols = [c for c in new_column_order_readable if c in heatmap_data.columns]

    heatmap_data = heatmap_data[available_cols]
    heatmap_anno = heatmap_anno[available_cols]

    column_groups = [
    ("WoRMS + LLM", ["WoRMS + ChatGPT", "WoRMS + Claude", "WoRMS + Gemini", "WoRMS + Qwen"]), 
    ("Direct prompting", [ "ChatGPT", "Claude", "Gemini", "Qwen"]),
    ("Ensemble:\n self-evaluated\n soft voting", ["All models"]),
    ("Ensemble: cross-model evaluated soft voting", [
        "  ChatGPT", 
        "  Claude",
        "  Gemini",
        "  Qwen",
        "  All models"
    ]),
    ("Ensemble: LLM as a judge", [" ChatGPT"," Claude", " Gemini", 
                             " Qwen"]), 
    ]

    SEPARATOR = "   "  
    ordered_cols = []
    for i, (group_label, cols) in enumerate(column_groups):
        existing = [c for c in cols if c in heatmap_data.columns]
        if not existing:
            continue
        if ordered_cols: 
            sep = SEPARATOR * (i + 1)  
            heatmap_data[sep] = np.nan
            heatmap_anno[sep] = ""
            ordered_cols.append(sep)
        ordered_cols.extend(existing)

    heatmap_data = heatmap_data[ordered_cols]
    heatmap_anno = heatmap_anno[ordered_cols]

    # --- VISUALISATION ---
    fig, ax = plt.subplots(figsize=(30, 15))
    '''
    sns.heatmap(
        heatmap_data,
        annot=heatmap_anno,
        fmt="",
        vmin=0,
        vmax=1,
        cmap="Blues",
        cbar=False,
        cbar_kws={"label": "Mean Agreement"},
        annot_kws={"size": 14},
        linewidths=0.5,
        ax=ax
    )
    '''
    sns.heatmap(
            heatmap_data,
            annot=False,         
            fmt="",
            vmin=0,
            vmax=1,
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            ax=ax
        )

    for i, row_label in enumerate(heatmap_data.index):
        for j, col_label in enumerate(heatmap_data.columns):
            anno = heatmap_anno.loc[row_label, col_label]
            if not anno or str(anno).strip() == "":
                continue
            parts = str(anno).split("\n")
            main_val = parts[0] if len(parts) > 0 else ""
            ci_val   = parts[1] if len(parts) > 1 else ""

            cell_val = heatmap_data.loc[row_label, col_label]
            text_color = "white" if (pd.notna(cell_val) and cell_val > 0.7) else "black"

            # Main value — large and bold
            ax.text(j + 0.5, i + 0.38, main_val,
                    ha="center", va="center",
                    fontsize=16, color=text_color, fontweight="bold")
            # CI interval — smaller, below main value
            ax.text(j + 0.5, i + 0.66, ci_val,
                    ha="center", va="center",
                    fontsize=12, color=text_color)
  
    
    ax.set_xlabel("Prompting strategy", fontsize=16)
    ax.set_ylabel("Taxonomic level", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels([label.get_text().capitalize() for label in ax.get_yticklabels()])

    for tick, col in zip(ax.get_xticklabels(), heatmap_data.columns):
        if col.strip() == "":
            tick.set_visible(False)

    # --- Group names ---
    n_cols = len(heatmap_data.columns)
    col_positions = {col: i + 0.5 for i, col in enumerate(heatmap_data.columns)}

    GROUP_LABEL_Y   = 1.02   # text height
    GROUP_BRACKET_Y = 1.01   # line height

    for group_label, cols in column_groups:
        existing = [c for c in cols if c in col_positions]
        if not existing:
            continue

        x_start_data = col_positions[existing[0]] - 0.5
        x_end_data   = col_positions[existing[-1]] + 0.5
        x_mid_frac   = (x_start_data + x_end_data) / 2 / n_cols
        x_start_frac = x_start_data / n_cols
        x_end_frac   = x_end_data   / n_cols

        ax.text(
            x_mid_frac, GROUP_LABEL_Y,
            group_label,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=16, fontweight="bold",
            clip_on=False
        )

        ax.annotate(
            "",
            xy=(x_end_frac, GROUP_BRACKET_Y),
            xycoords="axes fraction",
            xytext=(x_start_frac, GROUP_BRACKET_Y),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-", color="gray", lw=1.5),
            annotation_clip=False
        )

    plt.subplots_adjust(top=0.82, bottom=0.25)
    path = f"Figures/{version}/"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'Figures/{version}/llm_{option}_heatmap_ci_{version}_nanstrategy_{nanstrategy}.pdf', format='pdf', bbox_inches='tight')
    #plt.show()

    print("CI heatmap saved")

def stacked_heatmap(df_stats, version, nanstrategy):
    if nanstrategy != '2':
        return
    
    categories = ['accuracy', 'hallucination_rate', 'missed_rate', 'incorrect_rate']
    titles = ['Correct / both empty', 'Hallucination', 'Missed', 'Incorrect']
    cmaps  = ['Blues', 'Reds', 'Oranges', 'Purples']
    
    taxonomy_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    
    fig, axes = plt.subplots(1, 4, figsize=(48, 15))
    
    for ax, cat, title, cmap in zip(axes, categories, titles, cmaps):
        data = df_stats.pivot(index="taxonomy_level", columns="model", values=cat)
        data = data.reindex(taxonomy_order)
        
        sns.heatmap(
            data, annot=True, fmt=".0%",
            vmin=0, vmax=1,
            cmap=cmap, cbar=False,
            linewidths=0.5, ax=ax,
            annot_kws={"size": 11}
        )
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Taxonomic level" if ax == axes[0] else "")
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Error breakdown (nanstrategy 1)", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    path = f"Figures/{version}/"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/llm_error_breakdown_{version}_nanstrategy_{nanstrategy}.pdf', 
                format='pdf', bbox_inches='tight')
    print("Error breakdown saved")

def double_heatmap(df_stats, version, nanstrategy):
    options_pair = [
        ('accuracy',          'Accuracy',        'Blues'),
        ('hallucination_rate','Overspecification','Reds'),
    ]

    taxonomy_order = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # ta sama logika separatorów i rename co w ci_heatmap
    # ale robisz fig, (ax1, ax2) = plt.subplots(2, 1, ...)
    fig, axes = plt.subplots(2, 1, figsize=(30, 26))  # 2x wyższy

    for ax, (option, title, cmap) in zip(axes, options_pair):

        # --- pivot i rename (skopiuj logikę z ci_heatmap) ---
        heatmap_data = df_stats.pivot(index="taxonomy_level", columns="model", values=option)
        # ... rename_dict, reindex, separatory ... (identyczne jak w ci_heatmap)

        # --- adnotacje z CI ---
        ci_low_col  = f"{option}_ci_low"
        ci_high_col = f"{option}_ci_high"
        anno_data = df_stats.copy()
        anno_data['anno_text'] = anno_data.apply(
            lambda row: f"{row[option]*100:.0f}\n[{row[ci_low_col]*100:.0f}, {row[ci_high_col]*100:.0f}]", axis=1
        )
        heatmap_anno = anno_data.pivot(index="taxonomy_level", columns="model", values="anno_text")
        # ... ten sam rename i reindex ...

        sns.heatmap(heatmap_data, annot=False, vmin=0, vmax=1,
                    cmap=cmap, cbar=False, linewidths=0.5, ax=ax)

        # --- tekst w komórkach (skopiuj pętlę z ci_heatmap) ---
        for i, row_label in enumerate(heatmap_data.index):
            for j, col_label in enumerate(heatmap_data.columns):
                # ... identyczna logika main_val / ci_val ...
                pass

        # --- tytuł subplotu zamiast osobnego pliku ---
        ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("Taxonomic level", fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        # --- sekcje tematyczne tylko na górnym subplocie ---
        if option == 'accuracy':
            # ... cała logika GROUP_LABEL_Y / annotate z ci_heatmap ...
            pass

    plt.subplots_adjust(hspace=0.35, top=0.88, bottom=0.12)
    path = f"Figures/{version}/"
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/llm_accuracy_overspecification_{version}_nanstrategy_{nanstrategy}.pdf',
                format='pdf', bbox_inches='tight')
    print("Double heatmap saved")



for option in options:
    ci_heatmap(df_stats, option)

stacked_heatmap(df_stats, version, nanstrategy)
double_heatmap(df_stats, version, nanstrategy)