import pandas as pd


df_level = pd.read_excel('LLM features/Processed/all_models/tax_level_1606_gen/ALL_MODELS_COMBINED.xlsx')
df_taxonomy = pd.read_excel('LLM features/Processed/all_models/gen1_0603/ALL_MODELS_COMBINED.xlsx')
df_taxonomy = df_taxonomy[['file_name', 'node']+ [x for x in df_taxonomy.columns if 'chatgpt' in x 
            and 'data' not in x and 'confidence' not in x and 'representative' not in x 
            and 'latin' not in x and 'included' not in x and 'Alive' not in x]]

df_all = df_level.merge(df_taxonomy, how='outer', on = ['file_name', 'node'])

TAXONOMY_HIERARCHY = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

TAXON_COLUMNS = {
    'kingdom': 'kingdom_chatgpt',
    'phylum':  'phylum_chatgpt',
    'class':   'class_chatgpt',
    'order':   'order_chatgpt',
    'family':  'family_chatgpt',
    'genus':   'genus_chatgpt',
    'species': 'species_chatgpt',
}


def trim_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zeruje wszystkie rangi bardziej szczegółowe niż poziom
    wskazany w 'Taxonomic Level_chatgpt'.
    """
    df = df.copy()

    for idx, row in df.iterrows():
        raw_level = row['Taxonomic Level_chatgpt']

        if pd.isna(raw_level) or str(raw_level).strip() == '':
            continue

        normalized = str(raw_level).strip().lower()

        if normalized not in TAXONOMY_HIERARCHY:
            print(f"[UWAGA] Nieznana ranga '{raw_level}' w wierszu {idx} ({row.get('node', '')})")
            continue

        target_idx = TAXONOMY_HIERARCHY.index(normalized)

        for rank in TAXONOMY_HIERARCHY[target_idx + 1:]:
            col = TAXON_COLUMNS[rank]
            if col in df.columns:
                df.at[idx, col] = None

    return df


df_trimmed = trim_taxonomy(df_all)
df_trimmed.to_csv('Separate_session_prompting/Taxonomies_from_separate_session.csv')

