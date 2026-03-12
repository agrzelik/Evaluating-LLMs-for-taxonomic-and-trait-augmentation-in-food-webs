import requests
import argparse
import pandas as pd
import time
import foodwebviz as fw
import os

MODELS = ['claude', 'gemini', 'chatgpt', 'qwen']

def parse_args():
    parser = argparse.ArgumentParser(description='Process food web taxonomic data.')
    parser.add_argument('--version', type=str, required=True, help='Version string, e.g. v1')
    return parser.parse_args()

def get_worms_record(species_name):
    """
    Fetches single best taxonomic record for a given species name.
    Get documentation for WORMS REST here: https://www.marinespecies.org/rest/
    """
    base_url = "https://www.marinespecies.org/rest/AphiaRecordsByName"
    url = f"{base_url}/{species_name}?like=true&marine_only=false"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                record = data[0]
                return {
                    "QueryName":      species_name,
                    "ScientificName": record.get("scientificname"),
                    "AphiaID":        record.get("AphiaID"),
                    "Rank":           record.get("rank"),
                    "Status":         record.get("status"),
                    "Kingdom":        record.get("kingdom"),
                    "Phylum":         record.get("phylum"),
                    "Class":          record.get("class"),
                    "Order":          record.get("order"),
                    "Family":         record.get("family"),
                    "Genus":          record.get("genus"),
                    "Species":        record.get("species")
                }
        return {"QueryName": species_name, "Status": "Not Found"}
    except Exception as e:
        return {"QueryName": species_name, "Status": f"Error: {e}"}


def process_foodweb_data(file_path):
    try:
        if not hasattr(fw, "read_from_SCOR"):
            raise AttributeError("read_from_SCOR not found in local foodwebviz.py")
        foodweb = fw.read_from_SCOR(file_path)
        G = foodweb.get_graph()
        attributes_dict = {node: attr_dict for node, attr_dict in G.nodes(data=True)}
        df = pd.DataFrame.from_dict(attributes_dict, orient="index")
        df.reset_index(inplace=True)
        df.rename(columns={"index": "node"}, inplace=True)
        df["file_name"] = os.path.basename(file_path)
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        raise e


# --- Load data ---
args = parse_args()
version = args.version

base_path = 'dataset_20260126_ecobase/processed'
df_all = pd.read_excel(f'LLM features/Processed/all_models/{version}/ALL_MODELS_COMBINED.xlsx')
scor_files = list(df_all['file_name'].unique())

df_db = pd.DataFrame()
for scor_file in scor_files:
    file_path = os.path.join(base_path, scor_file + '.scor')
    df = process_foodweb_data(file_path)[['node', 'file_name', 'IsAlive']]
    df_db = pd.concat([df_db, df], ignore_index=True)

df_db = df_db[df_db['IsAlive'] == True].reset_index(drop=True)

# --- Query WoRMS for each model ---
taxonomy_cols = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'ScientificName']
percents_df = pd.DataFrame(columns=['model', 'percent'])

for model in MODELS:
    print(f"\n=== Processing model: {model} ===")
    results = []

    for _, row in df_all.iterrows():
        species = row[f'latin_name_{model}']
        if pd.isna(species) or species in ("['N/A']", 'nan'):
            species = 'NO_DATA'

        print(f"  Processing: {species}")
        record = get_worms_record(species)
        record['FileName'] = row['file_name']
        record['OriginalName'] = row['node']
        results.append(record)
        time.sleep(0.5)

    df_result = pd.DataFrame(results)

    for col in taxonomy_cols:
        if col in df_result.columns:
            df_result[col] = (df_result[col].astype(str).str.strip().str.capitalize())

    df_result.loc[df_result['Rank'] == 'Species', 'Species'] = df_result['ScientificName']
    df_result.to_csv(f'LLM_worms_files/processed_taxonomic_data_{version}_{model}.csv', sep=';')
