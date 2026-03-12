import pandas as pd
import requests
import time

df = pd.read_excel('Input/annotation_file_0217_CORRECTED_FINE_VERSION_JP_0103.xlsx')
df.rename(columns={'node': 'Node name', 'file_name': 'File name'}, inplace=True)

df_full = df[df['IsAlive']==True]
df_full = df_full[df_full['Latin name'].notna() | df_full['AphiaID'].notna()]
df_full['Latin name'] = df_full['Latin name'].str.replace('spp.','',regex=False).str.strip()
df_full['AphiaID'] = df_full['AphiaID']

df_full = df_full.drop_duplicates(subset=['Node name', 'File name', 'AphiaID'], keep='first')
df_full = df_full.drop_duplicates(subset=['Node name', 'File name', 'Latin name'], keep='first')

def get_multiple_records(input_data, limit=5):
    """
    Retrieves taxonomic records based on species name or AphiaID.
    If input_data is a string of numbers, it searches by ID. Otherwise, it searches by name.
    """
    is_id = isinstance(input_data, float) and input_data.is_integer()
    
    if is_id:
        input_data = str(int(input_data))
        url = f"https://www.marinespecies.org/rest/AphiaRecordByAphiaID/{input_data}"
    else:
        base_url = "https://www.marinespecies.org/rest/AphiaRecordsByName"
        url = f"{base_url}/{input_data}?like=true&marine_only=false"
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            records = [data] if is_id else data[:limit]
            
            results = []
            for record in records:
                results.append({
                    "QueryInput": input_data,
                    "ScientificName": record.get("scientificname"),
                    "AphiaID": record.get("AphiaID"),
                    "Rank": record.get("rank"),
                    "Status": record.get("status"),
                    "Kingdom": record.get("kingdom"),
                    "Phylum": record.get("phylum"),
                    "Class": record.get("class"),
                    "Order": record.get("order"),
                    "Family": record.get("family"),
                    "Genus": record.get("genus"),
                    "Species": record.get("species")
                })
            return results
            
        return [{"QueryInput": input_data, "Status": "Not Found"}]
        
    except Exception as e:
        return [{"QueryInput": input_data, "Status": f"Error: {e}"}]

all_results = []
for idx, row in df_full.iterrows():
    input_data = row['AphiaID'] if pd.notna(row['AphiaID']) and row['AphiaID'] != '' else row['Latin name']
    print(f"Processing: {(input_data)}")
    species_records = get_multiple_records(input_data, limit=1)
    species_records = [{**record, 'FileName': row['File name'], 'Node name': row['Node name'], 'IsAlive':row['IsAlive']} for record in species_records]
    all_results.extend(species_records)
    time.sleep(0.5)

df_all = pd.DataFrame(all_results)
df_all.to_csv('baseline/baseline_file_raw.csv', sep=';', index=False)

total_queries = len(df_full)
total_results = len(df_all)

print('STATS')
print(f'Queries attempted: {total_queries}')
print(f'Records returned: {total_results}')
print(f'Percent generated: {total_results / total_queries:.2%}')

df_ok = df_all[df_all['Status'] != 'Not Found'].copy()

ok_results = len(df_ok)

print('STATS')
print(f'Records with valid status: {ok_results}')
print(f'Percent status ok: {ok_results / total_queries:.2%}')

df_ok.loc[df_ok['Rank'] == 'Species', 'Species'] = df_ok['ScientificName']

df_ok.to_csv('baseline/baseline_file_1.csv', sep=';', index=False)
