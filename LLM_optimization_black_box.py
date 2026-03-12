import pandas as pd
from openai import OpenAI
import os
import uuid
import random
import time
import json
import anthropic
import argparse
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Process food web taxonomic data.')
    parser.add_argument('--version', type=str, required=True, help='Version string, e.g. v1')
    return parser.parse_args()

args = parse_args()
version = args.version

output_dir = f'Ensembles/black_box_approach/{version}/'
os.makedirs(output_dir, exist_ok=True)

def get_openrouter_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        print("OpenRouter API key loaded")
        return key
    print("OpenRouter API key (OPENROUTER_API_KEY) not found in environment.")
    return None

def get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        print("OpenAI API key loaded")
        return key
    print("OpenAI API key (OPENAI_API_KEY) not found in environment.")
    return None

def get_anthropic_api_key():
    key = os.getenv("ANTHROPIC_API_KEY")
    if key and key.strip():
        print("Anthropic API key loaded")
        return key
    print("Anthropic API key (ANTHROPIC_API_KEY) not found in environment.")
    return None

MODEL_ROUTING = {
    "gpt": {"client": get_openai_api_key(), "id": "gpt-5-mini", "provider": "openai"},
    "claude": {"client": get_anthropic_api_key(), "id": "claude-haiku-4-5", "provider": "anthropic" },
    "qwen": {"client": get_openrouter_api_key(), "id": "qwen/qwen-plus-2025-07-28", "provider": "openai"},
    "gemini": {"client": get_openrouter_api_key(), "id": "google/gemini-2.5-flash", "provider": "openai"}
}

def process_with_model_selection(client, model_name, prompt, provider, max_retries=3):
    """
    Sends a prompt to the selected AI model and returns a response.
    """
    unique_id = str(uuid.uuid4())
    system_msg = (  "You are a senior Marine Ecologist. Your role is to act as an 'LLM-as-a-Judge' to resolve taxonomic discrepancies between different LLMs' outputs (OpenAI, Claude, Gemini, and Qwen)."
                    'RULES FOR ARBITRATION:'
                    '1. TAXONOMIC HIERARCHY INTEGRITY: The final output must be a biologically valid taxonomy (Kingdom -> Phylum -> Class -> Order -> Family -> Genus -> Species).'
                    '2. SCIENTIFIC NOMENCLATURE: Use official Latin names only (e.g., "Gadus morhua" instead of "Cod"). Correct any obvious typos or slight spelling variations.'
                    '3. MAJORITY VS. QUALITY: If three models agree and one disagrees, generally favor the majority. However, if the majority provides a non-existent name and the minority provides a scientifically valid name, favor the valid name.'
                    '4. HANDLING "NA": If a model provides "NA" but the taxon can be inferred from other levels or other models valid responses, fill in the correct taxonomic information.'
                    '5. STRICT OUTPUT: You must respond ONLY with a valid JSON object.'
                    '6. YOUR OWN EXPERTISE: if you can provide a better answer based on your own knowledge, please do it.'
                    '7. CONTEXTUAL DATA: please use the provided location and ecosystem type data to choose the best answer.'
    )

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    temperature=0.7, 
                    system=system_msg,
                    messages=[{"role": "user", "content": f"[UID: {unique_id}]\n\n{prompt}"}]
                )
                
                content = "".join([part.text for part in response.content if hasattr(part, 'text')])
                usage = {"total_tokens": response.usage.input_tokens + response.usage.output_tokens}
            elif model_name.startswith("gpt"):
                response = client.chat.completions.create(
                model=model_name,
                messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"[UID: {unique_id}]\n\n{prompt}"}
                    ],
                )
                content = response.choices[0].message.content
                usage = {"total_tokens": response.usage.total_tokens}
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"[UID: {unique_id}]\n\n{prompt}"}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                )
                content = response.choices[0].message.content
                usage = {"total_tokens": response.usage.total_tokens}

            if content and content.strip():
                return {"status": "ok", "content": content, "usage": usage}
            
        except Exception as e:
            err = str(e).lower()
            print(f"Error with {model_name}: {str(e)[:100]}")
            
            if "rate_limit" in err or "429" in err:
                wait = min(60, attempt * 15) + random.randint(1, 5)
                print(f"Rate limit hit - waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                wait_time = attempt * 2
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    print(f"Failed to get response from {model_name} after {max_retries} attempts")
    return {"status": "error", "content": None, "usage": {"total_tokens": 0}}

def get_metadata_context(network_name, metadata_df):
    """Retrieves information about the location and type of ecosystem."""
    try:
        row = metadata_df.loc[metadata_df["Network name"] == network_name].iloc[0]
        return (
            f"- Location: Lat {row.get('Latitude', 'N/A')}, Lon {row.get('Longitude', 'N/A')}\n"
            f"- Ecobase Type: {row.get('Ecobase type', 'N/A')}\n"
            f"- Ecosystem Type: {row.get('Type', 'N/A')}"
        )
    except Exception:
        return "No specific metadata available for this network."


def prepare_data_payload(row, metadata_str):
    """
    Combines network metadata with model suggestions into a single block of text.
    """
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    model_suffixes = ['openai', 'claude', 'gemini', 'qwen']
    
    payload_parts = [f"### NETWORK CONTEXT:\n{metadata_str}"]
    
    for model in model_suffixes:
        details = [f"{lvl.upper()}: {row.get(f'{lvl}_{model}', 'NA')}" for lvl in taxonomy_levels]
        payload_parts.append(f"### Model {model.upper()} Proposal:\n" + " | ".join(details))
    
    return "\n\n".join(payload_parts)
    
def run_pipeline_in_batches(df, metadata_df, model_configs, batch_size=10):
    """
    The main function that processes data frames in batches.
    model_configs: list of dictionaries [{‘name’: ‘gpt’, ‘client’: client_obj, ‘id’: ‘gpt-5-mini’, ‘provider’: ‘openai’}, ...]
    """
    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE")
    print(f"{'='*60}")
    print(f"Total rows to process: {len(df)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {len(df) // batch_size + (1 if len(df) % batch_size else 0)}")
    print(f"Models configured: {[cfg['name'] for cfg in model_configs]}")
    print(f"{'='*60}\n")
    
    batches = [df.iloc[i:i + batch_size].copy() for i in range(0, len(df), batch_size)]
    processed_batches = []

    for i, batch in enumerate(batches):
        print(f"BATCH {i+1}/{len(batches)} (rows {i*batch_size + 1} to {min((i+1)*batch_size, len(df))})")
    
        for idx, row in batch.iterrows():
            print(f"Row {idx + 1}: Processing node '{row['node']}' from '{row['file_name']}'")
            
            net_context = get_metadata_context(row['file_name'], metadata_df)
            current_payload = prepare_data_payload(row, net_context)
            
            prompt = (
                f'Analyze the following taxonomic proposals for the ecological node: "{row["node"]}" '
                f'(Source: {row["file_name"]}). '
                f"CONTEXTUAL INFORMATION:\n{current_payload}\n\n"
                f'TASK: Reconcile the proposals into a single, authoritative taxonomic lineage. '
                f'Ensure the classification is consistent across all levels. If there is a conflict, '
                f'use your internal biological knowledge and contextual information to select the most accurate scientific classification. '
                f'OUTPUT FORMAT (JSON ONLY): '
                f'{{"kingdom": "Latin name", "phylum": "Latin name", "class": "Latin name", "order": "Latin name", '
                f'"family": "Latin name", "genus": "Latin name", "species": "Latin name", '
                f'"consensus_reasoning": "A brief 1-sentence explanation of why this lineage was chosen in case of model conflict."}}'
            )

            for config in model_configs:
                result = process_with_model_selection(
                    config['client'], 
                    config['id'], 
                    prompt, 
                    config['provider']
                )
                
                col_name = f"res_{config['name']}"
                batch.at[idx, col_name] = result['content'] if result['status'] == 'ok' else None

        with open(f"{output_dir}prompt_{i}.txt", "w") as f:
            f.write(prompt)
        checkpoint_file = f"checkpoint_batch_{i}.csv"
        batch.to_csv(f"{output_dir}{checkpoint_file}", sep=";", index=False)
        print(f"Batch saved to: {checkpoint_file}")
        processed_batches.append(batch)
        
        if i < len(batches) - 1:
            time.sleep(2)

    print(f"{'='*60}")
    print(f"ALL BATCHES PROCESSED")
    print(f"{'='*60}\n")
    
    return pd.concat(processed_batches, ignore_index=True)


def parse_json_responses(df, model_names):
    """
    Parses JSON responses from res_{model} columns and creates separate columns for each taxonomic level.
    """
    print(f"\n{'='*60}")
    print(f"PARSING JSON RESPONSES")
    print(f"{'='*60}\n")
    
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    for model in model_names:
        col_name = f"res_{model}"
        print(f"Parsing responses from {model}...")
        
        if col_name not in df.columns:
            print(f"Column {col_name} not found, skipping...")
            continue
        
        for level in taxonomy_levels:
            new_col = f"{level}_arbitrated_{model}"
            df[new_col] = None
        
        for idx, row in df.iterrows():
            if pd.notna(row[col_name]) and row[col_name]:
                try:
                    json_str = row[col_name].strip()
                    if json_str.startswith("```"):
                        json_str = json_str.replace("```json", "").replace("```", "").strip()
                    
                    data = json.loads(json_str)
                    
                    for level in taxonomy_levels:
                        if level in data:
                            df.at[idx, f"{level}_arbitrated_{model}"] = data[level]
                    
                    if "consensus_reasoning" in data:
                        df.at[idx, f"consensus_reasoning"] = data["consensus_reasoning"]
                    
                except json.JSONDecodeError as e:
                    print(f"Row {idx}: Failed to parse JSON from {model}: {str(e)[:50]}")
                except Exception as e:
                    print(f"Row {idx}: Error processing {model}: {str(e)[:50]}")
        
        print(f"Completed parsing {model}")
    
    print(f"JSON parsing complete")
    return df


def main():
    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print(f"CHOOSE BEST RESPONSE PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    input_file = f'LLM features/Processed/all_models/{version}/ALL_MODELS_COMBINED.xlsx'
    metadata_file = 'dataset_20260126_ecobase/metadata_20260128.xlsx'
    df = pd.read_excel(input_file)
    metadata_df = pd.read_excel(metadata_file)

    model_configs = []
    for model_name, info in MODEL_ROUTING.items():
        if info['client']:
            try:
                if info['provider'] == 'anthropic':
                    client_obj = anthropic.Anthropic(api_key=info['client'])
                elif info['provider'] == 'openai':
                    if model_name in ['qwen', 'gemini']:
                        client_obj = OpenAI(
                            api_key=info['client'],
                            base_url="https://openrouter.ai/api/v1"
                        )
                    else:
                        client_obj = OpenAI(api_key=info['client'])
                
                model_configs.append({
                    'name': model_name,
                    'id': info['id'],
                    'client': client_obj,
                    'provider': info['provider']
                })
            except Exception as e:
                print(f"Failed to configure {model_name}: {str(e)}")
        else:
            print(f"{model_name} skipped (no API key)")
    
    if not model_configs:
        print("ERROR: No models configured! Check your API keys.")
        return
    
    print(f"Total models configured: {len(model_configs)}\n")
    
    result_df = run_pipeline_in_batches(df, metadata_df, model_configs, batch_size=5)
    
    # Parse JSON responses
    model_names = [cfg['name'] for cfg in model_configs]
    result_df = parse_json_responses(result_df, model_names)
    
    # Save results
    output_file = f'{output_dir}arbitrated_taxonomy_all_models.xlsx'

    result_df_cols = ['file_name','node'] + [x for x in result_df.columns if 'arbitrated' in x] 

    result_df[result_df_cols].to_excel(output_file, index=False)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"{'#'*60}")
    print(f"PIPELINE COMPLETED")
    print(f"{'#'*60}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Rows processed: {len(result_df)}")
    print(f"Output file: {output_file}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()