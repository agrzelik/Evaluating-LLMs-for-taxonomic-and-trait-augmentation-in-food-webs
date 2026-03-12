import pandas as pd
from openai import OpenAI
import os
import uuid
import random
import time
import json
import anthropic
import argparse
import ast
from dotenv import load_dotenv
from datetime import datetime


load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description='Process food web taxonomic data.')
    parser.add_argument('--version', type=str, required=True, help='Version string, e.g. v1')
    return parser.parse_args()

args = parse_args()
version = args.version

output_dir = f'Ensembles/cross_evaluated_soft_voting/{version}/'
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
    "chatgpt": {"client": get_openai_api_key(), "id": "gpt-5-mini", "provider": "openai"},
    "claude": {"client": get_anthropic_api_key(), "id": "claude-haiku-4-5", "provider": "anthropic" },
    "qwen": {"client": get_openrouter_api_key(), "id": "qwen/qwen-plus-2025-07-28", "provider": "openai"},
    "gemini": {"client": get_openrouter_api_key(), "id": "google/gemini-2.5-flash", "provider": "openai"}
}

def process_with_model_selection(client, model_name, prompt, provider, max_retries=3):
    """
    Sends a prompt to the selected AI model and returns a response.
    """
    unique_id = str(uuid.uuid4())
    system_msg = (  '''
                    You are a senior Marine Ecologist evaluating taxonomic outputs.
                    For each model (OpenAI, Claude, Gemini, Qwen), you must evaluate EACH taxonomic level separately.
                    
                    For EACH of the 7 taxonomic levels (Kingdom, Phylum, Class, Order, Family, Genus, Species), 
                    assign a score from 0 to 1 for EACH model reflecting scientific correctness.
                    
                    Criteria for each level:
                    - Correct Latin nomenclature
                    - Biological validity for that specific level (use your own knowledge and contextual data)
                    - Consistency with other levels in the hierarchy
                    - Proper handling of "NA"
                    
                    Score independently for each level and each model.
                    No consensus.
                    No explanation.
                    Be judgemental and precise, don't score all answers with 1.
                    
                    Return scores in a nested JSON structure.'''
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
    model_suffixes = ['chatgpt', 'claude', 'gemini', 'qwen']
    
    payload_parts = [f"### NETWORK CONTEXT:\n{metadata_str}"]
    
    for model in model_suffixes:
        details = [f"{lvl.upper()}: {row.get(f'{lvl}_{model}', 'NA')}" for lvl in taxonomy_levels]
        payload_parts.append(f"### Model {model.upper()} Proposal:\n" + " | ".join(details))
    
    return "\n\n".join(payload_parts)
    
def run_pipeline_in_batches(df, metadata_df, model_configs, batch_size=10):
    """
    The main function that processes data frames in batches.
    model_configs: list of dictionaries [{‘name’: ‘gpt’, ‘client’: client_obj, ‘id’: ‘gpt-5-mini’, ‘provider’: ‘chatgpt’}, ...]
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
            

            prompt = f'''
Analyze the following taxonomic proposals for the ecological node: "{row['node']}" (Source: {row['file_name']}).

INPUT DATA FROM MODELS:
{current_payload}

TASK:
Evaluate EACH taxonomic level (Kingdom, Phylum, Class, Order, Family, Genus, Species) for EACH model independently.

For each of the 4 models (OpenAI, Claude, Gemini, Qwen), assign a score between 0 and 1 for EACH of the 7 taxonomic levels.

Score each level based on:
- Correctness of Latin scientific nomenclature at that level
- Biological validity at that specific level (use contextual data and your own biological knowledge)
- Consistency with the overall taxonomic hierarchy
- Appropriate handling of missing values ("NA")

Do NOT reconcile into a single lineage.
Do NOT select a consensus taxonomy.
Do NOT perform majority voting.

Be precise and judgemental, don't score all options with 1.

OUTPUT FORMAT (JSON ONLY):
{{
  "OpenAI": {{
    "kingdom": <float 0-1>,
    "phylum": <float 0-1>,
    "class": <float 0-1>,
    "order": <float 0-1>,
    "family": <float 0-1>,
    "genus": <float 0-1>,
    "species": <float 0-1>
  }},
  "Claude": {{
    "kingdom": <float 0-1>,
    "phylum": <float 0-1>,
    "class": <float 0-1>,
    "order": <float 0-1>,
    "family": <float 0-1>,
    "genus": <float 0-1>,
    "species": <float 0-1>
  }},
  "Gemini": {{
    "kingdom": <float 0-1>,
    "phylum": <float 0-1>,
    "class": <float 0-1>,
    "order": <float 0-1>,
    "family": <float 0-1>,
    "genus": <float 0-1>,
    "species": <float 0-1>
  }},
  "Qwen": {{
    "kingdom": <float 0-1>,
    "phylum": <float 0-1>,
    "class": <float 0-1>,
    "order": <float 0-1>,
    "family": <float 0-1>,
    "genus": <float 0-1>,
    "species": <float 0-1>
  }}
}}
'''
            
            for config in model_configs:
                result = process_with_model_selection(
                    config['client'], 
                    config['id'], 
                    prompt, 
                    config['provider']
                )
                
                col_name = f"res_{config['name']}"
                batch.at[idx, col_name] = result['content'] if result['status'] == 'ok' else None

        checkpoint_file = f"checkpoint_batch_{i}.csv"
        batch.to_csv(f"{output_dir}{checkpoint_file}", sep=";", index=False)
        print(f"Batch saved to: {checkpoint_file}")
        processed_batches.append(batch)
        
        if i < len(batches) - 1:
            time.sleep(2)

    print(f"\n{'='*60}")
    print(f"ALL BATCHES PROCESSED")
    print(f"{'='*60}\n")
    
    return pd.concat(processed_batches, ignore_index=True)


def parse_probability_responses(df, judge_model_names):
    """
    Parses JSON responses containing probabilities (scores) for each model and each taxonomy level.
    Creates columns: kingdom_chatgpt_score_by_gpt, phylum_chatgpt_score_by_gpt, etc.
    """
    print(f"{'='*60}")
    print(f"PARSING PROBABILITY RESPONSES")
    print(f"{'='*60}\n")
    
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    evaluated_models_variants = {
        'chatgpt': ['OpenAI', 'OPENAI', 'openai','gpt'],
        'claude': ['Claude', 'CLAUDE', 'claude', 'Anthropic','anthropic'],
        'gemini': ['Gemini', 'GEMINI', 'gemini'],
        'qwen': ['Qwen', 'QWEN', 'qwen']
    }
    
    for judge in judge_model_names:
        col_name = f"res_{judge}"
        print(f"Parsing scores from judge: {judge}...")
        
        if col_name not in df.columns:
            print(f"Column {col_name} not found, skipping...")
            continue
        
        for model_key in evaluated_models_variants.keys():
            for level in taxonomy_levels:
                score_col = f"{level}_{model_key}_score_by_{judge}"
                df[score_col] = None
        
        for idx, row in df.iterrows():
            if pd.notna(row[col_name]) and row[col_name]:
                try:
                    json_str = row[col_name].strip()
                    
                    if json_str.startswith("```"):
                        json_str = json_str.replace("```json", "").replace("```", "").strip()
                    
                    data = json.loads(json_str)
                    
                    if "model_scores" in data:
                        data = data["model_scores"]
                    
                    for model_key, variants in evaluated_models_variants.items():
                        model_data = None
                        for variant in variants:
                            if variant in data:
                                model_data = data[variant]
                                break
                        
                        if model_data is None:
                            print(f"Row {idx}: No data found for {model_key} by {judge}")
                            continue
             
                        if isinstance(model_data, (int, float)):
                            print(f"Row {idx}: {judge} returned single score for {model_key}, applying to all levels")
                            for level in taxonomy_levels:
                                score_col = f"{level}_{model_key}_score_by_{judge}"
                                try:
                                    score_value = float(model_data)
                                    if 0 <= score_value <= 1:
                                        df.at[idx, score_col] = score_value
                                except (ValueError, TypeError):
                                    pass

                        elif isinstance(model_data, dict):
                            for level in taxonomy_levels:
                                level_variants = [level, level.lower(), level.upper(), level.capitalize()]
                                score_value = None
                                
                                for level_variant in level_variants:
                                    if level_variant in model_data:
                                        score_value = model_data[level_variant]
                                        break
                                
                                if score_value is not None:
                                    score_col = f"{level}_{model_key}_score_by_{judge}"
                                    try:
                                        score_value = float(score_value)
                                        if 0 <= score_value <= 1:
                                            df.at[idx, score_col] = score_value
                                        else:
                                            print(f"Row {idx}: Score {score_value} out of range [0,1] for {level} of {model_key} by {judge}")
                                    except (ValueError, TypeError):
                                        print(f"Row {idx}: Invalid score value '{score_value}' for {level} of {model_key} by {judge}")
                                else:
                                    print(f"Row {idx}: No score for {level} of {model_key} by {judge}")
                        else:
                            print(f"Row {idx}: Unexpected data type for {model_key} by {judge}: {type(model_data)}")
                    
                except json.JSONDecodeError as e:
                    print(f"Row {idx}: Failed to parse JSON from {judge}: {str(e)[:100]}")
                    print(f"Raw content: {row[col_name][:300]}...")
                except Exception as e:
                    print(f"Row {idx}: Error processing {judge}: {str(e)[:100]}")
        
        print(f"Completed parsing {judge}")
    
    print(f"Probability parsing complete\n")
    return df

def restructure_columns_for_output(df):
    """
    Reorganizes columns so that for each model there are:
    1. All taxonomic levels (kingdom_openai, phylum_openai, ...)
    2. All scores for these levels (kingdom_openai_score_by_gpt, kingdom_openai_score_by_claude, ...)
    """
    print("\nRestructuring columns for output...")
    
    base_cols = ['file_name', 'node']
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    model_names = ['chatgpt', 'claude', 'gemini', 'qwen']
    judge_names = ['chatgpt', 'claude', 'qwen', 'gemini']
    
    structured_cols = base_cols.copy()
    
    for model in model_names:
        print(f"Processing {model}...")
        for level in taxonomy_levels:
            taxonomy_col = f"{level}_{model}"
            if taxonomy_col in df.columns:
                structured_cols.append(taxonomy_col)
            else:
                print(f"Missing column: {taxonomy_col}")
            
            for judge in judge_names:
                score_col = f"{level}_{model}_score_by_{judge}"
                if score_col in df.columns:
                    structured_cols.append(score_col)

        res_col = f"res_{model}"
        if res_col in df.columns:
            structured_cols.append(res_col)

    remaining_cols = [col for col in df.columns if col not in structured_cols]
    structured_cols.extend(remaining_cols)
    
    print(f"Total columns structured: {len(structured_cols)}")

    return df[structured_cols]


def safe_float_convert(conf):
    if pd.isna(conf):
        return None
    try:
        return float(str(conf).replace(',', '.'))
    except (ValueError, TypeError):
        try:
            if isinstance(conf, str):
                parsed = ast.literal_eval(conf.replace(',', '.'))
                if isinstance(parsed, list):
                    return sum(float(x) for x in parsed) / len(parsed)
                return float(parsed)
            elif isinstance(conf, list):
                return sum(float(x) for x in conf) / len(conf)
        except:
            return None
    return None


def get_best_value_by_scorer(row, level, scorer, models):
    """
    Option 1: one selected scorer, all models.
    Score for each value = sum of scores from that scorer across all models.
    Normalization by number of models.
    """
    scores = {}

    for model in models:
        val_col = f'{level}_{model}'
        score_col = f'{level}_{model}_score_by_{scorer}'

        if val_col not in row or score_col not in row:
            continue

        val = row[val_col]
        if pd.isna(val) or str(val).strip() == '':
            continue

        s = safe_float_convert(row[score_col])
        if s is None:
            continue

        val_clean = str(val).strip()
        scores[val_clean] = scores.get(val_clean, 0) + s

    if not scores:
        return None, None

    best_val = max(scores, key=scores.get)
    avg_score = scores[best_val] / len(models)
    return best_val, avg_score


def get_best_value_all(row, level, models, scorers):
    """
    Option 2: all models and all scorers.
    Score for each value = sum of scores from all scorers across all models.
    Normalization by number of models * number of scorers.
    """
    scores = {}

    for model in models:
        val_col = f'{level}_{model}'

        if val_col not in row:
            continue

        val = row[val_col]
        if pd.isna(val) or str(val).strip() == '':
            continue

        val_clean = str(val).strip()

        for scorer in scorers:
            score_col = f'{level}_{model}_score_by_{scorer}'
            if score_col not in row:
                continue

            s = safe_float_convert(row[score_col])
            if s is not None:
                scores[val_clean] = scores.get(val_clean, 0) + s

    if not scores:
        return None, None

    best_val = max(scores, key=scores.get)
    avg_score = scores[best_val] / (len(models) * len(scorers))
    return best_val, avg_score


def main():
    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print(f"TAXONOMIC SCORING PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    input_file = f'LLM features/Processed/all_models/{version}/ALL_MODELS_COMBINED.xlsx'
    metadata_file = 'dataset_20260126_ecobase/metadata_20260128.xlsx'
    df = pd.read_excel(input_file)
    metadata_df = pd.read_excel(metadata_file)

    print(f"Loaded {len(df)} rows\n")

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
                print(f"{model_name} configured")
            except Exception as e:
                print(f"Failed to configure {model_name}: {str(e)}")
        else:
            print(f"{model_name} skipped (no API key)")
    
    if not model_configs:
        print("ERROR: No models configured! Check your API keys.")
        return
    
    print(f"Total models configured: {len(model_configs)}\n")
    
    result_df = run_pipeline_in_batches(df, metadata_df, model_configs, batch_size=5)

    judge_model_names = [cfg['name'] for cfg in model_configs]
    result_df = parse_probability_responses(result_df, judge_model_names)

    result_df = restructure_columns_for_output(result_df)
    
    result_cols = [x for x in result_df.columns
                   if not any (y in x for y in ['res','source','included','representative','aggregation'])]

    output_file = f'{output_dir}taxonomy_level_scores_all_models.xlsx'
    result_df[result_cols].to_excel(output_file, index=False)
    print(f"Saved to: {output_file}")

    print("\n" + "="*60)
    print("SCORING STATISTICS BY TAXONOMIC LEVEL")
    print("="*60)
    
    taxonomy_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    model_names = ['chatgpt', 'claude', 'gemini', 'qwen']
    
    for level in taxonomy_levels:
        print(f"{level.upper()}:")
        for model in model_names:
            score_cols = [col for col in result_df.columns if col.startswith(f"{level}_{model}_score_")]
            if score_cols:
                all_scores = pd.concat([result_df[col].dropna() for col in score_cols])
                if len(all_scores) > 0:
                    print(f"  {model}: Mean={all_scores.mean():.3f}, Min={all_scores.min():.3f}, Max={all_scores.max():.3f}, Count={len(all_scores)}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'#'*60}")
    print(f"PIPELINE COMPLETED")
    print(f"{'#'*60}")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"Rows processed: {len(result_df)}")
    print(f"Output file: {output_file}")
    print(f"{'#'*60}\n")

    models = ['chatgpt', 'claude', 'gemini', 'qwen']
    scorers = ['chatgpt', 'claude', 'qwen', 'gemini']
    taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

    new_columns = {}

    for scorer in scorers:
        for level in taxonomic_levels:
            computed = result_df.apply(
                lambda r, l=level, sc=scorer: pd.Series(get_best_value_by_scorer(r, l, sc, models)), axis=1
            )
            new_columns[f'{level}_best_{scorer}']       = computed[0]
            new_columns[f'{level}_best_{scorer}_score'] = computed[1]

    for level in taxonomic_levels:
        computed = result_df.apply(
            lambda r, l=level: pd.Series(get_best_value_all(r, l, models, scorers)), axis=1
        )
        new_columns[f'{level}_best_all']       = computed[0]
        new_columns[f'{level}_best_all_score'] = computed[1]

    result_df = pd.concat([result_df, pd.DataFrame(new_columns, index=result_df.index)], axis=1)

    for scorer in scorers + ['all']:
        cols_to_keep = ['file_name', 'node'] + [x for x in result_df.columns if f'best_{scorer}' in x]
        df_small = result_df[cols_to_keep]
        df_small.to_csv(f'Ensembles/cross_evaluated_soft_voting/{version}/best_scoring_{scorer}.csv', sep=';')


if __name__ == "__main__":
    main()