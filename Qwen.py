import sys
import importlib.util
import os
import argparse
import random
import pandas as pd
import json
import time
from dotenv import load_dotenv
import requests
import uuid
import re

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

_fw_path = os.path.join(CURRENT_DIR, "foodwebviz.py")
if os.path.isfile(_fw_path):
    spec = importlib.util.spec_from_file_location("foodwebviz", _fw_path)
    fw = importlib.util.module_from_spec(spec)
    sys.modules["foodwebviz"] = fw
    assert spec.loader is not None
    spec.loader.exec_module(fw)
else:
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    import foodwebviz as fw

DEFAULT_VERSION = 'v1_baseline'

DATA_PATH = os.path.join(CURRENT_DIR, "dataset_20260126_ecobase", "processed")
OUTPUT_PATH = os.path.join(CURRENT_DIR, "LLM features", "qwen")
COST_PATH = os.path.join(CURRENT_DIR, "LLM features", "llmcost")
ENV_PATH = os.path.join(CURRENT_DIR, ".env")
METADATA_PATH = os.path.join(CURRENT_DIR, "dataset_20260126_ecobase", "metadata_20260128.xlsx")

DEFAULT_OPENROUTER_MODEL = "qwen/qwen-plus-2025-07-28"
OPENROUTER_NODES_PER_CYCLE = 2

# ----------------------------
def load_environment_variables():
    if not os.path.exists(ENV_PATH):
        print(f".env file not found at: {ENV_PATH}")
        return False
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    print("Environment variables loaded from .env file")
    return True

def get_openrouter_api_key():
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        return key
    print("OpenRouter API key (OPENROUTER_API_KEY) not found in environment.")
    return None

def setup_openrouter_client(model_name: str = DEFAULT_OPENROUTER_MODEL):
    api_key = get_openrouter_api_key()
    if not api_key:
        return None, None
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        site_url = os.getenv("OPENROUTER_SITE_URL")
        site_title = os.getenv("OPENROUTER_SITE_NAME") or os.getenv("OPENROUTER_SITE_TITLE")
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_title:
            headers["X-Title"] = site_title

        client = {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "headers": headers,
            "model": model_name,
        }
        print(f"OpenRouter client initialized: {model_name}")
        return client, model_name
    except Exception as e:
        print(f"Failed to initialize OpenRouter client: {e}")
        return None, None

def setup_openrouter_openai_client(model_name: str = DEFAULT_OPENROUTER_MODEL):
    api_key = get_openrouter_api_key()
    if not api_key:
        return None, None
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        print(f"Official OpenAI client (pointing at OpenRouter) initialized: {model_name}")
        return client, model_name
    except Exception as e:
        print(f"Failed to initialize OpenAI/OpenRouter client: {e}")
        return None, None

def estimate_tokens(text):
    return len(text) // 4

def create_batches(df, max_nodes_per_request=OPENROUTER_NODES_PER_CYCLE):
    final_batches = []
    n_nodes = len(df)
    for i in range(0, n_nodes, max_nodes_per_request):
        final_batches.append(df.iloc[i:i+max_nodes_per_request])
    return final_batches

def generate_with_openrouter(client, model_name, prompt, max_retries=3, skip_on_rate_limit: bool = False):
    if not client or "headers" not in client:
        return "API_ERROR", None
    
    request_id = str(uuid.uuid4())
    salted_prompt = f"[Request ID: {request_id}\n\n{prompt}]"

    url = f"{client.get('base_url', 'https://openrouter.ai/api/v1')}/chat/completions"
    headers = client["headers"]

    body = {
        "model": model_name,
        "messages": [
            {"role":"system", "content": "You are a senior marine ecologist with expertise in trophic networks and ecological interactions. CRITICAL INSTRUCTION: Your response must be ONLY a valid JSON array. Start with [ and end with ]. Do NOT add any notes, explanations, markdown, or text before or after the JSON. Return NOTHING except the JSON array itself."},
            {"role": "user", "content": salted_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 8000,
    }

    def _extract_content_from_choice(choice):
        try:
            msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            if msg is not None:
                content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content") or p.get("data")
                            if isinstance(t, str):
                                parts.append(t)
                    if parts:
                        return " ".join(parts)
            text = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
            if isinstance(text, str):
                return text
        except Exception:
            return None
        return None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=300)
            if response.status_code == 200:
                result = response.json()
                usage = result.get("usage", {})
                if "choices" in result and result["choices"]:
                    content = _extract_content_from_choice(result["choices"][0])
                    if isinstance(content, str) and content.strip():
                        return content, usage
                    return "NO_CONTENT", usage
                return "NO_CHOICES", usage
            elif response.status_code == 429:
                if skip_on_rate_limit:
                    print(f"Rate limit on {model_name}; skipping retries")
                    break
                delay = min(60, 2 ** attempt + random.randint(1, 5))
                print(f"Rate limit on {model_name}, retrying in {delay}s (attempt {attempt}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                print(f"OpenRouter API error {response.status_code}: {response.text}")
                if attempt < max_retries:
                    time.sleep(min(60, attempt * 5))
                    continue
                return "API_ERROR", None
        except Exception as e:
            print(f"Attempt {attempt} on {model_name} failed: {e}")
            if attempt < max_retries:
                time.sleep(min(60, attempt * 5))
                continue
    return "API_ERROR", None


def generate_with_openrouter_openai(openai_client, model_name, prompt, max_retries=3, skip_on_rate_limit: bool = False):
    if not openai_client:
        print("No OpenRouter client instance provided")
        return "API_ERROR", 0.0
    
    request_id = str(uuid.uuid4())
    salted_prompt = f"[Request ID: {request_id}]\n\n{prompt}"

    extra_headers = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_title = os.getenv("OPENROUTER_SITE_NAME") or os.getenv("OPENROUTER_SITE_TITLE")
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if site_title:
        extra_headers["X-Title"] = site_title

    body = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": salted_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 8000,
    }

    def _extract_content_from_choice(choice):
        try:
            msg = getattr(choice, "message", None) or (choice.get("message") if isinstance(choice, dict) else None)
            if msg is not None:
                content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content") or p.get("data")
                            if isinstance(t, str):
                                parts.append(t)
                    if parts:
                        return " ".join(parts)
            text = getattr(choice, "text", None) or (choice.get("text") if isinstance(choice, dict) else None)
            if isinstance(text, str):
                return text
        except Exception:
            return None
        return None

    for attempt in range(1, max_retries + 1):
        try:
            # The OpenAI client object used here follows the new API surface where
            # chat completions are created via client.chat.completions.create(...)
            completion = openai_client.chat.completions.create(
                extra_headers=extra_headers if extra_headers else None,
                extra_body={},
                model=model_name,
                messages=body["messages"],
                temperature=body["temperature"],
                max_tokens=body["max_tokens"],
                timeout=120,
            )

            if completion and hasattr(completion, "choices") and len(completion.choices) > 0:
                choice = completion.choices[0]
                content = _extract_content_from_choice(choice)
                
                usage = getattr(completion, "usage", None)
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0)
                } if usage else {}
                
                if isinstance(content, str) and content.strip():
                    return content, usage_dict
                # Save raw object repr for debugging
                try:
                    debug_path = os.path.join(OUTPUT_PATH, f"openai_sdk_raw_debug_{int(time.time())}.txt")
                    with open(debug_path, "w", encoding="utf-8") as df:
                        df.write(str(completion))
                    print(f"Saved raw OpenAI/OpenRouter completion for inspection: {debug_path}")
                except Exception:
                    pass
                return "NO_CONTENT", usage_dict
            else:
                return "NO_CHOICES", None

        except Exception as e:
            err = str(e)
            print(f"OpenAI/OpenRouter attempt {attempt} failed: {err}")
            if attempt < max_retries:
                time.sleep(min(60, attempt * 5))
                continue
            return "API_ERROR", None

    return "API_ERROR", None

# ----------------------------
# PROCESS FOODWEB DATA
# ----------------------------
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

def load_augmentation_prompt(df_batch, batch_num, total_batches, metadata_df, network_name, template_file="Input/prompt.txt"):
    try:
        metadata_row = metadata_df.loc[metadata_df["Network name"] == network_name].squeeze()
    except Exception:
        metadata_row = None

    if metadata_row is not None:
        lat = metadata_row.get("Latitude", "Unknown")
        lon = metadata_row.get("Longitude", "Unknown")
        ecobase_type = metadata_row.get("Ecobase type", "Unknown")
        net_type = metadata_row.get("Type", "Unknown")
    else:
        lat = lon = ecobase_type = net_type = "Unknown"

    json_data = df_batch.to_json(orient="records", indent=2)

    metadata_str = f"""
    **Network Metadata:**
    - Network Name: {network_name}
    - Latitude: {lat}
    - Longitude: {lon}
    - Ecobase Type: {ecobase_type}
    - Type: {net_type}
    """

    # Load prompt template from file
    template_path = os.path.join(CURRENT_DIR, template_file)
    with open(template_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    # Replace placeholders
    prompt = prompt_template.format(
        batch_num=batch_num,
        total_batches=total_batches,
        metadata_str=metadata_str,
        json_data=json_data
    )
    
    return prompt


# ----------------------------
# SAVE BATCH RESULTS
# ----------------------------
def save_batch_data(response_text, batch_df, output_path, file_name, batch_num, version):
    os.makedirs(output_path, exist_ok=True)
    raw_output_file = os.path.join(output_path, f"{file_name}_batch_{batch_num}_raw_response_{version}.txt")
    with open(raw_output_file, "w", encoding="utf-8") as f:
        f.write(response_text)
    try:
        json_str = response_text.strip()
        if json_str.startswith("```"):
            json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        # Try to fix common JSON issues
        json_str = json_str.replace('\n', ' ').replace('\r', ' ')
        
        # Attempt to parse JSON with better error handling
        try:
            batch_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempting to fix JSON syntax...")
            
            # Try to fix common JSON issues
            try:
                json_match = re.search(r'\[.*?\]', json_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    batch_data = json.loads(json_str)
                else:
                    json_str = json_str.replace('“', '"').replace('”', '"')  
                    json_str = json_str.replace('’', "'")  
                    batch_data = json.loads(json_str)
            except Exception as fix_error:
                print(f"Could not fix JSON: {fix_error}")
                # Return the original data as a list with just the node names
                batch_data = [{"node": row.get("node", f"node_{i}")} for i, row in batch_df.iterrows()]
        batch_json_file = os.path.join(output_path, f"{file_name}_batch_{batch_num}_augmented.json")
        with open(batch_json_file, "w", encoding="utf-8") as f:
            json.dump(batch_data, f, indent=2, ensure_ascii=False)
        batch_augmented_df = pd.DataFrame(batch_data) if isinstance(batch_data, list) else pd.json_normalize(batch_data)
        batch_csv_file = os.path.join(output_path, f"{file_name}_batch_{batch_num}_augmented.csv")
        batch_augmented_df.to_csv(batch_csv_file, index=False)
        return batch_augmented_df
    except Exception as e:
        print(f"Error processing batch {batch_num}: {str(e)}")
        return None

# ----------------------------
# COMBINE BATCHES
# ----------------------------
def combine_batches(output_path, file_name, version):
    combined_data = []
    batch_files = [f for f in os.listdir(output_path) if f.startswith(f"{file_name}_batch_") and f.endswith("_augmented.json")]
    batch_files.sort(key=lambda x: int(x.split("_batch_")[1].split("_")[0]))
    for batch_file in batch_files:
        with open(os.path.join(output_path, batch_file), "r") as f:
            batch_data = json.load(f)
            if isinstance(batch_data, list):
                combined_data.extend(batch_data)
            else:
                combined_data.append(batch_data)
    if combined_data:
        combined_json_file = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.json")
        with open(combined_json_file, "w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        combined_csv_file = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.csv")
        pd.DataFrame(combined_data).to_csv(combined_csv_file, index=False)
        return len(combined_data)
    return 0

# ----------------------------
# MISSING NODE CHECK + REPAIR (optional)
# ----------------------------
def _read_combined(output_path, file_name, version):
    path = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
    except Exception:
        return []

def _save_combined(output_path, file_name, records, version):
    seen = set()
    uniq = []
    for rec in records:
        name = str(rec.get("node", "")).strip()
        if name and name not in seen:
            seen.add(name)
            uniq.append(rec)
    combined_json = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.json")
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(uniq, f, indent=2, ensure_ascii=False)
    combined_csv = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.csv")
    pd.DataFrame(uniq).to_csv(combined_csv, index=False)
    return len(uniq)

def _parse_response_records(response_text, fallback_node=None):
    try:
        s = response_text.strip()
        if s.startswith("```"):
            s = s.replace("```json", "").replace("```", "").strip()
        data = json.loads(s)
        if isinstance(data, dict):
            data = [data]
        if fallback_node:
            for rec in data:
                if not str(rec.get("node", "")).strip():
                    rec["node"] = fallback_node
        return data
    except Exception:
        return []

def repair_missing_nodes(client, model_name, df_original, metadata_df, network_name, output_path, base_name, version, max_repairs=0, max_retries=2, skip_on_rate_limit=False, client_type='requests'):
    original_nodes = set(df_original["node"].astype(str)) if "node" in df_original.columns else set()
    combined = _read_combined(output_path, base_name, version)
    have_nodes = set()
    for rec in combined:
        v = rec.get("node")
        if v is not None:
            have_nodes.add(str(v))
    missing = sorted(list(original_nodes - have_nodes))
    if not missing:
        return 0
    print(f"Missing {len(missing)} nodes; attempting single-node repairs…")
    added = []
    repairs = 0
    for node_name in missing:
        if max_repairs and repairs >= max_repairs:
            print("Max repair limit reached; stopping repairs.")
            break
        row_df = df_original[df_original["node"].astype(str) == node_name]
        if row_df.empty:
            continue
        prompt = load_augmentation_prompt(row_df, 1, 1, metadata_df, network_name)
        
        if client_type == 'openai':
            resp, usage = generate_with_openrouter_openai(client, model_name, prompt, max_retries=max_retries, skip_on_rate_limit=skip_on_rate_limit)
        else:
            resp, usage = generate_with_openrouter(client, model_name, prompt, max_retries=max_retries, skip_on_rate_limit=skip_on_rate_limit)
            
        recs = _parse_response_records(resp, fallback_node=node_name) if isinstance(resp, str) else []
        if recs:
            added.extend(recs)
            repairs += 1
    if not added:
        return 0
    all_recs = combined + added
    _save_combined(output_path, base_name, all_recs, version)
    return len(added)

def main():
    parser = argparse.ArgumentParser(description="Enrich food web nodes with Qwen (via OpenRouter) and save augmented traits")
    parser.add_argument(
        "--version",
        dest="version",
        default=DEFAULT_VERSION,
        help=f"Version identifier for output files (default: {DEFAULT_VERSION}). Example: v2_temperature_0.5"
    )
    parser.add_argument("--only", dest="only_base", help="Process only this base filename (without .scor)")
    parser.add_argument(
        "--start-after",
        dest="start_after",
        help="In lexicographic order, start processing after this base filename (without .scor)",
    )
    parser.add_argument(
        "--start-from",
        dest="start_from",
        help="In lexicographic order, start processing from this base filename (inclusive, without .scor)",
    )
    parser.add_argument(
        "--prefer-openai-client",
        action="store_true",
        help="Prefer the official OpenAI client; otherwise use requests (default).",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=DEFAULT_OPENROUTER_MODEL,
        help=f"Model name to use (default: {DEFAULT_OPENROUTER_MODEL})",
    )
    parser.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=3,
        help="Max retries per call on transient errors",
    )
    parser.add_argument(
        "--skip-on-rate-limit",
        dest="skip_on_rate_limit",
        action="store_true",
        help="If set, do not wait on 429 limits; skip retries and move on",
    )
    parser.add_argument(
        "--nodes-per-batch",
        dest="nodes_per_batch",
        type=int,
        default=None,
        help="Override nodes per batch for requests",
    )
    args = parser.parse_args()

    version = args.version
    OUTPUT_PATH = os.path.join(CURRENT_DIR, "LLM features", "Qwen", version)
    
    print("\n" + "=" * 60)
    print(f"qwen.py - Version: {version}")
    print(f"Output directory: {OUTPUT_PATH}")
    print("=" * 60 + "\n")

    load_environment_variables()
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    metadata_df = pd.read_excel(METADATA_PATH)
    
    client = None
    model_name = args.model_name
    client_type = 'requests'
    
    if args.prefer_openai_client:
        openai_client, model_name = setup_openrouter_openai_client(args.model_name)
        if openai_client:
            client = openai_client
            client_type = 'openai'
    
    if client is None:
        client, model_name = setup_openrouter_client(args.model_name)
        client_type = 'requests' if client else None
        
    if not client:
        return

    scor_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".scor")])

    if args.only_base:
        target = f"{args.only_base}.scor"
        scor_files = [target] if target in scor_files else []
    if args.start_after:
        marker = f"{args.start_after}.scor"
        scor_files = scor_files[scor_files.index(marker) + 1:] if marker in scor_files else scor_files
    if args.start_from:
        marker = f"{args.start_from}.scor"
        scor_files = scor_files[scor_files.index(marker):] if marker in scor_files else scor_files
    
    scor_files = scor_files[:5]

    for scor_file in scor_files:
        file_path = os.path.join(DATA_PATH, scor_file)
        print(f"\nProcessing file: {file_path}")
        
        df = process_foodweb_data(file_path)
        
        if args.nodes_per_batch and args.nodes_per_batch > 0:
            nodes_per_cycle = args.nodes_per_batch
        else:
            nodes_per_cycle = OPENROUTER_NODES_PER_CYCLE
            
        batches = create_batches(df, max_nodes_per_request=nodes_per_cycle)
        total_batches = len(batches)
        base_name = scor_file.replace(".scor", "")
        successful_batches = 0

        for batch_num, batch_df in enumerate(batches, 1):
            node_list = batch_df["node"].astype(str).tolist() if "node" in batch_df.columns else batch_df.index.astype(str).tolist()
            print(f"\nBatch {batch_num}/{total_batches} nodes ({len(node_list)}): {', '.join(node_list)}")

            prompt = load_augmentation_prompt(batch_df, batch_num, total_batches, metadata_df, base_name)
            
            if batch_num == 1:
                try:
                    prompt_file = os.path.join(OUTPUT_PATH, f"{base_name}_sample_prompt_{version}.txt")
                    with open(prompt_file, "w", encoding="utf-8") as pf:
                        pf.write(prompt)
                    print(f"Sample prompt (batch 1) saved at: {prompt_file}")
                except Exception as e:
                    print(f"Failed to write sample prompt for {base_name}: {e}")

            if client_type == 'openai':
                response, usage = generate_with_openrouter_openai(
                    client,
                    model_name,
                    prompt,
                    max_retries=args.max_retries,
                    skip_on_rate_limit=args.skip_on_rate_limit,
                )
            else:
                response, usage = generate_with_openrouter(
                    client,
                    model_name,
                    prompt,
                    max_retries=args.max_retries,
                    skip_on_rate_limit=args.skip_on_rate_limit,
                )
            
            if response not in ["API_ERROR", "NO_CONTENT", "NO_CHOICES", "CONTENT_FILTERED"]:
                result = save_batch_data(response, batch_df, OUTPUT_PATH, base_name, batch_num, version)
                if result is not None:
                    successful_batches += 1
                print(f"Batch {batch_num} successful.")
            else:
                print(f"Batch {batch_num} failed with: {response}.")
            
            time.sleep(2)

        if successful_batches > 0:
            total_combined = combine_batches(OUTPUT_PATH, base_name, version)
            try:
                added = repair_missing_nodes(
                    client=client,
                    model_name=model_name,
                    df_original=df,
                    metadata_df=metadata_df,
                    network_name=base_name,
                    output_path=OUTPUT_PATH,
                    base_name=base_name,
                    version=version,
                    max_repairs=10,
                    max_retries=2,
                    skip_on_rate_limit=args.skip_on_rate_limit,
                    client_type=client_type
                )
                suffix = f", repaired {added} missing" if added else ""
            except Exception as _:
                suffix = ""
            print(f"{scor_file}: {successful_batches}/{total_batches} batches, {total_combined} total nodes{suffix}")
        else:
            print(f"No successful batches for {scor_file}")

    print("\nProcessing complete!")
    print(f"Check outputs in: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()