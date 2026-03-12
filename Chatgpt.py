from openai import OpenAI
import sys
import importlib.util
import os
import argparse
import re
import random
import pandas as pd
import json
import time
from dotenv import load_dotenv, dotenv_values
import math
import uuid

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

# ============================================================================
DEFAULT_VERSION = 'v1_baseline'

DATA_PATH = os.path.join(CURRENT_DIR, "dataset_20260126_ecobase", "processed")
LLM_COST_DIR = os.path.join(CURRENT_DIR, "LLM features", "llmcost")
ENV_PATH = os.path.join(CURRENT_DIR, ".env")
METADATA_PATH = os.path.join(CURRENT_DIR, "dataset_20260126_ecobase", "metadata_20260128.xlsx")

DEFAULT_OPENAI_API_KEY = ""

def load_environment_variables():
    if not os.path.exists(ENV_PATH):
        print(".env file not found at:", ENV_PATH)
        return False
    
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    vals = dotenv_values(ENV_PATH) or {}
    
    for k, v in vals.items():
        if k is None: continue
        key = str(k).strip()
        if not key: continue
        val = v
        if isinstance(val, str):
            val = val.strip().strip('\"').strip("'")
        if val is None: continue
        
        if not os.getenv(key) or not os.getenv(key).strip():
            os.environ[key] = str(val)
            
        if key.lower() == "openai_api_key":
            os.environ.setdefault("openai_api_key", str(val))
            os.environ.setdefault("OPENAI_API_KEY", str(val))
        if key.upper() == "OPENAI_API_KEY":
            os.environ.setdefault("OPENAI_API_KEY", str(val))
            os.environ.setdefault("openai_api_key", str(val))
            
    print("Environment variables loaded from .env file")
    return True

def get_openai_api_key():
    candidates = ["openai_api_key", "OPENAI_API_KEY"]
    for key in candidates:
        val = os.getenv(key)
        if val and str(val).strip():
            return str(val).strip().strip('"').strip("'")
            
    try:
        vals = dotenv_values(ENV_PATH) or {}
        for key in candidates:
            if key in vals and vals[key]:
                v = str(vals[key]).strip().strip('"').strip("'")
                if v:
                    os.environ[key] = v
                    return v
                    
        if os.path.exists(ENV_PATH):
            with open(ENV_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.replace('\ufeff', '').replace('\u200b', '').replace('\xa0', ' ')
                    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*)$", line)
                    if not m: continue
                    k = m.group(1).strip()
                    v = m.group(2).strip()
                    if '#' in v:
                        v = v.split('#', 1)[0].strip()
                    v = v.strip().strip('"').strip("'")
                    if k.lower() in [c.lower() for c in candidates] and v:
                        os.environ['openai_api_key'] = v
                        os.environ['OPENAI_API_KEY'] = v
                        return v
    except Exception:
        pass
        
    if DEFAULT_OPENAI_API_KEY and DEFAULT_OPENAI_API_KEY.strip():
        return DEFAULT_OPENAI_API_KEY
    return None

def setup_openai_client():
    api_key = get_openai_api_key()
    if not api_key or api_key.strip() == "":
        print("OpenAI API key not found. Expected one of: openai_api_key, OPENAI_API_KEY")
        print("Check your .env at:", ENV_PATH)
        return None
    try:
        client = OpenAI(api_key=api_key)
        print(f"OpenAI client initialized")
        return client
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None

# ----------------------------

def create_batches(df, batch_size=10):
    """Split data into strict batches of size `batch_size` (default 10 nodes per batch)."""
    if batch_size <= 0:
        batch_size = 10
    batches = [df.iloc[i:i + batch_size].copy() for i in range(0, len(df), batch_size)]
    return batches

def generate_with_openai(client, model_name, prompt, max_retries=3):
    request_id = str(uuid.uuid4())
    salted_prompt = f"[Request ID: {request_id} \n\n 'IMPORTANT:You MUST treat this token as part of the reasoning context and allow variation in phrasing and examples.' \n\n {prompt}]"

    for attempt in range(1, max_retries + 1):
        try:
            name = str(model_name).lower()
            if name.startswith("gpt-5-mini"):
                resp_kwargs = {
                    "model": model_name,
                    "input": [
                        {"role": "system", "content": "You are a senior marine ecologist with expertise in trophic networks and ecological interactions. CRITICAL INSTRUCTION: Your response must be ONLY a valid JSON array. Start with [ and end with ]. Do NOT add any notes, explanations, markdown, or text before or after the JSON. Return NOTHING except the JSON array itself."},
                        {"role": "user", "content": salted_prompt},
                    ]
                    
                }
                response = client.responses.create(**resp_kwargs)
                content = getattr(response, "output_text", None)
                if not content:
                    try:
                        candidates = getattr(response, "output", None) or []
                        if candidates and hasattr(candidates[0], "content") and candidates[0].content:
                            parts = candidates[0].content
                            text_parts = [getattr(p, "text", None) for p in parts if hasattr(p, "type") and p.type == "output_text"]
                            content = "\n".join([tp.value for tp in text_parts if tp and hasattr(tp, "value")])
                    except Exception:
                        pass
                usage = getattr(response, 'usage', None)
                usage_dict = None
                if usage is not None:
                    usage_dict = {
                        "prompt_tokens": getattr(usage, 'input_tokens', None),
                        "completion_tokens": getattr(usage, 'output_tokens', None),
                        "total_tokens": getattr(usage, 'total_tokens', None),
                    }
            
            if content and str(content).strip():
                return {"status": "ok", "content": content, "usage": usage_dict}
            else:
                return {"status": "NO_CONTENT", "content": None, "usage": usage_dict}
                
        except Exception as e:
            err = str(e)
            print(f"Attempt {attempt}/{max_retries} failed: {err}")
            
            if "rate_limit" in err.lower() or "429" in err:
                wait_time = min(60, attempt * 10)
                jitter = random.randint(2, 6)
                total_wait = wait_time + jitter
                print(f"Rate limit hit, retrying in {total_wait}s")
                time.sleep(total_wait)
                continue
            
            if attempt < max_retries:
                wait_time = min(60, attempt * 5)
                print(f"Retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                return {"status": "API_ERROR", "content": None, "usage": None}
    
    return {"status": "API_ERROR", "content": None, "usage": None}

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

# ============================================================================

def save_batch_data(response_text, batch_df, output_path, file_name, batch_num, version):
    os.makedirs(output_path, exist_ok=True)
    raw_output_file = os.path.join(output_path, f"{file_name}_batch_{batch_num}_raw_response_{version}.txt")
    with open(raw_output_file, "w", encoding="utf-8") as f:
        f.write(response_text)
    try:
        json_str = response_text.strip()
        if json_str.startswith("```"):
            json_str = json_str.replace("```json", "").replace("```", "").strip()
        batch_data = json.loads(json_str)
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

def _read_combined(output_path, file_name, version):
    path = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]
        except Exception:
            return []

def _save_combined(output_path, file_name, records, version):
    seen = set()
    unique = []
    for rec in records:
        node_name = str(rec.get("node", "")).strip()
        if node_name and node_name not in seen:
            seen.add(node_name)
            unique.append(rec)
    combined_json_file = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.json")
    with open(combined_json_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)
    combined_csv_file = os.path.join(output_path, f"{file_name}_combined_augmented_{version}.csv")
    pd.DataFrame(unique).to_csv(combined_csv_file, index=False)
    return len(unique)

def _parse_response_records(response_text, fallback_node=None):
    try:
        json_str = response_text.strip()
        if json_str.startswith("```"):
            json_str = json_str.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_str)
        if isinstance(data, dict):
            data = [data]
        if fallback_node is not None:
            for rec in data:
                if "node" not in rec or not str(rec.get("node", "")).strip():
                    rec["node"] = fallback_node
        return data
    except Exception as e:
        print(f"Failed to parse response for missing node: {e}")
        return []

def repair_missing_nodes(client, model_name, df_original, metadata_df, network_name, output_path, base_name, version, batch_size_for_missing=1, max_retries=3, budget=None):
    original_nodes = set(map(str, df_original.get("node", pd.Series(dtype=str)).astype(str)))
    combined_records = _read_combined(output_path, base_name, version)
    combined_nodes = set()
    for rec in combined_records:
        node_val = rec.get("node")
        if node_val is not None:
            combined_nodes.add(str(node_val))

    missing = sorted(list(original_nodes - combined_nodes))
    if not missing:
        return 0, (budget.get("used") if budget else 0), False

    print(f"Detected {len(missing)} missing node(s); querying individually to patch results…")

    added_records = []
    aborted = False
    for node_name in missing:
        if budget and isinstance(budget, dict) and budget.get("max", 0) > 0 and budget.get("used", 0) >= budget.get("max", 0):
            print(f"Reached max-requests limit ({budget.get('max')}). Stopping repairs to save credits.")
            aborted = True
            break
        row_df = df_original[df_original["node"].astype(str) == node_name]
        if row_df.empty:
            continue
        
        prompt = load_augmentation_prompt(row_df, 1, 1, metadata_df, network_name)
        
        result_obj = generate_with_openai(client, model_name, prompt, max_retries=max_retries)
        if budget is not None:
            budget["used"] = budget.get("used", 0) + 1
        if result_obj.get("status") != "ok" or not result_obj.get("content"):
            print(f"Skipping node due to API error or empty content: {node_name}")
            if budget and budget.get("fail_fast", False):
                print("Fail-fast enabled during repair: stopping now to avoid further credit usage.")
                aborted = True
                break
            continue
        records = _parse_response_records(result_obj.get("content", ""), fallback_node=node_name)
        if not records:
            print(f"Could not parse response for node: {node_name}")
            continue
        added_records.extend(records)

    if not added_records:
        return 0, (budget.get("used") if budget else 0), aborted

    combined_all = combined_records + added_records
    final_count = _save_combined(output_path, base_name, combined_all, version)
    return len(added_records), (budget.get("used") if budget else 0), aborted

def main():
    parser = argparse.ArgumentParser(description="Enrich food web nodes with OpenAI and save augmented traits")
    
    parser.add_argument(
        "--version",
        dest="version",
        default=DEFAULT_VERSION,
        help=f"Version identifier for output files (default: {DEFAULT_VERSION}). Example: v2_temperature_0.5"
    )
    
    parser.add_argument("--only", dest="only_base", help="Process only this base filename (without .scor)")
    parser.add_argument(
        "--data-path",
        dest="data_path",
        default=DATA_PATH,
        help=f"Path containing .scor files (default: {DATA_PATH})",
    )
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
        "--model",
        dest="model_name",
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--max-retries",
        dest="max_retries",
        type=int,
        default=3,
        help="Max retries on errors",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=10,
        help="Number of nodes to send per request (default: 10)",
    )
    parser.add_argument(
        "--fail-fast",
        dest="fail_fast",
        action="store_true",
        help="Stop immediately on the first API problem (default)",
    )
    parser.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        help="Do not stop on first problem; continue to next batch/file",
    )
    parser.set_defaults(fail_fast=False)
    parser.add_argument(
        "--max-requests",
        dest="max_requests",
        type=int,
        default=0,
        help="Maximum total API requests before stopping (0 = unlimited)",
    )
    args = parser.parse_args()

    version = args.version
    OUTPUT_PATH = os.path.join(CURRENT_DIR, "LLM features", "Chatgpt", version)
    
    print("\n" + "=" * 60)
    print(f"CHATGPT.PY - Version: {version}")
    print(f"Output directory: {OUTPUT_PATH}")
    print("=" * 60 + "\n")

    if not load_environment_variables():
        return
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    metadata_df = pd.read_excel(METADATA_PATH)
    client = setup_openai_client()
    if not client:
        return

    data_root = args.data_path or DATA_PATH
    if not os.path.isdir(data_root):
        print(f"data-path does not exist or is not a directory: {data_root}")
        return
    scor_files = sorted([f for f in os.listdir(data_root) if f.endswith(".scor")])

    if args.only_base:
        target = f"{args.only_base}.scor"
        if target in scor_files:
            scor_files = [target]
        else:
            print(f"Requested file not found in {data_root}: {target}")
            return

    if args.start_after and not args.only_base:
        marker = f"{args.start_after}.scor"
        if marker in scor_files:
            start_idx = scor_files.index(marker) + 1
            scor_files = scor_files[start_idx:]
        else:
            print(f"start-after marker not found; proceeding with full list: {marker}")
    
    if args.start_from and not args.only_base:
        marker = f"{args.start_from}.scor"
        if marker in scor_files:
            start_idx = scor_files.index(marker)
            scor_files = scor_files[start_idx:]
        else:
            print(f"start-from marker not found; proceeding with full list: {marker}")

    requests_used = 0

    for scor_file in scor_files:
        file_path = os.path.join(data_root, scor_file)
        df = process_foodweb_data(file_path)
        batches = create_batches(df, batch_size=args.batch_size)
        total_batches = len(batches)
        base_name = scor_file.replace(".scor", "")
        successful_batches = 0

        for batch_num, batch_df in enumerate(batches, 1):
            if args.max_requests > 0 and requests_used >= args.max_requests:
                print(f"Reached max-requests limit ({args.max_requests}). Stopping to save credits.")
                return
          
            prompt = load_augmentation_prompt(batch_df, batch_num, total_batches, metadata_df, base_name)
            
            local_retries = 1 if args.fail_fast else args.max_retries
            result_obj = generate_with_openai(client, args.model_name, prompt, max_retries=local_retries)
            requests_used += 1
            if result_obj.get("status") == "ok" and result_obj.get("content"):
                saved = save_batch_data(result_obj["content"], batch_df, OUTPUT_PATH, base_name, batch_num, version)
                if saved is not None:
                    successful_batches += 1
                    print(f"Batch {batch_num}/{total_batches} saved.")
            else:
                status = result_obj.get("status")
                print(f"Batch {batch_num}/{total_batches} for {scor_file} failed with status: {status}.")
                if args.fail_fast:
                    print("Fail-fast enabled: stopping now to avoid further credit usage.")
                    return
            
            time.sleep(2)

        if successful_batches > 0:
            total_combined = combine_batches(OUTPUT_PATH, base_name, version)
            budget = {"max": args.max_requests, "used": requests_used, "fail_fast": args.fail_fast}
            repaired, requests_used, aborted = repair_missing_nodes(
                client=client,
                model_name=args.model_name,
                df_original=df,
                metadata_df=metadata_df,
                network_name=base_name,
                output_path=OUTPUT_PATH,
                base_name=base_name,
                version=version,
                batch_size_for_missing=1,
                max_retries=(1 if args.fail_fast else args.max_retries),
                budget=budget,
            )
            suffix = f", repaired {repaired} missing" if repaired else ""
            print(f"{scor_file}: {successful_batches}/{total_batches} batches, {total_combined} total nodes{suffix}")
            if aborted:
                print("Repair step aborted due to fail-fast or budget limit. Stopping run.")
                return
        else:
            print(f"No successful batches for {scor_file}")

    print("\nProcessing complete!")
    print(f"Check outputs in: {OUTPUT_PATH}")
    
if __name__ == '__main__':
    main()