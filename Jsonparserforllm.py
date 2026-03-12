import os
import json
import pandas as pd
from pathlib import Path
import argparse

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent
FEATURES_DIR = (BASE_DIR / "LLM features").resolve()
OUTPUT_BASE = (FEATURES_DIR / "Original data augmentation").resolve()
DEFAULT_VERSION = "v1_baseline"

def get_providers(version: str):
    """
    Returns the configuration of providers, taking the version into account.
    """
    
    return [
        {
            "name": "qwen",
            "input": (FEATURES_DIR / "Qwen" / version),  
            "output": (OUTPUT_BASE / "Qwen" / version), 
            "prefix": f"qwen",  
        },
        {
            "name": "chatgpt",
            "input": (FEATURES_DIR / "Chatgpt" / version),
            "output": (OUTPUT_BASE / "Chatgpt" / version),
            "prefix": f"chatgpt",
        },
        {
            "name": "gemini",
            "input": (FEATURES_DIR / "Gemini" / version),
            "output": (OUTPUT_BASE / "Gemini" / version),
            "prefix": f"gemini",
        },
        {
            "name": "claude",
            "input": (FEATURES_DIR / "Claude" / version),
            "output": (OUTPUT_BASE / "Claude" / version),
            "prefix": f"claude",
        },
    ]


def extract_clean_name(filename: str) -> str:
    """
    Cuts the filename to keep only the part BEFORE '_combined_augmented'.
    
    Example Input:  "Alaska_Prince_William_Sound_2_combined_augmented_v6.json"
    Example Output: "Alaska_Prince_William_Sound_2"
    """
    # 1. Check for Gemini specific pattern (longest match first)
    if "_orig_combined_augmented" in filename:
        clean_part = filename.split("_orig_combined_augmented")[0]
    
    # 2. Check for standard pattern (used by OpenAI, Qwen, Anthropic)
    elif "_combined_augmented" in filename:
        clean_part = filename.split("_combined_augmented")[0]
    
    # 3. Fallback: If pattern not found, use the original name (should not happen)
    else:
        clean_part = Path(filename).stem

    # Remove any trailing underscores (e.g. "Alaska_" -> "Alaska")
    return clean_part.rstrip("_")


def process_provider(provider: dict) -> set:
    """
    Reads JSON files from the provider's input folder, cleans the names,
    and saves them as Excel files in the output folder.
    """
    input_folder = provider["input"]
    output_folder = provider["output"]
    prefix = provider["prefix"]
    
    print("-" * 50)
    print(f"Processing: {provider['name']}")

    # 1. Validation
    if not input_folder.exists():
        print(f"Error: Input folder missing: {input_folder}")
        return set()

    output_folder.mkdir(parents=True, exist_ok=True)

    # 2. Find Files
    try:
        all_files = os.listdir(input_folder)
        # Only select files that have the augmented tag and are JSON
        target_files = [f for f in all_files if "_combined_augmented" in f and f.lower().endswith(".json")]
    except Exception as e:
        print(f"Error reading folder: {e}")
        return set()

    if not target_files:
        print(f"No matching files found in {input_folder}")
        return set()

    unique_columns = set()

    # 3. Process Files
    for filename in target_files:
        try:
            # Load Data
            file_path = input_folder / filename
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            unique_columns.update(df.columns)

            # Clean Name logic
            base_name = extract_clean_name(filename)
            
            # Create Final Filename (e.g. "Alaska v1_baseline qwen.xlsx")
            final_name = f"{base_name} {prefix}.xlsx"
            save_path = output_folder / final_name

            # Save
            df.to_excel(save_path, index=False)
            print(f"Saved: {final_name}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    return unique_columns


def main():
    parser = argparse.ArgumentParser(description="Parse LLM augmented JSON into Excel files")
    
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Version name to process (default: {DEFAULT_VERSION}). Example: v2_temperature_05"
    )
    
    parser.add_argument(
        "--provider",
        choices=["qwen", "chatgpt", "gemini", "claude", "all"],
        default="all",
        help="Which provider to process (default: all)",
    )
    
    args = parser.parse_args()

    PROVIDERS = get_providers(args.version)
    provider_names = [p["name"] for p in PROVIDERS]
    
    print("\n" + "=" * 60)
    print(f"JSON PARSER - Version: {args.version}")
    print("=" * 60 + "\n")

    global_columns = set()

    if args.provider == "all":
        selected_providers = PROVIDERS
    else:
        selected_providers = [p for p in PROVIDERS if p["name"] == args.provider]

    # Run the process for the selected provider(s)
    for provider in selected_providers:
        cols = process_provider(provider)
        global_columns.update(cols)

    # Summary
    print("\n" + "=" * 50)
    print("ALL UNIQUE COLUMNS FOUND")
    print("=" * 50)
    print(sorted(list(global_columns)))


if __name__ == "__main__":
    main()