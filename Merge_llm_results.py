import os
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent

DEFAULT_VERSION = "test2"

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LLM results from all models.")
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"Version string used to locate processed files (default: {DEFAULT_VERSION})",
    )
    return parser.parse_args()

def get_paths(version: str):
    processed_files = {
        "claude": _HERE / f"LLM features/Processed/Claude/{version}/claude_Processed_Final.xlsx",
        "gemini":    _HERE / f"LLM features/Processed/Gemini/{version}/gemini_Processed_Final.xlsx",
        "chatgpt":    _HERE / f"LLM features/Processed/Chatgpt/{version}/chatgpt_Processed_Final.xlsx",
        "qwen":      _HERE / f"LLM features/Processed/Qwen/{version}/qwen_Processed_Final.xlsx",
    }
    output_path = _HERE / f"LLM features/Processed/all_models/{version}/ALL_MODELS_COMBINED.xlsx"
    return processed_files, output_path

SHARED_IDENTIFIER_COLUMNS = [
    "file_name",
    "node",
]

MODEL_ORDER = ["claude", "gemini", "chatgpt", "qwen"]

def load_model_data() -> Dict[str, pd.DataFrame]:
    """Load all processed files from each model."""
    data = {}
    for model, path in PROCESSED_FILES.items():
        if path.exists():
            print(f"Loading {model} data from: {path}")
            df = pd.read_excel(path)
            print(f"  - Loaded {len(df)} rows, {len(df.columns)} columns")

            data[model] = df
        else:
            print(f"Warning: File not found for {model}: {path}")
    return data

def merge_llm_results(model_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge data from all models into a single DataFrame with columns organized as:
    - file_name and node (once)
    - For each other column: model1, model2, model3, model4
    """
    if not model_data:
        raise ValueError("No model data loaded")
    
    print("STARTING MERGE PROCESS")
 
    first_model = MODEL_ORDER[0] if MODEL_ORDER[0] in model_data else list(model_data.keys())[0]
    result_df = model_data[first_model][SHARED_IDENTIFIER_COLUMNS].copy()
    
    print(f"\nBase dataframe from {first_model}: {len(result_df)} rows")
    
    all_columns = set()
    for model, df in model_data.items():
        all_columns.update([col for col in df.columns if col not in SHARED_IDENTIFIER_COLUMNS])
    
    for col in sorted(all_columns):
        
        for model in MODEL_ORDER:
            if model not in model_data:
                continue
                
            df = model_data[model]
            
            if col in df.columns:
                new_col_name = f"{col}_{model}"
                
                temp_df = df[SHARED_IDENTIFIER_COLUMNS + [col]].copy()
                temp_df.rename(columns={col: new_col_name}, inplace=True)
                
                result_df = result_df.merge(
                    temp_df,
                    on=SHARED_IDENTIFIER_COLUMNS,
                    how='outer',  
                    suffixes=('', '_dup'))
                
                non_null = result_df[new_col_name].notna().sum()
                print(f"  - {model}: {non_null} non-null values")
    
    result_df.fillna("NA", inplace=True)
    
    print(f"\n" + "="*60)
    print(f"Final merged dataframe: {len(result_df)} rows, {len(result_df.columns)} columns")
    print("="*60)
    
    return result_df

def save_combined_file(df: pd.DataFrame, output_path: Path) -> None:
    """Save the combined DataFrame to Excel."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Combined file saved to: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

if __name__ == "__main__":
    args = parse_args()
    version = args.version

    PROCESSED_FILES, OUTPUT_PATH = get_paths(version)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"MERGING LLM RESULTS FROM ALL MODELS  [version: {version}]")
    print("=" * 60)

    model_data = load_model_data()

    if not model_data:
        print("Error: No model data could be loaded. Exiting.")
        exit(1)

    combined_df = merge_llm_results(model_data)
    save_combined_file(combined_df, OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)