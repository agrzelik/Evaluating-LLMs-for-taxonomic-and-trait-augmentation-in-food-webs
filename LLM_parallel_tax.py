import os
import subprocess
import sys
import time
import glob
import threading
import argparse

# Configuration
FILES_TO_PROCESS = 25       # Default if not specified via arguments
MAX_TERMINALS_GENERIC = 10  # Max terminals for ChatGPT, Claude, Qwen
MAX_TERMINALS_GEMINI = 25   # Max terminals for Gemini

# Per-model nodes-per-batch settings
NODES_PER_BATCH_CHATGPT = 5
NODES_PER_BATCH_CLAUDE = 5
NODES_PER_BATCH_QWEN = 5
NODES_PER_BATCH_GEMINI = 5

# ============================================================================
DEFAULT_VERSION = "v1_baseline"

SCRIPT_CHATGPT = "Chatgpt_tax.py"
SCRIPT_CLAUDE = "Claude.py"
SCRIPT_QWEN = "Qwen.py"
SCRIPT_GEMINI = "Gemini.py"

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "dataset_20260126_ecobase", "processed")

def get_scor_files(limit=None, specific_files=None):
    """Get list of .scor files from data directory."""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found: {DATA_DIR}")
        return []
    
    if specific_files:
        result = []
        for fname in specific_files:
            # Add .scor extension if not present
            if not fname.endswith('.scor'):
                fname = fname + '.scor'
            full_path = os.path.join(DATA_DIR, fname)
            if os.path.exists(full_path):
                result.append(full_path)
            else:
                print(f"Warning: File not found: {fname}")
        return result
    
    files = glob.glob(os.path.join(DATA_DIR, "*.scor"))
    files.sort()
    
    if limit:
        return files[:limit]
    return files

# ============================================================================
def process_manager(script_name, files, max_concurrent, nodes_per_batch, version_name):
    """
    Manages a pool of subprocesses for a specific script.
    Maintains 'max_concurrent' active processes until all files are processed.
    """
    print(f"[{script_name}] Starting manager. Files: {len(files)}, Max Concurrent: {max_concurrent}, Version: {version_name}")
    
    # Determine argument name for batch size
    if script_name in [SCRIPT_CHATGPT, SCRIPT_CLAUDE]:
        batch_arg = "--batch-size"
    else:
        batch_arg = "--nodes-per-batch"
        
    script_path = os.path.join(CURRENT_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[{script_name}] Error: Script not found: {script_path}")
        return

    active_processes = [] # List of tuples: (subprocess.Popen, filename)
    files_to_launch = list(files) # Copy to pop from
    
    # Loop until no files left to launch AND no processes running
    while files_to_launch or active_processes:
        # 1. Check for finished processes
        # Iterate over a copy of the list so we can remove items safely
        for p, fname in active_processes[:]:
            if p.poll() is not None: # Process finished
                print(f"[{script_name}] Finished: {fname}")
                active_processes.remove((p, fname))
        
        # 2. Launch new processes if we have capacity and files remaining
        while files_to_launch and len(active_processes) < max_concurrent:
            file_path = files_to_launch.pop(0)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            cmd = [
                sys.executable,
                script_path,
                "--only", base_name,
                batch_arg, str(nodes_per_batch),
                "--version", version_name,  
            ]
            
            print(f"[{script_name}] Launching: {base_name} (version: {version_name})")
            
            if sys.platform == "win32":
                # CREATE_NEW_CONSOLE opens a new window for each process
                p = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                p = subprocess.Popen(cmd)
                
            active_processes.append((p, base_name))
            
        # Sleep briefly to avoid busy waiting
        time.sleep(2)
        
    print(f"[{script_name}] All tasks completed.")

def main():
    parser = argparse.ArgumentParser(description="Run LLM scripts in parallel.")
    parser.add_argument("--files", type=int, default=FILES_TO_PROCESS, help="Number of files to process (default: 25)")
    parser.add_argument("--specific", nargs="+", help="Specific file names to process (without .scor)")
    parser.add_argument("--models", nargs="+", default=["all"], choices=["chatgpt", "claude", "qwen", "gemini", "all"], help="Models to run (default: all)")
    
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Version identifier for this run (default: {DEFAULT_VERSION}). Example: v2_temperature_0.5"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LLM PARALLEL RUNNER")
    print("=" * 70)
    print(f"Version: {args.version}")
    print(f"Models: {args.models}")
    print(f"Files to process: {args.files if not args.specific else 'specific files'}")
    print("=" * 70 + "\n")
    
    # Get files
    if args.specific:
        files = get_scor_files(specific_files=args.specific)
    else:
        limit = args.files if args.files and args.files > 0 else None
        files = get_scor_files(limit=limit)
        
    if not files:
        print("No files found to process.")
        return

    print(f"Processing {len(files)} files.\n")
    
    models_to_run = args.models
    if "all" in models_to_run:
        models_to_run = ["chatgpt", "claude", "qwen", "gemini"]

    # Create threads for each LLM to run them simultaneously
    threads = []
    
    # 1. ChatGPT
    if "chatgpt" in models_to_run:
        # Limit concurrent terminals to min(files, MAX_TERMINALS_GENERIC)
        limit_chatgpt = min(len(files), MAX_TERMINALS_GENERIC)
        t1 = threading.Thread(
            target=process_manager, 
            args=(SCRIPT_CHATGPT, files, limit_chatgpt, NODES_PER_BATCH_CHATGPT, args.version)  
        )
        threads.append(t1)
    
    # 2. Claude
    if "claude" in models_to_run:
        limit_claude = min(len(files), MAX_TERMINALS_GENERIC)
        t2 = threading.Thread(
            target=process_manager, 
            args=(SCRIPT_CLAUDE, files, limit_claude, NODES_PER_BATCH_CLAUDE, args.version) 
        )
        threads.append(t2)
    
    # 3. Qwen
    if "qwen" in models_to_run:
        limit_qwen = min(len(files), MAX_TERMINALS_GENERIC)
        t3 = threading.Thread(
            target=process_manager, 
            args=(SCRIPT_QWEN, files, limit_qwen, NODES_PER_BATCH_QWEN, args.version)  
        )
        threads.append(t3)
    
    # 4. Gemini
    if "gemini" in models_to_run:
        limit_gemini = min(len(files), MAX_TERMINALS_GEMINI)
        t4 = threading.Thread(
            target=process_manager, 
            args=(SCRIPT_GEMINI, files, limit_gemini, NODES_PER_BATCH_GEMINI, args.version) 
        )
        threads.append(t4)
    
    print(f"Launching managers for: {', '.join(models_to_run)}...")
    for t in threads:
        t.start()
        
    # Wait for all threads to complete
    for t in threads:
        t.join()
        
    print("\n" + "=" * 70)
    print("All LLM runs finished!")
    print(f"Output saved to: LLM features/[Model]/{args.version}/")
    print("=" * 70)

if __name__ == "__main__":
    main()