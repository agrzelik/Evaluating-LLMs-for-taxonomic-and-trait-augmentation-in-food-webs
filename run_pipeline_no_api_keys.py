#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from datetime import datetime

def run(cmd):
    """Run command and stop pipeline if it fails."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_name = f"logs/no_api_keys_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_name, "a")

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    return log_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("versions", nargs="*")

    args = parser.parse_args()

    run_baseline = args.baseline
    versions = args.versions if args.versions else ["gen1_0603"]

    log_file = setup_logging()

    print(f"Logging to: {log_file}")

    for version in versions:
        print("==========================================================")
        print(f"STARTING PROCESSING WITHOUT API KEYS FOR VERSION: {version}")
        print("==========================================================")

        print(f">>> Merging files: {version}")
        run(["python3", "Merge_llm_results.py", "--version", version])

        print(f">>> Merging taxonomies, version: {version}")
        run(["python3", "Merge_taxonomies.py", "--version", version])

        for nan_strategy in [1, 2]:
            print(f"Running taxonomies comparison with nan strategy value: {nan_strategy}")

            run([
                "python3", "Compare_taxonomies.py",
                "--version", version,
                "--nanstrategy", str(nan_strategy)
            ])

            run([
                "python3", "Viz_taxonomies_small_heatmaps.py",
                "--version", version,
                "--nanstrategy", str(nan_strategy)
            ])

            print(f"Taxonomies comparison done for value: {nan_strategy}")
            print("--------------------------")

        print(f">>> Finished individual tasks for {version}")
        print("----------------------------------------------------------")

    print("==========================================================")
    print("Multiple generations comparison")
    print("==========================================================")

    for nan_strategy in [1]:
        print(f"Running multiple generations comparison with nan strategy value: {nan_strategy}")

        run([
            "python3", "Intra_inter_model_comparison.py",
            "--versions", "gen1_0603", "gen2_0703", "gen3_0803",
            "--nanstrategy", str(nan_strategy),
            "--option_comparison", "inter"
        ])

        run([
            "python3", "Intra_inter_model_comparison.py",
            "--versions", "gen1_0603", "gen2_0703", "gen3_0803",
            "--nanstrategy", str(nan_strategy),
            "--option_comparison", "intra"
        ])

        run([
            "python3", "triangular_plots.py",
            "--versions", "gen1_0603", "gen2_0703", "gen3_0803",
            "--nanstrategy", str(nan_strategy)
        ])

        print(f"Multiple generations comparison done for value: {nan_strategy}")

    print("Running node-level prediction tasks")
    run(["python3", "node_level_prediction_original.py"])
    print("Finished node-level prediction tasks")

    print("Running taxonomy prunning")
    run(["python3", "separate_session_taxonomy.py"])
    run(["python3", "Compare_taxonomies_for_separate_session.py"])
    print("Finished taxonomy prunning")

    print("PIPELINE COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()