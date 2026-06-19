#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from datetime import datetime


NETWORKS = [
    "Aleutian_Islands_252",
    "Australia_North_West_Shelf_405",
    "Bamboung_180",
    "Bay_of_Biscay_335",
    "British_Columbia_Coast_478",
    "Celestun_247",
    "Central_Gulf_of_California_239",
    "Denmark_Faroe_Islands_46",
    "Eastern_Bering_Sea_183",
    "Galapagos_Floreana_Rocky_Reef_48",
    "Grand_Banks_of_Newfoundland_105",
    "Gulf_of_California_450",
    "Hudson_Bay_446",
    "Jalisco_and_Colima_Coast_307",
    "Jurien_Bay_456",
    "Mauritania_650",
    "North_Aegean_495",
    "North_Benguela_503",
    "Northern_Humboldt_Current_488",
    "Ria_Formosa_125",
    "Sierra_Leone_137",
    "Strait_of_Georgia_477",
    "Terminos_Lagoon_243",
    "Portofino_731",
    "Mount_St_Michel_Bay_742",
    "Aegean_Sea_760",
    "Mississippi_River_Delta_779"
]


def run(cmd):
    """Run command and stop pipeline if it fails."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    log_name = f"logs/tax_level_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    versions = args.versions if args.versions else ["tax_level_1606_gen"]

    log_file = setup_logging()
    print(f"Logging to: {log_file}")

    for version in versions:
        print("==========================================================")
        print(f"STARTING PROCESSING FOR VERSION: {version}")
        print("==========================================================")

        for net in NETWORKS:
            print(f">>> Processing network: {net} for version: {version}")

            run([
                "python3", "LLM_parallel_tax.py",
                "--models", "chatgpt",
                "--specific", net,
                "--version", version
            ])

            print(f">>> Network processed: {net} for version: {version}")

        print(f">>> Parsing version: {version}")
        run([
            "python3", "Jsonparserforllm.py",
            "--provider", "chatgpt",
            "--version", version
        ])

        print(f">>> Processing version: {version}")
        run([
            "python3", "Preprocessing_tax_level.py",
            "--provider", "chatgpt",
            "--version", version,
            "--skip-json-parser"
        ])

        print(f">>> Merging files: {version}")
        run([
            "python3", "Merge_llm_results.py",
            "--version", version
        ])
'''
        if run_baseline:
            print(">>> Processing baseline (ENABLED)")
            run(["python3", "Process_baseline.py"])
        else:
            print(">>> Skipping baseline (DISABLED)")

        print(f">>> Querying WoRMS with LLM-generated Latin name, version: {version}")
        run([
            "python3", "Query_DB_with_LLM_produced_names.py",
            "--version", version
        ])

        print(f">>> Applying self-evaluated soft voting, version: {version}")
        run([
            "python3",
            "LLM_optimization_self_evaluated_soft_voting.py",
            "--version", version
        ])

        print(f">>> Applying cross-evaluated soft voting, version: {version}")
        run([
            "python3",
            "LLM_optimization_cross_evaluated_soft_voting.py",
            "--version", version
        ])

        print(f">>> Applying black box approach, version: {version}")
        run([
            "python3",
            "LLM_optimization_black_box.py",
            "--version", version
        ])

        print(f">>> Comparing taxonomies, version: {version}")

        for nan_strategy in [1, 2]:
            print(f"Running taxonomies comparison with nan strategy value: {nan_strategy}")

            run([
                "python3", "Compare_taxonomies.py",
                "--version", version,
                "--nanstrategy", str(nan_strategy)
            ])

            run([
                "python3", "Viz_taxonomies_comparisons.py",
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

    for nan_strategy in [1, 2]:
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
    run(["python3", "node_level_prediction_full.py"])
    print("Finished node-level prediction tasks")

    print("PIPELINE COMPLETED SUCCESSFULLY.")

'''
if __name__ == "__main__":
    main()