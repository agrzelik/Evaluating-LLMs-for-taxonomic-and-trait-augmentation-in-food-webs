#!/bin/bash
set -oe pipefail

LOG="logs/no_api_keys_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

RUN_BASELINE=false

VERSIONS=()

for arg in "$@"; do
    if [ "$arg" == "--baseline" ]; then
        RUN_BASELINE=true
    else
        VERSIONS+=("$arg")
    fi
done

if [ ${#VERSIONS[@]} -eq 0 ]; then
    VERSIONS=("gen1_0603")
fi

for VERSION in "${VERSIONS[@]}"
do
    echo "=========================================================="
    echo "STARTING PROCESSING WITHOUT API KEYS FOR VERSION: $VERSION"
    echo "=========================================================="

    echo ">>> Merging files: $VERSION"
    python3 Merge_llm_results.py --version "$VERSION"

    echo ">>> Merging taxonomies, version: $VERSION"
    python3 Merge_taxonomies.py --version "$VERSION" 

    for NAN_STRATEGY in 2
    do
        echo "Running taxonomies comparison with nan strategy value: $NAN_STRATEGY"
        python3 Compare_taxonomies.py --version "$VERSION" --nanstrategy $NAN_STRATEGY
        python3 Viz_taxonomies_comparisons.py --version "$VERSION" --nanstrategy $NAN_STRATEGY
        echo "Taxonomies comparison done for value: $NAN_STRATEGY"
        echo "--------------------------"
    done

    echo ">>> Finished individual tasks for $VERSION"
    echo "----------------------------------------------------------"
done

echo "=========================================================="
echo "Multiple generations comparison"
echo "=========================================================="

for NAN_STRATEGY in 2
do
    echo "Running multiple generations comparison with nan strategy value: $NAN_STRATEGY"
    python3 Intra_inter_model_comparison.py --versions gen1_0603 gen2_0703 gen3_0803 --nanstrategy $NAN_STRATEGY --option_comparison 'inter'
    python3 Intra_inter_model_comparison.py --versions gen1_0603 gen2_0703 gen3_0803 --nanstrategy $NAN_STRATEGY --option_comparison 'intra'
    
    python3 triangular_plots.py --versions gen1_0603 gen2_0703 gen3_0803 --nanstrategy $NAN_STRATEGY
    echo "Multiple generations comparison done for value: $NAN_STRATEGY"
done


echo "Running node-level prediction tasks"
python3 node_level_prediction_full.py
echo "Finished node-level prediction tasks"

echo "PIPELINE COMPLETED SUCCESSFULLY."