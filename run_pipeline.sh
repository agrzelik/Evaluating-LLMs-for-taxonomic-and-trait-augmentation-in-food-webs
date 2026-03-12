#!/bin/bash
set -oe pipefail

LOG="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

NETWORKS=(
    "Aleutian_Islands_252"
    "Australia_North_West_Shelf_405"
    "Bamboung_180"
    "Bay_of_Biscay_335"
    "British_Columbia_Coast_478"
    "Celestun_247"
    "Central_Gulf_of_California_239"
    "Denmark_Faroe_Islands_46"
    "Eastern_Bering_Sea_183"
    "Galapagos_Floreana_Rocky_Reef_48"
    "Grand_Banks_of_Newfoundland_105"
    "Gulf_of_California_450"
    "Hudson_Bay_446"
    "Jalisco_and_Colima_Coast_307"
    "Jurien_Bay_456"
    "Mauritania_650"
    "North_Aegean_495"
    "North_Benguela_503"
    "Northern_Humboldt_Current_488"
    "Ria_Formosa_125"
    "Sierra_Leone_137"
    "Strait_of_Georgia_477"
    "Terminos_Lagoon_243"
    "Portofino_731"
    "Mount_St_Michel_Bay_742"
    "Aegean_Sea_760"
    "Mississippi_River_Delta_779")

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
    echo "STARTING PROCESSING FOR VERSION: $VERSION"
    echo "=========================================================="

    for NET in "${NETWORKS[@]}"
    do
        echo ">>> Processing network: $NET for version: $VERSION"
        # 1. Parallel data generation
        python3 LLM_parallel.py --models all --specific "$NET" --version "$VERSION"
        echo ">>> Network processed: $NET for version: $VERSION"
    done

    # 2. Parse JSON to Excel
    echo ">>> Parsing version: $VERSION"
    python3 Jsonparserforllm.py --provider all  --version "$VERSION"

    # 3. Preprocessing 
    echo ">>> Processing version: $VERSION"
    python3 Preprocessing.py --provider all --version "$VERSION" --skip-json-parser

    # 4. Merge version files
    echo ">>> Merging files: $VERSION"
    python3 Merge_llm_results.py --version "$VERSION"

    # 5. Process baseline
    if [ "$RUN_BASELINE" = true ]; then
        echo ">>> Processing baseline (ENABLED)"
        python3 Process_baseline.py
    else
        echo ">>> Skipping baseline (DISABLED)"
    fi

    # 6. Query WoRMS with LLM-genrated Latin names
    echo ">>> Querying WoRMS with LLM-genrated Latin name, version: $VERSION"
    python3 Query_DB_with_LLM_produced_names.py --version "$VERSION"

    # 7. Ensembles: self-evaluated soft voting
    echo ">>> Applying self-evaluated soft voting, version: $VERSION"
    python3 LLM_optimization_self_evaluated_soft_voting.py --version "$VERSION"

    # 8. Ensembles: cross-model evaluated soft voting 
    echo ">>> Applying cross-evaluated soft voting, version: $VERSION"
    python3 LLM_optimization_cross_evaluated_soft_voting.py --version "$VERSION"

    # 9. Ensembles: black box approach
    echo ">>> Applying black box approach, version: $VERSION"
    python3 LLM_optimization_black_box.py --version "$VERSION"

    # 10. Compare taxonomies
    echo ">>> Comparing taxonomies, version: $VERSION"
    for NAN_STRATEGY in 1 2 
    do
        echo "Running taxonomies comparison with nan strategy value: $NAN_STRATEGY"
        python3 Compare_taxonomies.py --version "$VERSION" --nanstrategy "$NAN_STRATEGY"
        python3 Viz_taxonomies_comparisons.py --version "$VERSION" --nanstrategy "$NAN_STRATEGY"
        echo "Taxonomies comparison done for value: $NAN_STRATEGY"
        echo "--------------------------"
    done

    echo ">>> Finished individual tasks for $VERSION"
    echo "----------------------------------------------------------"
done

echo "=========================================================="
echo "Multiple generations comparison"
echo "=========================================================="

for NAN_STRATEGY in 1 2 
do
    echo "Running multiple generations comparison with nan strategy value: $NAN_STRATEGY"
    python3 Intra_inter_model_comparison.py --versions gen1_0603 gen2_0703 gen3_0803 --nanstrategy "$NAN_STRATEGY" --option_comparison 'inter'
    python3 Intra_inter_model_comparison.py --versions gen1_0603 gen2_0703 gen3_0803 --nanstrategy "$NAN_STRATEGY" --option_comparison 'intra'
    
    python3 triangular_plots.py --versions gen1_0603 gen2_0703 gen3_0803 --nanstrategy "$NAN_STRATEGY"
    echo "Multiple generations comparison done for value: $NAN_STRATEGY"
done


echo "Running node-level prediction tasks"
python3 node_level_prediction_full.py
echo "Finished node-level prediction tasks"

echo "PIPELINE COMPLETED SUCCESSFULLY."