# Evaluating LLMs for taxonomic and trait augmentation in food webs


# LLM Data Pipeline: Data Augmentation

This repository provides a complete, automated workflow for generating, augmenting, preprocessing, and analyzing ecological datasets produced by Large Language Models (LLMs). The pipeline is designed to transform raw food web data into enriched, ML-ready datasets through a multi-stage process.

Core Pipeline Stages:

Parallel Generation & Processing: querying of multiple LLMs (GPT, Claude, Gemini, Qwen) followed by JSON-to-Excel conversion and data standardization.

Taxonomic Verification: Integration with the WoRMS API to validate LLM-generated Latin names and retrieve authoritative taxonomic lineages.

Ensemble Optimization: Implementation of advanced aggregation strategies—including self-evaluated soft voting, cross-model evaluated soft voting, and LLM-as-a-judge to mitigate hallucinations and improve data robustness.

Comparative Analysis: Systematic evaluation of model performance across different generations, utilizing intra- and inter-model comparisons.

Downstream Task: Execution of node-level prediction tasks using 43 ecological functional traits.

## Quick Start

If you want to skip the documentation and run the code immediately, you can use these shortcuts. Make sure you go through Installation & Environment Setup before.

### Option A: Full Pipeline (Requires API Keys)
Run this to generate new data from scratch using LLMs. 
```bash
./run_pipeline.sh gen1_0603
```
### Option B: Analysis Only (No API Keys needed)
Run this to use files provided in the LLM `features/Processed`. This skips the expensive generation step and starts directly with analysis.
```bash
./run_pipeline_no_api_keys.sh gen1_0603
```
This scripts are the recommended way to process all LLM outputs end-to-end.

---

## 1. Installation & Environment Setup

### A. Install Dependencies

A `requirements.txt` file has been added. Install all necessary packages using:

```bash
pip install -r requirements.txt
```

### B. Environment Variables

Create a `.env` file in the root directory and add your API keys:

```ini
# .env file configuration
OPENAI_API_KEY=sk-...        # OpenAI (ChatGPT)
ANTHROPIC_API_KEY=sk-ant-... # Anthropic (Claude)
OPENROUTER_API_KEY=sk-or-... # OpenRouter (Qwen / Open-Source Models)
GEMINI_API_KEY=AIza...       # Google Gemini
```

---

## 2. Step 1: LLM-Based Data Generation (Data Augmentation)

This project focuses on **Data Augmentation** using multiple LLMs for both **Argumentation** and **Respiration** domains.

### A. Individual LLM Scripts

You may run generation for each provider manually (defaults to internal settings):

* `python Chatgpt.py`
* `python Claude.py`
* `python Qwen.py`
* `python Gemini.py`

### B. Parallel Generation (Recommended)

Use **`LLM_parallel.py`** to run LLMs efficiently with batch control. You can specify which models to run and how many files to process using command-line arguments.

#### Arguments:

* `--models`: Specify the model(s) to run. Options: `chatgpt`, `claude`, `qwen`, `gemini`, or `all`.
* `--files`: Specify the number of files to process. Use `0` to process **ALL** available files.
* `--specific`: Instead of the number of files, you can pass the network names (without **.scor**) to process specific networks.

> NOTE: The pipeline also includes several internal configuration constants (defaults shown below). You can edit these constants in `LLM_parallel.py` or override via the command line (if implemented) to control concurrency, batching, and load.

### Internal configuration constants (defaults)

Place these near the top of `LLM_parallel.py` so they're easy to find and change:

```python
# How many files to process by default (used when --files is not provided)
FILES_TO_PROCESS = 2  # Default: number of files to process per model (use 0 for ALL)

# Maximum concurrent "terminals" / workers used for generic LLMs (ChatGPT, Claude, Qwen)
MAX_TERMINALS_GENERIC = 50

# Maximum concurrent terminals for Gemini (Gemini may have stricter limits)
MAX_TERMINALS_GEMINI = 2

# How many nodes (prompt calls / tasks) to batch together for mass runs
NODES_PER_BATCH_MASS = 5

# Nodes per batch for Gemini specifically (tuned separately)
NODES_PER_BATCH_GEMINI = 6
```

#### Quick guidance on changing these values

* `FILES_TO_PROCESS`: Set to `0` to process all files in your input folder. Otherwise set a positive integer to limit the number of files processed per run.
* `MAX_TERMINALS_GENERIC` / `MAX_TERMINALS_GEMINI`: Control concurrency. Increase for higher throughput only if your API quotas, rate limits, and machine resources permit it. Lower them to avoid throttling.
* `NODES_PER_BATCH_MASS` / `NODES_PER_BATCH_GEMINI`: Control prompt batching. Larger batches reduce overhead but increase memory/latency for each batch.

> Tip: if you expect to change these frequently, consider loading them from environment variables or a small `config.yaml` to avoid editing code repeatedly.

---

## Usage Examples

**1. Run for a specific model (e.g., ChatGPT)**

For a specific number of files (e.g., 5 files):

```bash
python LLM_parallel.py --models chatgpt --files 5
```

For **ALL** files:

```bash
python LLM_parallel.py --models chatgpt --files 0
```

**2. Run for other specific models**
Replace `chatgpt` with `claude`, `qwen`, or `gemini`.

Example: Claude for 10 files

```bash
python LLM_parallel.py --models claude --files 10
```

Example: Gemini for all files

```bash
python LLM_parallel.py --models gemini --files 0
```

**3. Run for ALL models at once**

For a specific number of files (e.g., 2 files per model):

```bash
python LLM_parallel.py --models all --files 2
```

For **ALL** files across **ALL** models:

```bash
python LLM_parallel.py --models all --files 0
```

**4. Run for a specific network using gemini**

Example: Gemini for Aleutian_Islands_252 file:

```bash
python LLM_parallel.py --models gemini --specific Aleutian_Islands_252
```

---

## 3. Step 2: JSON Parsing & Formatting (JSON -> Excel)

Use **`Jsonparserforllm.py`** to convert raw JSON outputs into structured Excel datasets.

### Features

* Scans the `LLM features` directory
* Extracts files containing `_combined_augmented`
* Normalizes filenames
* Audits column names across different LLMs
* Produces clean Excel files

### Commands

- Run for **all providers** (OpenAI, Qwen, Gemini, Anthropic):

```bash
python Jsonparserforllm.py --provider all
```

- Run for a **single provider** only, using its name:

```bash
python Jsonparserforllm.py --provider gemini
python Jsonparserforllm.py --provider openai
python Jsonparserforllm.py --provider qwen
python Jsonparserforllm.py --provider anthropic
```

If you omit the flag, it defaults to processing all providers:

```bash
python Jsonparserforllm.py
```

> **Note:** Run this at least once to inspect column names, then create/update your **value-mapping dictionary** for standardization using manual inspection or AI assistance.

---

## 4. Step 3: Preprocessing & Merging

Use **preprocessing.py** to merge outputs from Step 2 into unified datasets.

### Process

1. Loads all Excel files
2. Cleans inconsistent values
3. Merges into one master dataset per LLM

### Command

```bash
python Preprocessing.py
```

---
If you don't dispose of API keys to generate the responses, you can start processing from step 5. using data provided in `LLM features/Processed`. Use dedicated bash script `run_pipeline_no_api_keys.sh` to get the results without API keys. 
___
## 5. Merging Results Across Networks

After preprocessing individual model files, results for all processed networks and models can be merged into consolidated datasets.

Use **`Merge_llm_results.py`** to combine outputs across models for a given pipeline version.

### Command

```bash
python Merge_llm_results.py --version v1
```
---
## 6. Baseline Processing (Optional)

To evaluate LLM performance against human annotated baseline, the pipeline optionally processes baseline file to download taxonomic data from a domain database.
This step can be enabled or disabled in the pipeline script.

```bash
python Process_baseline.py
```
---
## 7. Qurying domain database (WoRMS)

This step queries the World Register of Marine Species (WoRMS) with LLM-generated Latin species names to retrieve taxonomy information. 

```bash
python Query_DB_with_LLM_produced_names.py --version v1
```
---

## 8. Ensemble Optimization of LLM Predictions

The pipeline includes several ensemble strategies to combine results from multiple LLMs. 
These methods aim to improve accuracy and robustness by aggregating model outputs.

# 8.1 Self-Evaluated Soft Voting

Each LLM evaluates the confidence of its own predictions.
Predictions are then combined using soft voting weighted by self-reported confidence.

```bash
python LLM_optimization_self_evaluated_soft_voting.py --version v1
```

# 8.2 Cross-Model Evaluated Soft Voting

Each model evaluates predictions produced by other models, the scores given by model are used to obtain final taxonomic values via soft voting.

```bash
python LLM_optimization_cross_evaluated_soft_voting.py --version v1
```
# 8.3 Black-Box Ensemble Approach

This strategy treats LLM as black-box judge, which aggregates models' answers using heuristic or statistical strategies internally (without relying on confidence scores).

```bash
python LLM_optimization_black_box.py --version v1
```
---
## 9. Taxonomy Integration & Comparison

Taxonomies generated by different models or ensemble strategies are merged and compared with baseline values. All of the variants are shown on a heatmap.

```bash
python Merge_taxonomies.py --version v1
python Compare_taxonomies.py --version v1
python Viz_taxonomies_comparisons.py --version v1
```
---

# 10. Model self-consistency and cross-model accuracy

This section evaluates the reliability and stability of LLM outputs across multiple generations and different models. It focuses on identifying level of accuracy across a few generations of the taxonomic data.

```bash
python3 Intra_and_inter_model_viz.py --versions gen1_0603 gen2_0703 gen3_0803
python3 triangular_plots.py
```
# 11. Node-level prediction tasks

This section represents a standalone analytical task that utilizes the previously enriched dataset: `Input/functional_features_0803/Gemini_Processed_Final.xlsx`. The goal is to evaluate the practical utility of LLM-augmented data in predicting ecological role of at individual node level. Functional features used in this task, were generated using attatched prompt: `Input/functional_features_0803/functional_features_prompt.txt` 

```bash
python3 node_level_prediction_full.py
```

---

## 8. Directory Structure

```text
├── .env
├── requirements.txt               # Project dependencies
│
├── foodwebviz.py                  # Reads .scor files for inputs
├── LLM_parallel.py                # Parallel data generation controller
│
├── Chatgpt.py                     # OpenAI (ChatGPT) data generation script
├── Claude.py                      # Anthropic Claude generation script
├── Qwen.py                        # Qwen / OpenRouter generation script
├── Gemini.py                      # Google Gemini generation script
│
├── Jsonparserforllm.py            # Converts raw LLM JSON outputs into structured Excel datasets
├── Preprocessing.py               # Cleans, normalizes, and standardizes parsed LLM outputs
├── Merge_llm_results.py           # Merges processed outputs across models 
│
├── Process_baseline.py            # Processes reference/baseline for comparison with LLM outputs
│
├── LLM_optimization_self_evaluated_soft_voting.py   # Ensemble method using LLM self-reported confidence scores
├── LLM_optimization_cross_evaluated_soft_voting.py  # Ensemble method where models evaluate predictions of other models
├── LLM_optimization_black_box.py                    # Ensemble aggregation treating model outputs as black-box predictions
│
├── Query_DB_with_LLM_produced_names.py              # Queries WoRMS with Latin species names generated by LLMs
│
├── Merge_taxonomies.py              # Combines taxonomy outputs from different models and ensemble methods
├── Compare_taxonomies.py            # Computes agreement metrics between predicted and reference taxonomies
├── Viz_taxonomies_comparisons.py    # Generates visualizations for taxonomy comparison results
│
├── Intra_model_comparison.py        # Evaluates consistency and variability within predictions of the same model
├── aggregation_stats copy.py        # Computes summary statistics and aggregated performance metrics
│
├── run_Preprocessing_pipeline.py    # Executes the full preprocessing and analysis workflow
│
├── LLM_worms_features/
├── LLM features/
│    ├── Original data augmentation/ # Output of JSON parser
│    ├── Processed/                  # Merged outputs
│    ├── Qwen/                       # Raw JSON
│    ├── Openai/                     # Raw JSON
│    ├── Gemini/                     # Raw JSON
│    └── Anthropic/                  # Raw JSON
├── Versions_Divergence/
├── Ensembles/
├── dataset_20260126_ecobase/
├── Comparisons/
├── baseline/


```



