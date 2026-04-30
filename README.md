# Gold-DAG

Gold-DAG is a Python project built to explore causal drivers of daily gold returns using a directed acyclic graph (DAG), financial market data, and predictive models.

The project collects macroeconomic and market variables related to gold to construct a causal graph of potential relationships and then uses the selected features to model gold returns and gold price direction.

## Project Objective

The goal of this project is to identify and visualize potential causal drivers of gold price movements, especially for GLD and gold-related assets.

The project combines:

- Financial data collection
- Feature engineering
- DAG-based causal structure design
- Lasso regression for gold return prediction
- Logistic regression for gold up/down direction prediction
- Visualization of causal relationships and model results

## Repository Structure

```text
Gold-DAG/
└── src/
    ├── data/
    │   ├── gold_causal_data.csv
    │   ├── dag_pruning_results.csv
    │   ├── lasso_coefficients.csv
    │   └── logistic_coefficients.csv
    │
    ├── notes/
    │   ├── glossary_data.txt
    │   ├── llm_causal_edges.txt
    │   └── steps.md
    │
    ├── results/
    │   ├── gold_dag_clean.png
    │   ├── chart_return_prediction.png
    │   └── resultschart_lasso_coefficients.png
    │
    └── scripts/
        ├── data_collection.py
        ├── dag_generation.py
        └── causal_model.py
