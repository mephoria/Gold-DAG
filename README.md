# Gold Return Causal Factor Model
> An end-to-end pipeline that uses **LLM-generated Directed Acyclic Graphs (DAGs)** and **Lasso regression** to identify the causal drivers of daily gold returns.

---

## Overview

This project replicates and extends the methodology from a 2023 academic paper on causal modelling with large language models. Rather than selecting features through correlation or arbitrary domain knowledge alone, we use GPT-4 to generate a causal graph of macroeconomic relationships, statistically prune it against real data, and then fit a regularised regression model using only the causally-validated features.

The result is a model that is both **interpretable** (every feature has an explicit causal justification) and **predictive** (69.3% directional accuracy vs. a 56% naive baseline).

---

## Repository Structure
├── src/
│ ├── data/
│ │ ├── gold_causal_data.csv # Processed feature dataset
│ │ ├── dag_pruning_results.csv # Edge-level statistical pruning results
│ │ ├── lasso_coefficients.csv # Lasso β coefficients
│ │ └── logistic_coefficients.csv # Logistic regression coefficients
│ └── results/
│ ├── gold_dag_clean.png # Final DAG visualisation
│ ├── chart_return_prediction.png # Actual vs predicted gold return
│ └── resultschart_lasso_coefficients.png
├── data_collection.py # Data pipeline (Yahoo Finance + FRED)
├── dag_generation.py # DAG construction, pruning, and Graphviz render
├── causal_model.py # Lasso + logistic regression, charts
├── llm_causal_edges.txt # Raw GPT-4 causal edge output
├── glossary_data.txt # Feature definitions fed to GPT-4
└── README.md


---

## Methodology

The pipeline runs in four sequential steps:

### Step 1 — Data Collection (`data_collection.py`)
- Downloads **11 market instruments** from Yahoo Finance (gold, DXY, EUR/USD, oil, copper, VIX, GVZ, S&P 500, silver, GLD, GDX) and **6 macro series** from FRED (10Y/2Y yields, real yield, 5Y breakeven inflation, Fed Funds rate, HY credit spread)
- Computes **log returns** for price series and **first differences** for rate/level series
- Derives a term spread feature (10Y − 2Y yield)
- Outputs a clean daily dataset saved to `src/data/gold_causal_data.csv`

### Step 2 — Feature Definitions (`glossary_data.txt`)
- Every feature is given a one-sentence plain-English definition
- These definitions are the sole input to GPT-4 — the paper's key requirement is that features have valid names and unambiguous definitions

### Step 3 — LLM DAG Generation + Pruning (`dag_generation.py`)
- GPT-4 generates **22 directed causal edges** across 18 features via a three-stage prompting process: *Causal Exploration → Causal Inference → Causal Validation*
- Each proposed edge is tested with **bivariate linear regression** (p < 0.05 threshold)
- Statistically weak edges are rendered as dashed lines in the DAG; strong edges are solid
- Final DAG is rendered with Graphviz and saved as a PNG

### Step 4 — Causal Factor Model (`causal_model.py`)
- **7 causally-validated parent features** of gold return are extracted from the pruned DAG
- **Lag-1 and Lag-2 features** are added for gold, GDX, silver, GLD, and GVZ (10 additional features), giving **17 total input features**
- Features are standardised with `StandardScaler`
- `LassoCV` with 5-fold time-series cross-validation selects the optimal regularisation α
- `LogisticRegressionCV` (L1 penalty, SAGA solver) predicts daily directional movement (up/down)

---

## Results

| Metric | Value |
|---|---|
| Lasso R² (in-sample) | **0.24** |
| Directional accuracy | **69.3%** |
| Naive baseline (always predict up) | 56.0% |
| Lift over baseline | **+13.3 percentage points** |
| Total features input | 17 |
| DAG edges proposed by GPT-4 | 22 |
| Edges kept after pruning (p < 0.05) | ~14 |
| Feature reduction to gold's parents | 18 → 7 causal parents (**61% reduction**) |

### Top Causal Drivers of Gold Return

| Feature | β | Direction | Interpretation |
|---|---|---|---|
| Gold Return (lag 1) | −0.0079 | Negative | **Mean reversion** — yesterday's gain predicts a reversal |
| GLD ETF Return (lag 1) | +0.0062 | Positive | GLD leads spot gold — ETF flow signal |
| DXY Dollar Return | −0.0041 | Negative | Stronger dollar suppresses gold (inverse relationship) |
| GDX Miners (lag 2) | +0.0036 | Positive | Miners lead gold by 2 days |
| Gold Return (lag 2) | −0.0031 | Negative | Secondary mean-reversion signal |
| GDX Miners (lag 1) | +0.0023 | Positive | Miners lead gold by 1 day |
| GVZ Volatility (lag 1) | −0.0011 | Negative | Rising gold vol predicts lower next-day return |
| Real 10Y Yield Δ | −0.0005 | Negative | Higher real yields reduce gold's appeal as zero-yield asset |

### Key Findings

1. **Mean reversion dominates**: The strongest single signal is yesterday's gold return with a negative coefficient — gold systematically reverts after large moves over a 1–3 day horizon.
2. **GDX miners lead spot gold**: The GDX ETF carries significant predictive power at both 1-day and 2-day lags, consistent with equity markets pricing in gold fundamentals before the spot commodity adjusts.
3. **DXY inverse relationship confirmed**: The dollar–gold inverse relationship is causally validated — DXY return is the strongest contemporaneous causal driver.
4. **Real yields matter, not nominal yields**: The 10Y real yield change survives pruning and retains a negative coefficient, confirming the standard macro framework. Nominal yield changes were weak and dropped.
5. **VIX is weak once GVZ is included**: VIX has a near-zero Lasso coefficient once gold-specific volatility (GVZ) is included — gold responds to its own implied vol, not general equity fear.

---

## Visualisations

| Chart | Description |
|---|---|
| `gold_dag_clean.png` | Full causal DAG — solid edges passed p < 0.05 pruning, dashed edges are weak |
| `chart_return_prediction.png` | Actual vs Lasso-predicted daily gold return (2023–2026) |
| `resultschart_lasso_coefficients.png` | Horizontal bar chart of all surviving Lasso β coefficients |

---

## Requirements

```bash
pip install yfinance pandas numpy scikit-learn scipy graphviz matplotlib
```

> **Note:** Graphviz also requires the system-level binary. Install via `brew install graphviz` (macOS) or `apt install graphviz` (Linux).

---

## Usage

Run the scripts in order:

```bash
python data_collection.py     # Build the dataset
python dag_generation.py      # Generate and prune the DAG
python causal_model.py        # Fit models and produce charts
```

---

## References

- *LLMs and Causal Modelling* (2023) — the methodology this project replicates
- FRED data: https://fred.stlouisfed.org
- Yahoo Finance data via `yfinance
