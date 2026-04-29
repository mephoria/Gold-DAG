Step 2 — Write your feature definitions
Before touching GPT, write a one-sentence plain-English definition for each of your 18 features. This is critical — the paper explicitly states the only hard requirement is that every feature has a valid name and definition. The quality of the DAG depends entirely on how clear these definitions are.

Step 3 — Generate the DAG with GPT-4
Feed your feature names + definitions into GPT using structured prompts (the prompt pack is in the guide I built you). The paper breaks this into three sub-steps: Causal Exploration (generate candidate edges), Causal Inference (orient the directions), and Causal Validation (check for mistakes).

Step 4 — Prune the DAG with data
For each edge the LLM proposes, run a quick regression to check if the relationship actually holds in your data. Edges that are statistically insignificant or have the wrong sign get removed.

Step 5 — Build the causal factor model
Fit a Lasso regression using only the DAG-selected features. This is your causal model of gold returns.

Step 6 — Layer 2 prediction model
On top of the causal model, build the directional signal — the fair value residual, regime score, and up/down classifier we discussed earlier.