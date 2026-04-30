import pandas as pd
from scipy import stats
import graphviz

# ── 1. Define edges ───────────────────────────────────────────
edges = [
    ("fed_funds_change",        "yield_2y_change"),
    ("fed_funds_change",        "yield_10y_change"),
    ("fed_funds_change",        "real_yield_10y_change"),
    ("real_yield_10y_change",   "yield_10y_change"),
    ("breakeven_5y_change",     "yield_10y_change"),
    ("yield_10y_change",        "term_spread_chg"),
    ("yield_2y_change",         "term_spread_chg"),
    ("yield_10y_change",        "dxy_return"),
    ("eurusd_return",           "dxy_return"),
    ("oil_return",              "breakeven_5y_change"),
    ("copper_return",           "hy_spread_change"),
    ("real_yield_10y_change",   "gold_return"),
    ("breakeven_5y_change",     "gold_return"),
    ("dxy_return",              "gold_return"),
    ("oil_return",              "gold_return"),
    ("vix_chg",                 "gold_return"),
    ("hy_spread_change",        "gold_return"),
    ("sp500_return",            "gold_return"),
    ("gold_return",             "gvz_chg"),
    ("gold_return",             "gld_return"),
    ("gold_return",             "gdx_return"),
    ("gold_return",             "silver_return"),
]

# ── 2. Load data ──────────────────────────────────────────────
df = pd.read_csv("src/data/gold_causal_data.csv", index_col=0, parse_dates=True)
print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── 3. Statistical pruning ────────────────────────────────────
results = []
for cause, effect in edges:
    if cause not in df.columns or effect not in df.columns:
        results.append((cause, effect, None, None, None, "MISSING COLUMN"))
        continue
    tmp = df[[cause, effect]].dropna()
    if len(tmp) < 50:
        results.append((cause, effect, None, None, None, "INSUFFICIENT DATA"))
        continue
    slope, intercept, r, p, se = stats.linregress(tmp[cause], tmp[effect])
    status = "KEEP" if p < 0.05 else "WEAK"
    results.append((cause, effect, round(slope, 4), round(p, 4), round(r**2, 4), status))

results_df = pd.DataFrame(
    results,
    columns=["cause", "effect", "slope", "p_value", "r_squared", "status"]
)

print("\n── Statistical Pruning Results ──")
print(results_df.to_string(index=False))

kept_edges  = [(r.cause, r.effect) for r in results_df.itertuples() if r.status == "KEEP"]
weak_edges  = [(r.cause, r.effect) for r in results_df.itertuples() if r.status == "WEAK"]
error_edges = [(r.cause, r.effect) for r in results_df.itertuples() if r.status not in ("KEEP", "WEAK")]

print(f"\nKept:  {len(kept_edges)} edges")
print(f"Weak:  {len(weak_edges)} edges")
print(f"Error: {len(error_edges)} edges")

results_df.to_csv("src/data/dag_pruning_results.csv", index=False)
print("Pruning results saved to dag_pruning_results.csv")

# ── 4. Friendly labels ────────────────────────────────────────
labels = {
    "fed_funds_change":       "Fed Funds\nRate Δ",
    "yield_10y_change":       "10Y Yield Δ",
    "yield_2y_change":        "2Y Yield Δ",
    "real_yield_10y_change":  "Real 10Y\nYield Δ",
    "breakeven_5y_change":    "5Y Breakeven\nInflation Δ",
    "term_spread_chg":        "Term\nSpread Δ",
    "dxy_return":             "DXY\nReturn",
    "eurusd_return":          "EUR/USD\nReturn",
    "oil_return":             "Oil WTI\nReturn",
    "copper_return":          "Copper\nReturn",
    "sp500_return":           "S&P 500\nReturn",
    "vix_chg":                "VIX Δ",
    "hy_spread_change":       "HY Credit\nSpread Δ",
    "gold_return":            "Gold Return",
    "gvz_chg":                "GVZ Δ",
    "gld_return":             "GLD ETF\nReturn",
    "gdx_return":             "GDX\nReturn",
    "silver_return":          "Silver\nReturn",
}

# ── 5. Build graphviz dot graph ───────────────────────────────
dot = graphviz.Digraph(
    name="GoldDAG",
    graph_attr={
        "rankdir":  "TB",
        "splines":  "ortho",
        "nodesep":  "0.6",
        "ranksep":  "0.8",
        "bgcolor":  "white",
        "fontname": "Helvetica",
        "size":     "14,18",
        "dpi":      "150",
    },
    node_attr={
        "shape":    "ellipse",
        "style":    "filled",
        "fillcolor":"white",
        "color":    "black",
        "fontname": "Helvetica",
        "fontsize": "11",
        "margin":   "0.15,0.08",
    },
    edge_attr={
        "fontname": "Helvetica",
        "fontsize": "9",
        "color":    "black",
        "arrowsize":"0.7",
    }
)

# Add nodes
all_nodes = set()
for c, e in edges:
    all_nodes.add(c)
    all_nodes.add(e)

for n in all_nodes:
    lbl = labels.get(n, n)
    if n == "gold_return":
        dot.node(n, label=lbl, penwidth="2.5", fontsize="12")
    else:
        dot.node(n, label=lbl)

# Add edges — solid for KEEP, dashed for WEAK
edge_status = {(r.cause, r.effect): r.status for r in results_df.itertuples()}
for cause, effect in edges:
    status = edge_status.get((cause, effect), "WEAK")
    if status == "KEEP":
        dot.edge(cause, effect)
    else:
        dot.edge(cause, effect, style="dashed", color="gray50")

# ── 6. Render ─────────────────────────────────────────────────
dot.render("src/results/gold_dag_clean", format="png", cleanup=True)
print("\nDAG saved to gold_dag_clean.png")