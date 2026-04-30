import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load data ──────────────────────────────────────────────
df = pd.read_csv("src/data/gold_causal_data.csv", index_col=0, parse_dates=True)
print(f"Data loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ── 2. Define causal parent features (KEEP edges → gold_return)
causal_features = [
    "real_yield_10y_change",
    "breakeven_5y_change",
    "dxy_return",
    "oil_return",
    "vix_chg",
    "hy_spread_change",
    "sp500_return",
]

# ── 3. Add lagged features (yesterday → today signals) ────────
lag_cols = ["gold_return", "gdx_return", "silver_return",
            "gld_return", "gvz_chg"]
for col in lag_cols:
    if col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

lagged_features = [f"{col}_lag1" for col in lag_cols if col in df.columns] + \
                  [f"{col}_lag2" for col in lag_cols if col in df.columns]

# ── 4. Full feature set ───────────────────────────────────────
all_features = causal_features + lagged_features

model_df = df[all_features + ["target_gold_ret", "target_gold_dir"]].dropna()
print(f"Model dataset: {model_df.shape[0]} rows after dropping NaNs")

X = model_df[all_features]
y_ret = model_df["target_gold_ret"]
y_dir = model_df["target_gold_dir"]

# ── 5. Scale features ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=all_features, index=model_df.index)

# ── 6. Time-series cross-validation setup ─────────────────────
tscv = TimeSeriesSplit(n_splits=5)

# ── 7. Lasso regression (continuous return prediction) ────────
print("\n── Lasso Regression (predicting gold return) ──")

lasso = LassoCV(cv=tscv, max_iter=10000, random_state=42)
lasso.fit(X_scaled, y_ret)

y_pred_ret = lasso.predict(X_scaled)
r2   = r2_score(y_ret, y_pred_ret)
rmse = np.sqrt(mean_squared_error(y_ret, y_pred_ret))

print(f"R²:   {r2:.4f}")
print(f"RMSE: {rmse:.6f}")
print(f"Best alpha (regularisation): {lasso.alpha_:.6f}")

coef_df = pd.DataFrame({
    "feature":     all_features,
    "coefficient": lasso.coef_,
}).sort_values("coefficient", key=abs, ascending=False)
coef_df = coef_df[coef_df["coefficient"] != 0].reset_index(drop=True)

print("\nSurviving features after Lasso:")
print(coef_df.to_string(index=False))

coef_df.to_csv("src/data/lasso_coefficients.csv", index=False)

# ── 8. Logistic regression (directional classifier) ───────────
print("\n── Logistic Regression (predicting up/down direction) ──")

log_reg = LogisticRegressionCV(
    cv=tscv, max_iter=10000, random_state=42,
    penalty="l1", solver="saga"
)
log_reg.fit(X_scaled, y_dir)

y_pred_dir = log_reg.predict(X_scaled)
acc = accuracy_score(y_dir, y_pred_dir)

print(f"Directional accuracy: {acc:.4f} ({acc*100:.1f}%)")
print("\nClassification report:")
print(classification_report(y_dir, y_pred_dir, target_names=["Down (0)", "Up (1)"]))

log_coef_df = pd.DataFrame({
    "feature":     all_features,
    "coefficient": log_reg.coef_[0],
}).sort_values("coefficient", key=abs, ascending=False)
log_coef_df = log_coef_df[log_coef_df["coefficient"] != 0].reset_index(drop=True)

print("Surviving features (logistic):")
print(log_coef_df.to_string(index=False))
log_coef_df.to_csv("src/data/logistic_coefficients.csv", index=False)

# ── 9. Plot 1 — Actual vs Predicted Return ────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
fig.patch.set_facecolor("white")

ax.plot(model_df.index, y_ret.values, color="#bbbbbb", linewidth=0.7,
        label="Actual Gold Return", alpha=1.0)
ax.plot(model_df.index, y_pred_ret, color="#1a6e7a", linewidth=1.4,
        label="Lasso Predicted Return", alpha=0.9)
ax.axhline(0, color="#999999", linewidth=0.6, linestyle="--")
ax.set_title("Actual vs Predicted Gold Daily Return (Lasso)",
             fontsize=15, fontweight="bold", color="#111111", pad=14)
ax.set_ylabel("Daily Return", fontsize=12)
ax.set_xlabel("Date", fontsize=12)
ax.legend(fontsize=11, frameon=False)
ax.tick_params(axis="both", labelsize=11)
ax.set_facecolor("white")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("src/results/chart_return_prediction.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Chart 1 saved to chart_return_prediction.png")

# ── Plot 2 — Lasso Coefficients with beta labels ──────────────
name_map = {
    "gold_return_lag1":       "Gold Return (lag 1)",
    "gold_return_lag2":       "Gold Return (lag 2)",
    "gld_return_lag1":        "GLD ETF Return (lag 1)",
    "gld_return_lag2":        "GLD ETF Return (lag 2)",
    "gdx_return_lag1":        "GDX Miners (lag 1)",
    "gdx_return_lag2":        "GDX Miners (lag 2)",
    "silver_return_lag1":     "Silver Return (lag 1)",
    "silver_return_lag2":     "Silver Return (lag 2)",
    "gvz_chg_lag1":           "GVZ Volatility (lag 1)",
    "gvz_chg_lag2":           "GVZ Volatility (lag 2)",
    "dxy_return":             "DXY Dollar Return",
    "sp500_return":           "S&P 500 Return",
    "oil_return":             "Oil WTI Return",
    "vix_chg":                "VIX Change",
    "real_yield_10y_change":  "Real 10Y Yield Δ",
    "breakeven_5y_change":    "5Y Breakeven Inflation Δ",
    "hy_spread_change":       "HY Credit Spread Δ",
}

plot_df = coef_df.copy()
plot_df["label"] = plot_df["feature"].map(name_map).fillna(plot_df["feature"])
plot_df = plot_df.sort_values("coefficient")

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor("white")

colors = ["#c21a0e" if c < 0 else "#128f07" for c in plot_df["coefficient"]]
bars = ax.barh(plot_df["label"], plot_df["coefficient"], color=colors,
               height=0.62, edgecolor="white", linewidth=0.4)

ax.axvline(0, color="#888888", linewidth=0.8)

# Beta annotations — shown INSIDE or just outside each bar
for i, (val, label) in enumerate(zip(plot_df["coefficient"], plot_df["label"])):
    beta_str = f"β = {val:.4f}"
    if val >= 0:
        # Positive bar — label just to the right of the bar end
        ax.text(val + abs(plot_df["coefficient"]).max() * 0.01,
                i, beta_str, va="center", ha="left",
                fontsize=9, color="#128f07", fontweight="bold")
    else:
        # Negative bar — label just to the left of the bar end
        ax.text(val - abs(plot_df["coefficient"]).max() * 0.01,
                i, beta_str, va="center", ha="right",
                fontsize=9, color="#c21a0e", fontweight="bold")

# Extend x-axis slightly to make room for labels
x_max = abs(plot_df["coefficient"]).max()
ax.set_xlim(-x_max * 1.4, x_max * 1.4)

ax.set_title("Lasso Coefficients (β) — Surviving Causal Features",
             fontsize=15, fontweight="bold", color="#111111", pad=14)
ax.set_xlabel("β Coefficient (standardised features)", fontsize=12)
ax.tick_params(axis="y", labelsize=11)
ax.tick_params(axis="x", labelsize=10)
ax.set_facecolor("white")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("src/data/resultschart_lasso_coefficients.png", dpi=180, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Chart 2 saved to chart_lasso_coefficients.png")
# ── 10. Summary ───────────────────────────────────────────────
print("\n── Summary ──")
print(f"Total features input:          {len(all_features)}")
print(f"  Causal (DAG):                {len(causal_features)}")
print(f"  Lagged:                      {len(lagged_features)}")
print(f"Lasso survivors:               {len(coef_df)}")
print(f"Logistic survivors:            {len(log_coef_df)}")
print(f"Regression R²:                 {r2:.4f}")
print(f"Directional accuracy:          {acc*100:.1f}%")
print(f"Baseline (always predict up):  {y_dir.mean()*100:.1f}%")