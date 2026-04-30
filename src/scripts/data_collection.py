import yfinance as yf
import pandas as pd

START = "2005-01-01"
END   = "2026-04-28"

# ── 1. Yahoo Finance ──────────────────────────────────────────
yf_tickers = {
    "GC=F"     : "gold",
    "DX-Y.NYB" : "dxy",
    "EURUSD=X" : "eurusd",
    "CL=F"     : "oil",
    "HG=F"     : "copper",
    "^VIX"     : "vix",
    "^GVZ"     : "gvz",
    "^GSPC"    : "sp500",
    "SI=F"     : "silver",
    "GLD"      : "gld",
    "GDX"      : "gdx",
}

raw_yf = yf.download(list(yf_tickers.keys()), start=START, end=END, auto_adjust=True)["Close"]
raw_yf.columns = [yf_tickers[t] for t in raw_yf.columns]

# ── 2. FRED (direct CSV — no pandas_datareader needed) ────────
fred_map = {
    "DGS10"        : "yield_10y",
    "DGS2"         : "yield_2y",
    "DFII10"       : "real_yield_10y",
    "T5YIE"        : "breakeven_5y",
    "DFF"          : "fed_funds",
    "BAMLH0A0HYM2" : "hy_spread",
}

fred_frames = []
for fred_id, col_name in fred_map.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
    s = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    s.columns = [col_name]
    fred_frames.append(s)

raw_fred = pd.concat(fred_frames, axis=1, sort=True)
raw_fred = raw_fred.loc[START:END]

# ── 3. Merge ──────────────────────────────────────────────────
df = raw_yf.join(raw_fred, how="outer")

fred_cols = list(fred_map.values())
df[fred_cols] = df[fred_cols].ffill()

df = df[df["gold"].notna()].copy()

# ── 4. Compute returns & changes ──────────────────────────────
price_cols = ["gold","dxy","eurusd","oil","copper","sp500","silver","gld","gdx"]
for col in price_cols:
    df[f"{col}_return"] = df[col].pct_change()

level_cols = ["yield_10y","yield_2y","real_yield_10y",
              "breakeven_5y","fed_funds","hy_spread"]
for col in level_cols:
    df[f"{col}_change"] = df[col].diff()

df["vix_chg"] = df["vix"].diff()
df["gvz_chg"] = df["gvz"].diff()

# ── 5. Derived feature ────────────────────────────────────────
df["term_spread"]     = df["yield_10y"] - df["yield_2y"]
df["term_spread_chg"] = df["term_spread"].diff()

# ── 6. Target variable ────────────────────────────────────────
df["target_gold_ret"] = df["gold_return"]
df["target_gold_dir"] = (df["gold_return"] > 0).astype(int)

# ── 7. Drop raw level columns ─────────────────────────────────
drop_cols = price_cols + level_cols + ["vix", "gvz", "term_spread"]
df_clean = df.drop(columns=drop_cols).dropna()

# ── 8. Save ───────────────────────────────────────────────────
df_clean.to_csv("src/data/gold_causal_data.csv")
print(df_clean.shape)
print(df_clean.head())