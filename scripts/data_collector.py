import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred

START = "2018-01-01"
END = "2024-12-31"
fred = Fred(api_key="793ea442d5187f9e5b7b0c30de96367a")

# Download data
jpm = yf.download("JPM", start=START, end=END)
jpm.columns = [c[0] for c in jpm.columns]

vix = yf.download("^VIX", start=START, end=END)
vix.columns = [c[0] for c in vix.columns]

treasury_series = {
    "DGS3MO": "Yield_3M",
    "DGS1": "Yield_1Y",
    "DGS2": "Yield_2Y",
    "DGS10": "Yield_10Y"
}
treasury = pd.DataFrame()
for code, name in treasury_series.items():
    s = fred.get_series(code, observation_start=START, observation_end=END)
    treasury[name] = s

# Merge
jpm_close = jpm[["Close"]].copy()
jpm_close.columns = ["JPM_Close"]
jpm_close.index = pd.to_datetime(jpm_close.index)

vix_close = vix[["Close"]].copy()
vix_close.columns = ["VIX_Close"]
vix_close.index = pd.to_datetime(vix_close.index)

treasury.index = pd.to_datetime(treasury.index)

master = jpm_close.join(vix_close, how="left")
master = master.join(treasury, how="left")
master = master.ffill().dropna()

# Feature engineering
df = master.copy()
df["Log_Return"] = np.log(df["JPM_Close"] / df["JPM_Close"].shift(1))
df["RealVol_10d"] = df["Log_Return"].rolling(10).std() * np.sqrt(252)
df["RealVol_30d"] = df["Log_Return"].rolling(21).std() * np.sqrt(252)
df["RealVol_60d"] = df["Log_Return"].rolling(63).std() * np.sqrt(252)

lo = df["VIX_Close"].quantile(0.33)
hi = df["VIX_Close"].quantile(0.66)
df["VIX_Regime"] = pd.cut(df["VIX_Close"],
                           bins=[-np.inf, lo, hi, np.inf],
                           labels=["Low", "Medium", "High"])

df["Rate_Momentum"] = df["Yield_3M"].diff(10)
df["Yield_Spread"] = df["Yield_10Y"] - df["Yield_2Y"]
df["VIX_JPM_Corr"] = df["VIX_Close"].rolling(30).corr(df["JPM_Close"])
df = df.dropna()

# Save
df.to_csv("data/processed/featured_dataset.csv")
print(f"Data updated: {df.shape[0]} rows x {df.shape[1]} columns")

