
import numpy as np
import pandas as pd
from scipy.stats import norm

def bsm_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bsm_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def chooser_option(S, K, T, tc, r, sigma):
    call = bsm_call(S, K, T, r, sigma)
    K_adj = K * np.exp(-r*(T-tc))
    put = bsm_put(S, K_adj, tc, r, sigma)
    return call + put, call, put

# Load data
df = pd.read_csv("data/processed/featured_dataset.csv", index_col=0, parse_dates=True)

# BSM pricing
results = []
for date, row in df.iterrows():
    S     = row["JPM_Close"]
    K     = S
    r     = row["Yield_3M"] / 100
    sigma = row["RealVol_30d"]
    T     = 1.0
    tc    = 0.5

    if pd.isna(sigma) or sigma <= 0:
        continue

    price, call, put = chooser_option(S, K, T, tc, r, sigma)

    results.append({
        "Date":           date,
        "JPM_Close":      S,
        "K":              K,
        "sigma":          sigma,
        "r":              r,
        "Call":           call,
        "Put":            put,
        "Chooser_Price":  price,
        "Call_Put_Ratio": call / put,
        "VIX_Regime":     row["VIX_Regime"]
    })

results_df = pd.DataFrame(results).set_index("Date")

# Save full results
results_df.to_csv("data/processed/bsm_results.csv")
print(f"BSM pricing complete: {len(results_df)} rows")

# Regime comparison
print("\n-- Chooser Price by VIX Regime --")
regime_stats = results_df.groupby("VIX_Regime")["Chooser_Price"].agg([
    "mean", "std", "min", "max", "count"
]).round(4)
print(regime_stats)

# Call/Put ratio by regime
print("\n-- Call/Put Ratio by VIX Regime --")
ratio_stats = results_df.groupby("VIX_Regime")["Call_Put_Ratio"].mean().round(4)
print(ratio_stats)

# COVID vs normal period
print("\n-- COVID vs Normal Period --")
covid  = results_df["2020-02-01":"2020-05-31"]
normal = results_df["2018-01-01":"2019-12-31"]
print(f"COVID period avg price  : ${covid['Chooser_Price'].mean():.4f}")
print(f"Normal period avg price : ${normal['Chooser_Price'].mean():.4f}")
print(f"Ratio                   : {covid['Chooser_Price'].mean()/normal['Chooser_Price'].mean():.2f}x")

# Volatility stability
print("\n-- Volatility Stability Test --")
print(f"Mean sigma : {results_df['sigma'].mean()*100:.2f}%")
print(f"Std sigma  : {results_df['sigma'].std()*100:.2f}%")
print(f"CV         : {results_df['sigma'].std()/results_df['sigma'].mean():.4f}")
print(f"BSM assumes CV = 0, actual CV = {results_df['sigma'].std()/results_df['sigma'].mean():.4f}")

# Save summary report
summary = {
    "total_days":          len(results_df),
    "avg_chooser_price":   round(results_df["Chooser_Price"].mean(), 4),
    "std_chooser_price":   round(results_df["Chooser_Price"].std(), 4),
    "avg_sigma":           round(results_df["sigma"].mean(), 4),
    "cv_sigma":            round(results_df["sigma"].std()/results_df["sigma"].mean(), 4),
    "high_regime_avg":     round(results_df[results_df["VIX_Regime"]=="High"]["Chooser_Price"].mean(), 4),
    "low_regime_avg":      round(results_df[results_df["VIX_Regime"]=="Low"]["Chooser_Price"].mean(), 4),
    "covid_avg":           round(covid["Chooser_Price"].mean(), 4),
    "normal_avg":          round(normal["Chooser_Price"].mean(), 4),
    "covid_normal_ratio":  round(covid["Chooser_Price"].mean()/normal["Chooser_Price"].mean(), 4),
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv("data/processed/bsm_summary.csv", index=False)
print("\nSummary report saved to data/processed/bsm_summary.csv")
