# Chooser Option Pricing
# Chooser Option Pricing with Real-World Data & Machine Learning

**Author:** 张筱涵 | **Department:** Quantitative Research & Trading | **Duration:** 5 Weeks

## Project Overview

This project builds an advanced Chooser Option pricing tool that combines the traditional Black-Scholes-Merton (BSM) model with Machine Learning to address BSM's core limitation of constant volatility assumption.

A **Chooser Option** grants the holder the right to decide, at a future choice date, whether the option will be a European call or put:

$$V_0 = C(S, K, T, r, \sigma) + P(S, Ke^{-r(T-t_c)}, t_c, r, \sigma)$$

## Key Findings

- BSM constant volatility assumption violated: **CV = 0.62** (BSM assumes 0)
- High vs Low VIX regime price gap: **40.1%**
- COVID crisis pricing: **3x** normal period
- Best ML model (GBDT): **MAE = $9.75, R² = 0.21**
- Volatility (RealVol_30d) is the dominant pricing feature: **SHAP = 5.79**

## Project Structure
chooser-option-pricing/
├── data/
│   ├── raw/                    # Raw data (JPM, VIX, Treasury)
│   └── processed/              # Cleaned dataset & BSM results
│       ├── featured_dataset.csv
│       └── bsm_results.csv
├── scripts/
│   ├── data_collector.py       # Automated data collection
│   └── bsm_pricing.py         # Automated BSM pricing
├── models/
│   ├── best_gbdt.pkl          # Best ML model (GBDT)
│   ├── best_rf.pkl            # Random Forest model
│   └── scaler.pkl             # Feature scaler
├── notebooks/
│   ├── week1_data_collection.ipynb
│   ├── week3_BSM_model.ipynb
│   └── week5_ML_model.ipynb
├── app/
│   └── app.py                 # Streamlit pricing tool
├── reports/                   # Weekly reports
├── requirements.txt
└── README.md

## Data Pipeline

| Source | Data | Frequency |
|--------|------|-----------|
| Yahoo Finance | JPM stock price, VIX | Daily |
| FRED API | Treasury yields (1M–10Y), Fed Funds, CPI | Daily/Monthly |

**Period:** 2018–2024 | **Size:** 1,760 rows × 14 features

### Engineered Features (14 total)

| Category | Features |
|----------|----------|
| BSM Core | JPM_Close, Yield_3M, RealVol_30d |
| Volatility | RealVol_10d, RealVol_30d, RealVol_60d, VIX_Close |
| Regime | VIX_Regime (Low/Medium/High), VIX_Regime_Code |
| Macro | Yield_2Y, Yield_10Y, Yield_Spread, Rate_Momentum |
| Sentiment | Sentiment_Score (VIX-based proxy, 0–1) |
| Correlation | VIX_JPM_Corr |


## Automated Pipelines (GitHub Actions)

Two automated workflows run daily:

**1. Auto Data Update** (`data_update.yml`)
- Triggers: Daily at 01:00 UTC
- Runs `scripts/data_collector.py`
- Updates `data/processed/featured_dataset.csv`

**2. Auto BSM Pricing** (`bsm_pricing.yml`)
- Triggers: After Auto Data Update completes
- Runs `scripts/bsm_pricing.py`
- Updates `data/processed/bsm_results.csv`

## BSM Model Results

Using ATM parameters (K = S) on 2018–2024 data:

| Metric | Value |
|--------|-------|
| Average Chooser Price | $20.53 |
| High VIX Regime Avg | $24.08 |
| Low VIX Regime Avg | $17.19 |
| Price Gap (High vs Low) | 40.1% |
| COVID Period Avg | $36.66 (3.03x normal) |
| Volatility CV | 0.62 (BSM assumes 0) |

---

## ML Model Results

Two approaches trained on 70/15/15 time-series split:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| BSM Baseline | $0.00 | $0.00 | 1.00 |
| Approach 1 (ML Vol + BSM) | $10.66 | $13.44 | 0.06 |
| Best RF (Tuned) | $11.73 | $14.06 | -0.03 |
| **Best GBDT (Tuned)** | **$9.75** | **$12.35** | **0.21** |

### SHAP Feature Importance (Top 5)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 1 | RealVol_30d | 5.79 |
| 2 | JPM_Close | 4.85 |
| 3 | Yield_3M | 0.30 |
| 4 | Yield_2Y | 0.15 |
| 5 | Yield_10Y | 0.11 |

## Pricing Tool

🔗 **Live Demo:** [Chooser Option Pricing Tool](https://chooser-option-pricing-uacpcfxmdunvzdck6kgpve.streamlit.app/)

The Streamlit app includes 5 pages:
- 🏠 **Home** — Project overview & key findings
- 💰 **Pricing** — BSM + GBDT dual-mode pricing with error margin
- 📊 **Sensitivity** — Volatility & rate shock analysis, combined heatmap
- 📉 **Dashboard** — Historical JPM, VIX & Chooser price data
- 🤖 **Model Comparison** — Performance metrics & SHAP feature importance

## Sensitivity Analysis (Extreme Scenarios)

Base parameters: JPM = $231.08, r = 4.37%, σ = 18.68%

| Scenario | Price | Change |
|----------|-------|--------|
| Base Case | $29.84 | — |
| Vol Spike +50% | $43.78 | +46.72% |
| Rate Hike +2% | $30.76 | +3.08% |
| Vol + Rate Combined | $44.16 | +48.02% |

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/xzha0487/chooser-option-pricing.git
cd chooser-option-pricing

# Install dependencies
pip install -r requirements.txt

# Run the pricing tool locally
streamlit run app/app.py
