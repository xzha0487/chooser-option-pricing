import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Project paths
# app.py location:
# chooser-option-pricing/app/app.py
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"


# ============================================================
# Load models
# ============================================================

@st.cache_resource
def load_models():
    gbdt_path = MODEL_DIR / "best_gbdt.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    if not gbdt_path.exists():
        raise FileNotFoundError(f"Model file not found: {gbdt_path}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    with open(gbdt_path, "rb") as f:
        gbdt = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return gbdt, scaler


# ============================================================
# Load data
# ============================================================

@st.cache_data
def load_data():
    featured_path = DATA_DIR / "featured_dataset.csv"
    results_path = DATA_DIR / "bsm_results.csv"

    if not featured_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {featured_path}"
        )

    if not results_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {results_path}"
        )

    df = pd.read_csv(
        featured_path,
        index_col=0,
        parse_dates=True
    )

    results_df = pd.read_csv(
        results_path,
        index_col=0,
        parse_dates=True
    )

    vix_range = df["VIX_Close"].max() - df["VIX_Close"].min()

    if vix_range == 0:
        df["Sentiment_Score"] = 0.5
    else:
        df["Sentiment_Score"] = 1 - (
            (df["VIX_Close"] - df["VIX_Close"].min())
            / vix_range
        )

    return df, results_df


# ============================================================
# Run loading functions
# ============================================================

gbdt, scaler = load_models()
df, results_df = load_data()
# ============================================================
# BSM Functions
# ============================================================
def bsm_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bsm_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def chooser_option(S, K, T, tc, r, sigma):
    call  = bsm_call(S, K, T, r, sigma)
    K_adj = K * np.exp(-r*(T-tc))
    put   = bsm_put(S, K_adj, tc, r, sigma)
    return call + put, call, put

def get_ml_features(S, r, sigma, vix, yield_2y, yield_10y):
    log_ret = 0.0
    feature_dict = {
        "JPM_Close":      S,
        "VIX_Close":      vix,
        "Yield_3M":       r * 100,
        "Yield_2Y":       yield_2y,
        "Yield_10Y":      yield_10y,
        "Log_Return":     log_ret,
        "RealVol_10d":    sigma,
        "RealVol_30d":    sigma,
        "RealVol_60d":    sigma,
        "Rate_Momentum":  0.0,
        "Yield_Spread":   yield_10y - yield_2y,
        "VIX_JPM_Corr":  -0.7,
        "Sentiment_Score": 1 - (vix - df["VIX_Close"].min()) / (
            df["VIX_Close"].max() - df["VIX_Close"].min()),
        "VIX_Regime_Code": 0 if vix < 15.7 else (1 if vix < 21.1 else 2),
    }
    feature_cols = [
        "JPM_Close", "VIX_Close", "Yield_3M", "Yield_2Y", "Yield_10Y",
        "Log_Return", "RealVol_10d", "RealVol_30d", "RealVol_60d",
        "Rate_Momentum", "Yield_Spread", "VIX_JPM_Corr",
        "Sentiment_Score", "VIX_Regime_Code",
    ]
    X = pd.DataFrame([feature_dict])[feature_cols]
    X_scaled = scaler.transform(X)
    return X_scaled

# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("📈 Chooser Option Tool")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "💰 Pricing", "📊 Sensitivity", 
     "📉 Dashboard", "🤖 Model Comparison"]
)

# ============================================================
# Page 1: Home
# ============================================================
if page == "🏠 Home":
    st.title("Chooser Option Pricing Tool")
    st.markdown("### Advanced BSM + Machine Learning Pricing System")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Underlying Asset", "JPM")
    with col2:
        latest_price = df["JPM_Close"].iloc[-1]
        st.metric("Latest JPM Price", f"${latest_price:.2f}")
    with col3:
        latest_vix = df["VIX_Close"].iloc[-1]
        st.metric("Latest VIX", f"{latest_vix:.2f}")
    with col4:
        latest_chooser = results_df["Chooser_Price"].iloc[-1]
        st.metric("Latest Chooser Price", f"${latest_chooser:.2f}")

    st.markdown("---")
    st.markdown("""
    #### What is a Chooser Option?
    A Chooser Option grants the holder the right to decide, at a future 
    **choice date**, whether the option will be a standard European **call** 
    or **put**, with a common strike and final expiration.
    
    **Pricing Formula:**
    """)
    st.latex(r"V_0 = C(S, K, T, r, \sigma) + P(S, Ke^{-r(T-t_c)}, t_c, r, \sigma)")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Project Overview
        - **Data:** JPM, VIX, Treasury Yields (2018-2024)
        - **Features:** 14 engineered features including sentiment score
        - **Models:** BSM baseline + GBDT ML model
        - **Training:** 70/15/15 time-series split
        """)
    with col2:
        st.markdown("""
        #### Key Findings
        - BSM constant volatility assumption violated (CV=0.62)
        - High vs Low VIX regime price gap: 40.1%
        - COVID crisis pricing: 3x normal period
        - Best ML model (GBDT): MAE=$9.75, R²=0.21
        """)

# ============================================================
# Page 2: Pricing
# ============================================================
elif page == "💰 Pricing":
    st.title("💰 Chooser Option Pricing")
    st.markdown("Enter parameters to calculate Chooser Option price")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Market Parameters")
        S     = st.number_input("Stock Price (S) $",
                                 min_value=50.0, max_value=500.0,
                                 value=float(df["JPM_Close"].iloc[-1]),
                                 step=1.0)
        K     = st.number_input("Strike Price (K) $",
                                 min_value=50.0, max_value=500.0,
                                 value=float(df["JPM_Close"].iloc[-1]),
                                 step=1.0)
        T     = st.slider("Time to Expiry (T) years",
                           min_value=0.25, max_value=2.0,
                           value=1.0, step=0.25)
        tc    = st.slider("Choice Date (tc) years",
                           min_value=0.1, max_value=T-0.1,
                           value=min(0.5, T-0.1), step=0.1)

    with col2:
        st.subheader("Model Parameters")
        r     = st.number_input("Risk-free Rate (%)",
                                 min_value=0.0, max_value=10.0,
                                 value=float(df["Yield_3M"].iloc[-1]),
                                 step=0.1) / 100
        sigma = st.number_input("Volatility (sigma) %",
                                 min_value=1.0, max_value=150.0,
                                 value=float(df["RealVol_30d"].iloc[-1]*100),
                                 step=1.0) / 100
        vix   = st.number_input("VIX Level",
                                 min_value=5.0, max_value=90.0,
                                 value=float(df["VIX_Close"].iloc[-1]),
                                 step=0.5)
        yield_2y  = st.number_input("2Y Yield (%)",
                                     min_value=0.0, max_value=10.0,
                                     value=float(df["Yield_2Y"].iloc[-1]),
                                     step=0.1)
        yield_10y = st.number_input("10Y Yield (%)",
                                     min_value=0.0, max_value=10.0,
                                     value=float(df["Yield_10Y"].iloc[-1]),
                                     step=0.1)

    st.markdown("---")
    if st.button("Calculate Price", type="primary"):
        # BSM pricing
        bsm_price, bsm_call_val, bsm_put_val = chooser_option(
            S, K, T, tc, r, sigma)

        # ML pricing
        X_scaled  = get_ml_features(S, r, sigma, vix, yield_2y, yield_10y)
        ml_price  = gbdt.predict(X_scaled)[0]
        error_margin = abs(bsm_price - ml_price)

        # VIX regime
        if vix < 15.7:
            regime = "🟢 Low"
        elif vix < 21.1:
            regime = "🟡 Medium"
        else:
            regime = "🔴 High"

        st.markdown("### Pricing Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BSM Price", f"${bsm_price:.4f}")
        with col2:
            st.metric("ML Price (GBDT)", f"${ml_price:.4f}")
        with col3:
            st.metric("Error Margin", f"${error_margin:.4f}")
        with col4:
            st.metric("VIX Regime", regime)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### BSM Components")
            st.write(f"- Call Component: **${bsm_call_val:.4f}**")
            st.write(f"- Put Component:  **${bsm_put_val:.4f}**")
            st.write(f"- Total (BSM):    **${bsm_price:.4f}**")
        with col2:
            st.markdown("#### Market Inputs Summary")
            st.write(f"- Sentiment Score: **{1-(vix-df['VIX_Close'].min())/(df['VIX_Close'].max()-df['VIX_Close'].min()):.4f}**")
            st.write(f"- Yield Spread (10Y-2Y): **{yield_10y-yield_2y:.2f}%**")
            st.write(f"- Moneyness (S/K): **{S/K:.4f}**")

# ============================================================
# Page 3: Sensitivity Analysis
# ============================================================
elif page == "📊 Sensitivity":
    st.title("📊 Sensitivity Analysis")
    st.markdown("---")

    latest = df.iloc[-1]
    S_base     = latest["JPM_Close"]
    r_base     = latest["Yield_3M"] / 100
    sigma_base = latest["RealVol_30d"]

    tab1, tab2, tab3 = st.tabs(
        ["Volatility Shock", "Rate Shock", "Combined Heatmap"])

    with tab1:
        st.subheader("Chooser Price vs Volatility Shock")
        vol_range  = np.arange(-0.5, 1.1, 0.1)
        prices_vol = [chooser_option(S_base, S_base, 1.0, 0.5,
                                      r_base, max(sigma_base*(1+v), 0.01))[0]
                      for v in vol_range]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(vol_range*100, prices_vol,
                color="red", lw=2, marker="o", markersize=5)
        ax.axvline(0,  color="black", ls="--", lw=1, label="Base")
        ax.axvline(50, color="red",   ls="--", lw=1, label="+50% spike")
        ax.set_xlabel("Volatility Change (%)")
        ax.set_ylabel("Chooser Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab2:
        st.subheader("Chooser Price vs Rate Shock")
        rate_range  = np.arange(-0.02, 0.051, 0.005)
        prices_rate = [chooser_option(S_base, S_base, 1.0, 0.5,
                                       r_base+dr, sigma_base)[0]
                       for dr in rate_range]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rate_range*100, prices_rate,
                color="steelblue", lw=2, marker="o", markersize=5)
        ax.axvline(0, color="black", ls="--", lw=1, label="Base")
        ax.axvline(2, color="red",   ls="--", lw=1, label="+2% hike")
        ax.set_xlabel("Rate Change (%)")
        ax.set_ylabel("Chooser Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab3:
        st.subheader("Combined Shock Heatmap")
        vol_changes  = np.arange(-0.3, 0.61, 0.1)
        rate_changes = np.arange(-0.01, 0.031, 0.005)
        price_matrix = np.zeros((len(vol_changes), len(rate_changes)))
        for i, dv in enumerate(vol_changes):
            for j, dr in enumerate(rate_changes):
                sigma_new = max(sigma_base*(1+dv), 0.01)
                price_matrix[i,j] = chooser_option(
                    S_base, S_base, 1.0, 0.5, r_base+dr, sigma_new)[0]
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(price_matrix, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(rate_changes)))
        ax.set_yticks(range(len(vol_changes)))
        ax.set_xticklabels([f"{r*100:+.1f}%" for r in rate_changes],
                            rotation=45)
        ax.set_yticklabels([f"{v*100:+.0f}%" for v in vol_changes])
        ax.set_xlabel("Rate Change")
        ax.set_ylabel("Volatility Change")
        plt.colorbar(im, ax=ax, label="Chooser Price ($)")
        for i in range(len(vol_changes)):
            for j in range(len(rate_changes)):
                ax.text(j, i, f"${price_matrix[i,j]:.1f}",
                        ha="center", va="center", fontsize=7)
        st.pyplot(fig)

# ============================================================
# Page 4: Dashboard
# ============================================================
elif page == "📉 Dashboard":
    st.title("📉 Historical Data Dashboard")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["JPM Price", "VIX & Regime", "Chooser Price History"])

    with tab1:
        st.subheader("JPM Stock Price (2018-2024)")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["JPM_Close"],
                color="steelblue", lw=1.2, label="JPM Close")
        ma20 = df["JPM_Close"].rolling(20).mean()
        ma60 = df["JPM_Close"].rolling(60).mean()
        ax.plot(df.index, ma20, color="orange", lw=0.8,
                ls="--", label="MA 20")
        ax.plot(df.index, ma60, color="red", lw=0.8,
                ls="--", label="MA 60")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab2:
        st.subheader("VIX & Regime Classification")
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = {"Low": "green", "Medium": "orange", "High": "red"}
        for regime, color in colors.items():
            mask = df["VIX_Regime"] == regime
            ax.scatter(df[mask].index, df[mask]["VIX_Close"],
                       color=color, alpha=0.6, s=5, label=regime)
        ax.axhline(15.7, color="green", ls="--", lw=1,
                   label="Low/Medium boundary (15.7)")
        ax.axhline(21.1, color="red",   ls="--", lw=1,
                   label="Medium/High boundary (21.1)")
        ax.set_ylabel("VIX Level")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab3:
        st.subheader("Historical Chooser Option Price")
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = {"Low": "green", "Medium": "orange", "High": "red"}
        for regime, color in colors.items():
            mask = results_df["VIX_Regime"] == regime
            ax.scatter(results_df[mask].index,
                       results_df[mask]["Chooser_Price"],
                       color=color, alpha=0.6, s=8, label=regime)
        ax.set_ylabel("Chooser Price ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ============================================================
# Page 5: Model Comparison
# ============================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Performance Comparison")
    st.markdown("---")

    # Performance metrics table
    st.subheader("Test Set Performance")
    metrics = pd.DataFrame({
        "Model": ["BSM Baseline", "Approach1 (ML Vol+BSM)",
                  "Best RF (Tuned)", "Best GBDT (Tuned)"],
        "MAE ($)":  [0.0000, 10.6596, 11.7256, 9.7521],
        "RMSE ($)": [0.0000, 13.4394, 14.0628, 12.3548],
        "R²":       [1.0000,  0.0623, -0.0267,  0.2075],
    })
    st.dataframe(metrics, use_container_width=True)

    st.markdown("---")

    # SHAP feature importance
    st.subheader("SHAP Feature Importance")
    shap_data = {
        "Feature": ["RealVol_30d", "JPM_Close", "Yield_3M", "Yield_2Y",
                    "Yield_10Y", "VIX_JPM_Corr", "RealVol_60d",
                    "Log_Return", "VIX_Close", "Sentiment_Score",
                    "RealVol_10d", "Yield_Spread", "Rate_Momentum",
                    "VIX_Regime_Code"],
        "SHAP Importance": [5.787, 4.847, 0.305, 0.151, 0.112,
                             0.099, 0.084, 0.073, 0.071, 0.054,
                             0.044, 0.038, 0.015, 0.000],
    }
    shap_df = pd.DataFrame(shap_data).sort_values(
        "SHAP Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(shap_df["Feature"], shap_df["SHAP Importance"],
            color="steelblue", alpha=0.8)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Feature Importance (SHAP)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **BSM Limitations Found:**
        - Volatility CV = 0.62 (BSM assumes 0)
        - High vs Low regime price gap: 40.1%
        - COVID crisis: 3x normal period pricing
        - Static σ misses regime-dependent dynamics
        """)
    with col2:
        st.markdown("""
        **ML Model Insights:**
        - GBDT best performer (MAE=$9.75, R²=0.21)
        - RealVol_30d dominant feature (SHAP=5.79)
        - JPM_Close second most important (SHAP=4.85)
        - Sentiment score limited contribution (SHAP=0.054)
        """)
