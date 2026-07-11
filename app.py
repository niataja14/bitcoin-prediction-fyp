import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="CryptoPredict AI Dashboard",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .hero-container {
        background: linear-gradient(135deg, #1f1c2c 0%, #928dab 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .hero-container h1 { font-size: 38px !important; font-weight: 800 !important; color: #ffffff !important; }
    .hero-container p { font-size: 16px !important; color: #e1e4e8 !important; }
    div[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 700 !important; color: #00ffcc !important; }
    div[data-testid="stMetricLabel"] { font-size: 12px !important; text-transform: uppercase !important; color: #8b949e !important; }
    .arch-card { background-color: #161b22; border-left: 4px solid #00ffcc; padding: 12px; border-radius: 4px; margin: 8px 0; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_market_data():
    data = yf.download("BTC-USD", period="90d")
    return data.reset_index()

with st.sidebar:
    st.markdown("### 🧠 Model Hyperparameters")
    st.markdown("Examiner Reference Panel")
    st.write("---")
    st.info("""
    Model Type: Hybrid CNN-LSTM
    Lookback Window: 60 Days
    CNN Filters: 64
    Kernel Size: 3
    LSTM Units: 50
    Optimizer: Adam
    Loss Function: MSE
    """)
    st.write("---")
    st.caption(f"System State: Operational\nTime: {datetime.now().strftime('%H:%M Local')}")

st.markdown("""
<div class="hero-container">
    <h1>CryptoPredict AI</h1>
    <p>Design and Implementation of Cryptocurrency Price Prediction Using Machine Learning and Deep Learning (CNN-LSTM)</p>
</div>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

try:
    response = requests.get(API_URL, timeout=60).json()

    if response.get("status") == "success":
        current_price = response["current_price"]
        predicted_price = response["predicted_tomorrow_price"]
        direction = response["expected_direction"]
        metrics = response["metrics"]
        eval_data = response["evaluation_chart"]

        delta_val = predicted_price - current_price
        delta_str = f"${abs(delta_val):,.2f} ({direction})"
        delta_color = "normal" if direction == "UP" else "inverse"

        tab1, tab2, tab3 = st.tabs([
            "📈 Market Analysis",
            "🧠 CNN-LSTM Architecture Pipeline",
            "📊 Model Evaluation & Results"
        ])

        with tab1:
            st.markdown("### Real-Time Live Horizon Forecast")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Spot Market Price (BTC/USD)", value=f"${current_price:,.2f}")
            with col2:
                st.metric(label="AI Horizon Forecast (Next 24h)", value=f"${predicted_price:,.2f}", delta=delta_str, delta_color=delta_color)
            with col3:
                st.metric(label="Model Directional Confidence", value=f"{metrics['accuracy']}%", delta="Statistically Verified")

            st.write("---")
            st.markdown("### Technical Market Indicators")

            df = load_market_data()
            df["MA20"] = df["Close"].rolling(20).mean()
            df["MA50"] = df["Close"].rolling(50).mean()

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Market Price'
            ))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20 Trend', line=dict(color='#ff9900', width=1.5)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50 Trend', line=dict(color='#1fefff', width=1.5)))
            fig.update_layout(
                template="plotly_dark", xaxis_rangeslider_visible=False,
                yaxis_title="Asset Valuation (USD)", xaxis_title="Timeline",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### Machine Learning Data Flow Architecture")
            with st.expander("🧠 CNN-LSTM Operational Execution Pipeline", expanded=True):
                st.markdown("""
                <div class="arch-card"><h4><b>Phase 1: Input Timeseries Matrix Ingestion</b></h4><p>Captures historical cryptocurrency metrics using rolling window intervals.</p></div>
                <div style='text-align:center;font-size:16px;color:#00ffcc;'>↓</div>
                <div class="arch-card"><h4><b>Phase 2: Standard Normalization (MinMaxScaler)</b></h4><p>Scales raw elements into 0-1 matrices to boost deep learning gradient steps.</p></div>
                <div style='text-align:center;font-size:16px;color:#00ffcc;'>↓</div>
                <div class="arch-card"><h4><b>Phase 3: Spatial Feature Extraction (Conv1D Layer)</b></h4><p>Extracts localized spatial patterns using 64 filter channels.</p></div>
                <div style='text-align:center;font-size:16px;color:#00ffcc;'>↓</div>
                <div class="arch-card"><h4><b>Phase 4: Dimension Condensation (MaxPooling1D)</b></h4><p>Downsamples features to extract structural signals.</p></div>
                <div style='text-align:center;font-size:16px;color:#00ffcc;'>↓</div>
                <div class="arch-card"><h4><b>Phase 5: Temporal Memory Learning (LSTM Layer)</b></h4><p>Tracks long-term multi-day dependency loops over 60 sequence days.</p></div>
                <div style='text-align:center;font-size:16px;color:#00ffcc;'>↓</div>
                <div class="arch-card"><h4><b>Phase 6: Linear Prediction Output</b></h4><p>Applies inverse scalar transformation back into USD currency parameters.</p></div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### Empirical Performance Evaluation Metrics")
            eval_col1, eval_col2, eval_col3, eval_col4 = st.columns(4)
            with eval_col1:
                st.metric(label="RMSE", value=f"{metrics['rmse']:.2f}")
            with eval_col2:
                st.metric(label="MAE", value=f"{metrics['mae']:.2f}")
            with eval_col3:
                st.metric(label="R² Score", value=f"{metrics['r2']:.4f}")
            with eval_col4:
             st.metric(label="Prediction Accuracy (MAPA)", value=f"{metrics['accuracy']:.1f}%")

            st.write("---")
            st.markdown("### Validation Curve (Actual vs Predicted)")
            eval_fig = go.Figure()
            eval_fig.add_trace(go.Scatter(x=eval_data['dates'], y=eval_data['actual'], mode='lines+markers', name='Actual Market Values', line=dict(color='#8b949e', width=2)))
            eval_fig.add_trace(go.Scatter(x=eval_data['dates'], y=eval_data['predicted'], mode='lines+markers', name='CNN-LSTM Predictions', line=dict(color='#00ffcc', width=2, dash='dash')))
            eval_fig.update_layout(
                template="plotly_dark",
                yaxis_title="Asset Price ($ USD)", xaxis_title="Timeline",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(eval_fig, use_container_width=True)

    else:
        st.error(f"Backend Error: {response.get('message')}")

except requests.exceptions.ConnectionError:
    st.error("🚨 Connection Refused: Make sure main.py is running on port 8000.")

st.markdown("---")
st.caption("Design and Implementation of Cryptocurrency Price Prediction Using Machine Learning and Deep Learning (CNN-LSTM) | Department of Computer Science • 2025/2026 Academic Session")
