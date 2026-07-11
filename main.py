import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = FastAPI(title="Crypto Prediction API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    scaler_meta = np.load('scaler_meta.npy')
    scaler_min = scaler_meta[0]
    scaler_scale = scaler_meta[1]
except FileNotFoundError:
    raise RuntimeError("Missing 'scaler_meta.npy'. Please run your training script first.")

try:
    model = load_model('crypto_lstm_model.keras')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.get("/")
def root():
    return {"message": "Crypto Prediction API is running seamlessly!"}

@app.get("/predict")
def predict_price():
    try:
        df = yf.download('BTC-USD', period='200d')
        close_prices = df['Close'].dropna().values
        close_prices = np.array(close_prices).flatten()
        dates = df.index.dropna().strftime('%Y-%m-%d').tolist()

        if len(close_prices) < 90:
            return {"status": "error", "message": "Insufficient market data returned from API."}

        # --- LIVE 24H FORECAST PIPELINE ---
        recent_60_days = close_prices[-60:]
        current_price = float(np.array(close_prices[-1]).flatten()[0])

        scaled_live = (recent_60_days.reshape(-1, 1) * scaler_scale) + scaler_min
        X_live = np.reshape(scaled_live, (1, 60, 1))

        pred_live_scaled = model.predict(X_live)
        predicted_tomorrow = float((pred_live_scaled[0][0] - scaler_min) / scaler_scale)
        direction = "UP" if predicted_tomorrow > current_price else "DOWN"

        # --- HISTORICAL EVALUATION PIPELINE (Last 30 Days) ---
        actual_last_30 = np.array(close_prices[-30:]).flatten()
        dates_last_30 = dates[-30:]
        predicted_last_30 = []

        for i in range(30, 0, -1):
            start_idx = len(close_prices) - i - 60
            end_idx = len(close_prices) - i
            window = close_prices[start_idx:end_idx]
            scaled_window = (window.reshape(-1, 1) * scaler_scale) + scaler_min
            X_window = np.reshape(scaled_window, (1, 60, 1))
            pred_scaled = model.predict(X_window)
            pred_usd = float((pred_scaled[0][0] - scaler_min) / scaler_scale)
            predicted_last_30.append(pred_usd)

        # --- TRUE EMPIRICAL METRICS ---
        rmse = float(np.sqrt(mean_squared_error(actual_last_30, predicted_last_30)))
        mae = float(mean_absolute_error(actual_last_30, predicted_last_30))
        r2 = float(r2_score(actual_last_30, predicted_last_30))

        # MAPA - Mean Absolute Percentage Accuracy (correct metric for regression models)
        mape_errors = [abs((a - p) / a) for a, p in zip(actual_last_30, predicted_last_30)]
        mapa = round((1 - (sum(mape_errors) / len(mape_errors))) * 100, 1)

        return {
            "status": "success",
            "current_price": round(current_price, 2),
            "predicted_tomorrow_price": round(predicted_tomorrow, 2),
            "expected_direction": direction,
            "metrics": {
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "r2": round(r2, 4),
                "accuracy": mapa
            },
            "evaluation_chart": {
                "dates": dates_last_30,
                "actual": [float(x) for x in actual_last_30],
                "predicted": [float(x) for x in predicted_last_30]
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}