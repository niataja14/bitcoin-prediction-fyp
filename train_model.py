import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten

print("--- Step 1: Fetching Bitcoin Data ---")
# Download historical Bitcoin data (BTC-USD) from 2020 to present
data = yf.download('BTC-USD', start='2020-01-01', end='2026-01-01')

if data is None or data.empty or 'Close' not in data.columns:
    raise RuntimeError('Failed to fetch BTC-USD close price data from yfinance.')

close_prices_df = data[['Close']].dropna()
if close_prices_df.empty:
    raise RuntimeError('Fetched BTC-USD data contains no valid Close price values.')

close_prices = close_prices_df.values

prediction_days = 60
if len(close_prices) <= prediction_days:
    raise ValueError(f'Not enough data to create training sequences. Need more than {prediction_days} rows.')

print("--- Step 2: Preprocessing and Scaling Data ---")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

X_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
if X_train.size == 0 or y_train.size == 0:
    raise ValueError('No training sequences were created. Check the prediction_days and the fetched data length.')

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print("--- Step 3: Building Hybrid CNN-LSTM Architecture ---")
model = Sequential()

# CNN Layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())

# Reshape for LSTM
model.add(Dense(units=50, activation='relu')) 
model.add(tf.keras.layers.Reshape((50, 1))) 

# LSTM Layer
model.add(LSTM(units=50, return_sequences=False))

# Output Layer
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("--- Step 4: Training the Model (This will take a moment) ---")
model.fit(X_train, y_train, epochs=10, batch_size=32)

print("--- Step 5: Saving the Trained Assets ---")
# Save the model brain
model.save('crypto_lstm_model.keras')

# Save the scaler metadata
np.save('scaler_meta.npy', [scaler.min_[0], scaler.scale_[0]])

print("🎉 Success! Your model and scaler have been trained and saved successfully.")