import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Parameters
LOOKBACK = 30  # Number of past days to use for prediction
EPOCHS = 30
BATCH_SIZE = 16

# Helper to prepare data for LSTM
def prepare_lstm_data(series, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)

# Train and evaluate LSTM for a given asset
def train_lstm(asset_name, csv_path):
    print(f"\n=== Training LSTM for {asset_name} ===")
    df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
    price = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    price_scaled = scaler.fit_transform(price)
    X, y = prepare_lstm_data(price_scaled)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test), verbose=1)
    # Predict
    y_pred = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred).flatten()
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_inv, label='Actual')
    plt.plot(y_pred_inv, label='Predicted')
    plt.title(f'{asset_name} LSTM Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_lstm('BTC', 'data/processed/btc_data_with_features.csv')
    train_lstm('ETH', 'data/processed/eth_data_with_features.csv') 