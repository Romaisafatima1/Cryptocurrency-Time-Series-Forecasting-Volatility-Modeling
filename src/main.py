import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt

# Load local BTC and ETH data
btc_df = pd.read_csv('data/raw/btc_data.csv', parse_dates=['timestamp'], index_col='timestamp')
eth_df = pd.read_csv('data/raw/eth_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Helper: Forecast price with ARIMA
def forecast_arima(series, steps=7):
    model = ARIMA(series, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Helper: Forecast volatility with GARCH (on log returns)
def forecast_garch(series, steps=7):
    log_returns = np.log(series).diff().dropna()
    model = arch_model(log_returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    forecasts = model_fit.forecast(horizon=steps)
    # Return the forecasted volatility (standard deviation)
    return np.sqrt(forecasts.variance.values[-1, :])

# Forecast for BTC
btc_price_forecast = forecast_arima(btc_df['price'])
btc_vol_forecast = forecast_garch(btc_df['price'])

# Forecast for ETH
eth_price_forecast = forecast_arima(eth_df['price'])
eth_vol_forecast = forecast_garch(eth_df['price'])

print('BTC Price Forecast (next 7 days):')
print(btc_price_forecast)
print('BTC Volatility Forecast (next 7 days, std of log returns):')
print(btc_vol_forecast)

print('ETH Price Forecast (next 7 days):')
print(eth_price_forecast)
print('ETH Volatility Forecast (next 7 days, std of log returns):')
print(eth_vol_forecast)
