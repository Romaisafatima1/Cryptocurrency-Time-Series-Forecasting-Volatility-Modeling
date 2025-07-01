# Cryptocurrency-Time-Series-Forecasting-Volatility-Modeling
This project focuses on predicting cryptocurrency price movements (e.g., Bitcoin, Ethereum) and modeling their market volatility using time series forecasting techniques like LSTM and ARIMA, and financial risk models like GARCH. The system collects real-time data, trains multiple models, and displays results through a Flask API and an interactive dashboard.

#Features

Real-Time Data Collection: Pulls live BTC and ETH data from public APIs.

Time Series Forecasting: Uses ARIMA and LSTM models to predict future prices.

Volatility Modeling: Implements GARCH and EGARCH to analyze market volatility.

Model Comparison: Evaluates models using metrics like RMSE, MAE, and MAPE.

Interactive Dashboard: Visualizes forecasts and volatility with dynamic charts.

Flask API: Provides endpoints to access model predictions programmatically.


#Prerequisites

Before you begin, make sure you have the following installed:

Python 3.8+

pip

Git



#Installation

Clone the repository:

git clone https://github.com/your-username/crypto-forecasting-volatility.git
cd crypto-forecasting-volatility
pip install -r requirements.txt