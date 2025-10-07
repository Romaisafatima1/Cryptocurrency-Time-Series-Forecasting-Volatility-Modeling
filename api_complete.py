#!/usr/bin/env python3
"""
Complete Cryptocurrency Forecasting API
Week 8: Backend & API Integration - COMPLETE VERSION
Includes all endpoints with full ARIMA, GARCH, EGARCH, and LSTM integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cryptocurrency Forecasting API - Complete",
    description="Comprehensive API for cryptocurrency price and volatility forecasting using ARIMA, GARCH, EGARCH, and LSTM models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# Pydantic Models
# =======================

class ForecastRequest(BaseModel):
    cryptocurrency: str = Field(..., description="Cryptocurrency symbol (BTC or ETH)")
    forecast_days: int = Field(7, ge=1, le=90, description="Number of days to forecast")
    test_size: float = Field(0.2, ge=0.1, le=0.4, description="Test set size ratio")
    confidence_level: float = Field(0.95, ge=0.80, le=0.99, description="Confidence level for intervals")
    models: Optional[List[str]] = Field(None, description="Specific models to run (ARIMA, GARCH, EGARCH, LSTM)")

class ForecastResponse(BaseModel):
    cryptocurrency: str
    forecast_days: int
    price_forecast: Dict[str, List[float]]
    price_confidence_intervals: Dict[str, List[Dict[str, float]]]
    volatility_forecast: Dict[str, List[float]]
    model_metrics: Dict[str, Any]
    best_models: Dict[str, str]
    residual_statistics: Dict[str, Any]
    timestamp: str

class ModelComparisonResponse(BaseModel):
    cryptocurrencies: List[str]
    models_compared: List[str]
    metrics_summary: Dict[str, Any]
    best_overall_models: Dict[str, str]
    timestamp: str

class HistoricalAnalysisResponse(BaseModel):
    cryptocurrency: str
    data_points: int
    date_range: Dict[str, str]
    price_statistics: Dict[str, float]
    volatility_statistics: Dict[str, float]
    returns_statistics: Dict[str, float]
    technical_indicators: Dict[str, Any]
    timestamp: str

# =======================
# Model Cache
# =======================

model_cache = {
    "forecasts": {},
    "metrics": {},
    "last_update": {}
}

# =======================
# Utility Functions
# =======================

def load_data(cryptocurrency: str) -> pd.DataFrame:
    """Load cryptocurrency data"""
    try:
        file_map = {
            "BTC": "data/raw/btc_data.csv",
            "ETH": "data/raw/eth_data.csv"
        }
        
        if cryptocurrency.upper() not in file_map:
            raise ValueError(f"Cryptocurrency must be BTC or ETH, got {cryptocurrency}")
        
        df = pd.read_csv(file_map[cryptocurrency.upper()], parse_dates=['timestamp'], index_col='timestamp')
        logger.info(f"Loaded {len(df)} data points for {cryptocurrency}")
        return df
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found for {cryptocurrency}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")

def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape)
    }

def generate_arima_forecast(df, forecast_days, test_size):
    """Generate ARIMA price forecast"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        prices = df['price'].values
        split_idx = int(len(prices) * (1 - test_size))
        train, test = prices[:split_idx], prices[split_idx:]
        
        # Fit ARIMA model
        model = ARIMA(train, order=(2, 1, 1))
        fitted_model = model.fit()
        
        # Generate forecasts
        forecast = fitted_model.forecast(steps=forecast_days)
        
        # Get test set predictions for metrics
        test_predictions = fitted_model.forecast(steps=len(test))
        metrics = calculate_metrics(test, test_predictions)
        metrics['bic'] = float(fitted_model.bic)
        metrics['aic'] = float(fitted_model.aic)
        metrics['order'] = "(2,1,1)"
        
        # Calculate confidence intervals
        forecast_std = np.std(train - fitted_model.fittedvalues)
        confidence_intervals = []
        for i, pred in enumerate(forecast):
            expanding_std = forecast_std * np.sqrt(i + 1)
            confidence_intervals.append({
                "lower": float(pred - 1.96 * expanding_std),
                "upper": float(pred + 1.96 * expanding_std)
            })
        
        return {
            "forecast": [float(x) for x in forecast],
            "confidence_intervals": confidence_intervals,
            "metrics": metrics,
            "residuals": (train - fitted_model.fittedvalues).tolist()
        }
    
    except Exception as e:
        logger.error(f"ARIMA forecast error: {str(e)}")
        return None

def generate_garch_forecast(df, forecast_days, test_size, model_type="GARCH"):
    """Generate GARCH/EGARCH volatility forecast"""
    try:
        from arch import arch_model
        
        # Calculate returns
        returns = df['price'].pct_change().dropna() * 100
        split_idx = int(len(returns) * (1 - test_size))
        train_returns = returns.iloc[:split_idx]
        test_returns = returns.iloc[split_idx:]
        
        # Fit GARCH/EGARCH model
        if model_type == "EGARCH":
            model = arch_model(train_returns, vol='EGARCH', p=1, q=1)
        else:
            model = arch_model(train_returns, vol='Garch', p=1, q=1)
        
        fitted_model = model.fit(disp='off')
        
        # Generate volatility forecast
        forecast = fitted_model.forecast(horizon=forecast_days)
        volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
        
        # Calculate metrics on test set
        test_forecast = fitted_model.forecast(horizon=len(test_returns))
        test_volatility = np.sqrt(test_forecast.variance.values[-1, :])
        actual_volatility = test_returns.rolling(window=5).std().dropna().values[:len(test_volatility)]
        
        metrics = calculate_metrics(actual_volatility, test_volatility)
        metrics['aic'] = float(fitted_model.aic)
        metrics['bic'] = float(fitted_model.bic)
        
        return {
            "forecast": [float(x) for x in volatility_forecast],
            "metrics": metrics,
            "model_type": model_type
        }
    
    except Exception as e:
        logger.error(f"{model_type} forecast error: {str(e)}")
        return None

def generate_lstm_forecast(df, forecast_days, test_size):
    """Generate LSTM price forecast"""
    try:
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        from tensorflow import keras
        
        prices = df['price'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        lookback = 30
        X, y = [], []
        for i in range(len(scaled_prices) - lookback):
            X.append(scaled_prices[i:i+lookback])
            y.append(scaled_prices[i+lookback])
        
        X, y = np.array(X), np.array(y)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build and train model
        model = keras.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0, validation_split=0.1)
        
        # Generate forecast
        last_sequence = scaled_prices[-lookback:]
        forecast = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            pred = model.predict(current_sequence.reshape(1, lookback, 1), verbose=0)
            forecast.append(pred[0, 0])
            current_sequence = np.vstack([current_sequence[1:], pred])
        
        forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        test_pred = model.predict(X_test, verbose=0)
        test_pred_inv = scaler.inverse_transform(test_pred)
        y_test_inv = scaler.inverse_transform(y_test)
        metrics = calculate_metrics(y_test_inv, test_pred_inv)
        
        return {
            "forecast": [float(x) for x in forecast],
            "metrics": metrics
        }
    
    except Exception as e:
        logger.error(f"LSTM forecast error: {str(e)}")
        return None

# =======================
# API Endpoints
# =======================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cryptocurrency Forecasting API - Complete Version",
        "version": "2.0.0",
        "features": {
            "models": ["ARIMA", "GARCH", "EGARCH", "LSTM"],
            "cryptocurrencies": ["BTC", "ETH"],
            "capabilities": [
                "Price forecasting",
                "Volatility modeling",
                "Confidence intervals",
                "Model comparison",
                "Historical analysis",
                "Residual diagnostics"
            ]
        },
        "endpoints": {
            "/forecast": "POST - Get comprehensive forecasts",
            "/metrics/{crypto}": "GET - Get model performance metrics",
            "/historical/{crypto}": "GET - Get historical data analysis",
            "/compare": "GET - Compare all models",
            "/models/{crypto}/{model}": "GET - Get specific model forecast",
            "/residuals/{crypto}/{model}": "GET - Get residual analysis",
            "/health": "GET - API health check",
            "/cache/status": "GET - Cache status",
            "/cache/clear": "DELETE - Clear cache"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_available": ["ARIMA", "GARCH", "EGARCH", "LSTM"],
        "cached_forecasts": len(model_cache["forecasts"]),
        "uptime": "running"
    }

@app.post("/forecast", response_model=ForecastResponse)
async def get_comprehensive_forecast(request: ForecastRequest):
    """
    Get comprehensive forecast from all models
    """
    try:
        crypto = request.cryptocurrency.upper()
        df = load_data(crypto)
        
        # Determine which models to run
        models_to_run = request.models if request.models else ["ARIMA", "GARCH", "EGARCH", "LSTM"]
        
        price_forecasts = {}
        confidence_intervals = {}
        volatility_forecasts = {}
        all_metrics = {}
        residuals = {}
        
        # Generate ARIMA forecast
        if "ARIMA" in models_to_run:
            logger.info(f"Generating ARIMA forecast for {crypto}...")
            arima_result = generate_arima_forecast(df, request.forecast_days, request.test_size)
            if arima_result:
                price_forecasts["ARIMA"] = arima_result["forecast"]
                confidence_intervals["ARIMA"] = arima_result["confidence_intervals"]
                all_metrics["ARIMA"] = arima_result["metrics"]
                residuals["ARIMA"] = arima_result["residuals"]
        
        # Generate GARCH forecast
        if "GARCH" in models_to_run:
            logger.info(f"Generating GARCH forecast for {crypto}...")
            garch_result = generate_garch_forecast(df, request.forecast_days, request.test_size, "GARCH")
            if garch_result:
                volatility_forecasts["GARCH"] = garch_result["forecast"]
                all_metrics["GARCH"] = garch_result["metrics"]
        
        # Generate EGARCH forecast
        if "EGARCH" in models_to_run:
            logger.info(f"Generating EGARCH forecast for {crypto}...")
            egarch_result = generate_garch_forecast(df, request.forecast_days, request.test_size, "EGARCH")
            if egarch_result:
                volatility_forecasts["EGARCH"] = egarch_result["forecast"]
                all_metrics["EGARCH"] = egarch_result["metrics"]
        
        # Generate LSTM forecast
        if "LSTM" in models_to_run:
            logger.info(f"Generating LSTM forecast for {crypto}...")
            lstm_result = generate_lstm_forecast(df, request.forecast_days, request.test_size)
            if lstm_result:
                price_forecasts["LSTM"] = lstm_result["forecast"]
                all_metrics["LSTM"] = lstm_result["metrics"]
        
        # Determine best models
        best_price_model = min(
            [(k, v.get('rmse', float('inf'))) for k, v in all_metrics.items() if k in ['ARIMA', 'LSTM']],
            key=lambda x: x[1],
            default=("ARIMA", 0)
        )[0]
        
        best_volatility_model = min(
            [(k, v.get('aic', float('inf'))) for k, v in all_metrics.items() if k in ['GARCH', 'EGARCH']],
            key=lambda x: x[1],
            default=("GARCH", 0)
        )[0]
        
        # Calculate residual statistics
        residual_stats = {}
        for model, res in residuals.items():
            residual_stats[model] = {
                "mean": float(np.mean(res)),
                "std": float(np.std(res)),
                "skewness": float(pd.Series(res).skew()),
                "kurtosis": float(pd.Series(res).kurtosis())
            }
        
        # Cache results
        cache_key = f"{crypto}_{request.forecast_days}"
        model_cache["forecasts"][cache_key] = {
            "price_forecasts": price_forecasts,
            "volatility_forecasts": volatility_forecasts,
            "timestamp": datetime.now()
        }
        
        return ForecastResponse(
            cryptocurrency=crypto,
            forecast_days=request.forecast_days,
            price_forecast=price_forecasts,
            price_confidence_intervals=confidence_intervals,
            volatility_forecast=volatility_forecasts,
            model_metrics=all_metrics,
            best_models={
                "price": best_price_model,
                "volatility": best_volatility_model
            },
            residual_statistics=residual_stats,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/metrics/{cryptocurrency}")
async def get_model_metrics(cryptocurrency: str):
    """Get detailed model performance metrics"""
    try:
        crypto = cryptocurrency.upper()
        df = load_data(crypto)
        
        # Generate forecasts if not cached
        cache_key = f"{crypto}_7"
        if cache_key not in model_cache["forecasts"]:
            request = ForecastRequest(cryptocurrency=crypto, forecast_days=7)
            await get_comprehensive_forecast(request)
        
        return {
            "cryptocurrency": crypto,
            "message": "Metrics retrieved successfully",
            "cached_data": model_cache["forecasts"].get(cache_key, {}),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.get("/historical/{cryptocurrency}", response_model=HistoricalAnalysisResponse)
async def get_historical_analysis(cryptocurrency: str):
    """Get comprehensive historical data analysis"""
    try:
        crypto = cryptocurrency.upper()
        df = load_data(crypto)
        
        # Price statistics
        price_stats = {
            "mean": float(df['price'].mean()),
            "std": float(df['price'].std()),
            "min": float(df['price'].min()),
            "max": float(df['price'].max()),
            "median": float(df['price'].median()),
            "q25": float(df['price'].quantile(0.25)),
            "q75": float(df['price'].quantile(0.75))
        }
        
        # Calculate returns and volatility
        returns = df['price'].pct_change().dropna()
        volatility_stats = {
            "daily_volatility": float(returns.std()),
            "annualized_volatility": float(returns.std() * np.sqrt(365)),
            "max_drawdown": float((df['price'] / df['price'].cummax() - 1).min()),
            "positive_days": int((returns > 0).sum()),
            "negative_days": int((returns < 0).sum())
        }
        
        # Returns statistics
        returns_stats = {
            "mean_return": float(returns.mean()),
            "median_return": float(returns.median()),
            "max_return": float(returns.max()),
            "min_return": float(returns.min()),
            "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
        }
        
        # Technical indicators
        technical = {
            "current_price": float(df['price'].iloc[-1]),
            "ma_7": float(df['price'].rolling(7).mean().iloc[-1]),
            "ma_30": float(df['price'].rolling(30).mean().iloc[-1]),
            "rsi_14": float(calculate_rsi(df['price'], 14)),
            "price_change_7d": float((df['price'].iloc[-1] - df['price'].iloc[-7]) / df['price'].iloc[-7] * 100),
            "price_change_30d": float((df['price'].iloc[-1] - df['price'].iloc[-30]) / df['price'].iloc[-30] * 100)
        }
        
        return HistoricalAnalysisResponse(
            cryptocurrency=crypto,
            data_points=len(df),
            date_range={
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat()
            },
            price_statistics=price_stats,
            volatility_statistics=volatility_stats,
            returns_statistics=returns_stats,
            technical_indicators=technical,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical analysis failed: {str(e)}")

@app.get("/compare", response_model=ModelComparisonResponse)
async def compare_all_models():
    """Compare all models across both cryptocurrencies"""
    try:
        results = {}
        
        # Generate forecasts for both cryptocurrencies
        for crypto in ["BTC", "ETH"]:
            request = ForecastRequest(cryptocurrency=crypto, forecast_days=7)
            forecast = await get_comprehensive_forecast(request)
            results[crypto] = forecast
        
        # Compile comparison metrics
        metrics_summary = {}
        for crypto, forecast in results.items():
            metrics_summary[crypto] = {
                "best_price_model": forecast.best_models["price"],
                "best_volatility_model": forecast.best_models["volatility"],
                "all_metrics": forecast.model_metrics
            }
        
        # Determine overall best models
        all_price_rmse = {}
        all_vol_aic = {}
        
        for crypto, data in metrics_summary.items():
            for model, metrics in data["all_metrics"].items():
                if model in ["ARIMA", "LSTM"]:
                    all_price_rmse[f"{crypto}_{model}"] = metrics.get("rmse", float('inf'))
                elif model in ["GARCH", "EGARCH"]:
                    all_vol_aic[f"{crypto}_{model}"] = metrics.get("aic", float('inf'))
        
        best_overall = {
            "best_price_model_overall": min(all_price_rmse.items(), key=lambda x: x[1])[0] if all_price_rmse else "N/A",
            "best_volatility_model_overall": min(all_vol_aic.items(), key=lambda x: x[1])[0] if all_vol_aic else "N/A"
        }
        
        return ModelComparisonResponse(
            cryptocurrencies=["BTC", "ETH"],
            models_compared=["ARIMA", "GARCH", "EGARCH", "LSTM"],
            metrics_summary=metrics_summary,
            best_overall_models=best_overall,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@app.get("/models/{cryptocurrency}/{model}")
async def get_specific_model_forecast(cryptocurrency: str, model: str, forecast_days: int = 7):
    """Get forecast from a specific model"""
    try:
        valid_models = ["ARIMA", "GARCH", "EGARCH", "LSTM"]
        if model.upper() not in valid_models:
            raise HTTPException(status_code=400, detail=f"Model must be one of {valid_models}")
        
        request = ForecastRequest(
            cryptocurrency=cryptocurrency,
            forecast_days=forecast_days,
            models=[model.upper()]
        )
        
        forecast = await get_comprehensive_forecast(request)
        
        return {
            "cryptocurrency": forecast.cryptocurrency,
            "model": model.upper(),
            "forecast": forecast.price_forecast.get(model.upper()) or forecast.volatility_forecast.get(model.upper()),
            "metrics": forecast.model_metrics.get(model.upper()),
            "timestamp": forecast.timestamp
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model forecast failed: {str(e)}")

@app.get("/residuals/{cryptocurrency}/{model}")
async def get_residual_analysis(cryptocurrency: str, model: str):
    """Get residual analysis for a specific model"""
    try:
        crypto = cryptocurrency.upper()
        model = model.upper()
        
        if model not in ["ARIMA", "LSTM"]:
            raise HTTPException(status_code=400, detail="Residual analysis only available for ARIMA and LSTM")
        
        df = load_data(crypto)
        
        # Generate forecast to get residuals
        if model == "ARIMA":
            result = generate_arima_forecast(df, 7, 0.2)
        else:
            result = generate_lstm_forecast(df, 7, 0.2)
        
        if not result or "residuals" not in result:
            raise HTTPException(status_code=500, detail="Could not generate residuals")
        
        residuals = result["residuals"]
        
        # Statistical tests
        from scipy import stats
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
        
        return {
            "cryptocurrency": crypto,
            "model": model,
            "residual_statistics": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "skewness": float(pd.Series(residuals).skew()),
                "kurtosis": float(pd.Series(residuals).kurtosis()),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            },
            "normality_test": {
                "shapiro_wilk_statistic": float(shapiro_stat),
                "shapiro_wilk_p_value": float(shapiro_p),
                "is_normal": bool(shapiro_p > 0.05)
            },
            "residuals_sample": residuals[:100],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Residual analysis failed: {str(e)}")

@app.get("/cache/status")
async def get_cache_status():
    """Get cache status"""
    cache_info = {}
    for key, value in model_cache["forecasts"].items():
        cache_info[key] = {
            "timestamp": value["timestamp"].isoformat(),
            "age_minutes": (datetime.now() - value["timestamp"]).total_seconds() / 60
        }
    
    return {
        "cached_forecasts": list(model_cache["forecasts"].keys()),
        "cache_details": cache_info,
        "total_cached": len(model_cache["forecasts"]),
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached results"""
    global model_cache
    cleared_count = len(model_cache["forecasts"])
    model_cache = {
        "forecasts": {},
        "metrics": {},
        "last_update": {}
    }
    return {
        "message": "Cache cleared successfully",
        "items_cleared": cleared_count,
        "timestamp": datetime.now().isoformat()
    }

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50.0

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Complete Cryptocurrency Forecasting API...")
    print("📊 Models: ARIMA, GARCH, EGARCH, LSTM")
    print("💰 Cryptocurrencies: BTC, ETH")
    print("\n🌐 API available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)