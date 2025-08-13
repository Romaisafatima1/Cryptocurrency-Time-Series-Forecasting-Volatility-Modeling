from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import model functions from main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    forecast_auto_arima, 
    forecast_garch_models, 
    plot_actual_vs_predicted,
    build_comparison_table
)

app = FastAPI(
    title="Cryptocurrency Forecasting API",
    description="API for cryptocurrency price and volatility forecasting using ARIMA, GARCH, and EGARCH models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ForecastRequest(BaseModel):
    cryptocurrency: str = "BTC"  # BTC or ETH
    forecast_days: int = 7
    test_size: float = 0.2

class ForecastResponse(BaseModel):
    cryptocurrency: str
    forecast_days: int
    price_forecast: List[float]
    price_confidence_intervals: List[Dict[str, float]]
    volatility_forecast: Dict[str, List[float]]
    model_metrics: Dict[str, Any]
    best_models: Dict[str, str]
    timestamp: str

class ModelMetricsResponse(BaseModel):
    cryptocurrency: str
    arima_metrics: Dict[str, float]
    garch_metrics: Dict[str, float]
    egarch_metrics: Dict[str, float]
    model_comparison: Dict[str, Any]

class HistoricalAnalysisResponse(BaseModel):
    cryptocurrency: str
    data_points: int
    date_range: Dict[str, str]
    price_statistics: Dict[str, float]
    volatility_statistics: Dict[str, float]

# Global variables to store model results
model_cache = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cryptocurrency Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "/forecast": "Get price and volatility forecasts",
            "/metrics": "Get model performance metrics",
            "/historical": "Get historical data analysis",
            "/compare": "Compare models across cryptocurrencies",
            "/health": "API health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(model_cache)
    }

@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    """
    Get price and volatility forecasts for a cryptocurrency
    """
    try:
        # Load data based on cryptocurrency
        if request.cryptocurrency.upper() == "BTC":
            df = pd.read_csv('data/raw/btc_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        elif request.cryptocurrency.upper() == "ETH":
            df = pd.read_csv('data/raw/eth_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        else:
            raise HTTPException(status_code=400, detail="Cryptocurrency must be BTC or ETH")
        
        # Generate price forecast using ARIMA
        price_results = forecast_auto_arima(
            df['price'], 
            steps=request.forecast_days, 
            test_size=request.test_size
        )
        
        # Generate volatility forecast using GARCH/EGARCH
        vol_results, returns = forecast_garch_models(
            df['price'], 
            steps=request.forecast_days, 
            test_size=request.test_size
        )
        
        # Prepare confidence intervals
        confidence_intervals = []
        for i in range(len(price_results['forecast'])):
            ci = price_results['confidence_interval'].iloc[i]
            confidence_intervals.append({
                "lower": float(ci.iloc[0]),
                "upper": float(ci.iloc[1])
            })
        
        # Determine best models
        best_vol_model = "GARCH" if vol_results['GARCH']['aic'] < vol_results['EGARCH']['aic'] else "EGARCH"
        
        # Cache results
        cache_key = f"{request.cryptocurrency}_{request.forecast_days}"
        model_cache[cache_key] = {
            "price_results": price_results,
            "vol_results": vol_results,
            "timestamp": datetime.now()
        }
        
        return ForecastResponse(
            cryptocurrency=request.cryptocurrency.upper(),
            forecast_days=request.forecast_days,
            price_forecast=[float(x) for x in price_results['forecast']],
            price_confidence_intervals=confidence_intervals,
            volatility_forecast={
                "GARCH": [float(x) for x in vol_results['GARCH']['forecast']],
                "EGARCH": [float(x) for x in vol_results['EGARCH']['forecast']]
            },
            model_metrics={
                "ARIMA": {
                    "rmse": float(price_results['metrics']['rmse']),
                    "mae": float(price_results['metrics']['mae']),
                    "mape": float(price_results['metrics']['mape']),
                    "bic": float(price_results['metrics']['bic']),
                    "order": price_results['order']
                },
                "GARCH": {
                    "rmse": float(vol_results['GARCH']['rmse']) if not np.isnan(vol_results['GARCH']['rmse']) else None,
                    "mae": float(vol_results['GARCH']['mae']) if not np.isnan(vol_results['GARCH']['mae']) else None,
                    "aic": float(vol_results['GARCH']['aic']),
                    "bic": float(vol_results['GARCH']['bic'])
                },
                "EGARCH": {
                    "rmse": float(vol_results['EGARCH']['rmse']) if not np.isnan(vol_results['EGARCH']['rmse']) else None,
                    "mae": float(vol_results['EGARCH']['mae']) if not np.isnan(vol_results['EGARCH']['mae']) else None,
                    "aic": float(vol_results['EGARCH']['aic']),
                    "bic": float(vol_results['EGARCH']['bic'])
                }
            },
            best_models={
                "price": "ARIMA",
                "volatility": best_vol_model
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/metrics/{cryptocurrency}", response_model=ModelMetricsResponse)
async def get_model_metrics(cryptocurrency: str):
    """
    Get detailed model performance metrics for a cryptocurrency
    """
    try:
        # Check if we have cached results
        cache_key = f"{cryptocurrency}_7"  # Default to 7 days
        if cache_key not in model_cache:
            # Generate forecasts if not cached
            request = ForecastRequest(cryptocurrency=cryptocurrency, forecast_days=7)
            await get_forecast(request)
        
        cached_data = model_cache[cache_key]
        price_results = cached_data["price_results"]
        vol_results = cached_data["vol_results"]
        
        # Prepare detailed metrics
        arima_metrics = {
            "rmse": float(price_results['metrics']['rmse']),
            "mae": float(price_results['metrics']['mae']),
            "mape": float(price_results['metrics']['mape']),
            "bic": float(price_results['metrics']['bic']),
            "order": price_results['order'],
            "test_data_points": len(price_results['test_data'])
        }
        
        garch_metrics = {
            "rmse": float(vol_results['GARCH']['rmse']) if not np.isnan(vol_results['GARCH']['rmse']) else None,
            "mae": float(vol_results['GARCH']['mae']) if not np.isnan(vol_results['GARCH']['mae']) else None,
            "aic": float(vol_results['GARCH']['aic']),
            "bic": float(vol_results['GARCH']['bic']),
            "forecast_horizon": len(vol_results['GARCH']['forecast'])
        }
        
        egarch_metrics = {
            "rmse": float(vol_results['EGARCH']['rmse']) if not np.isnan(vol_results['EGARCH']['rmse']) else None,
            "mae": float(vol_results['EGARCH']['mae']) if not np.isnan(vol_results['EGARCH']['mae']) else None,
            "aic": float(vol_results['EGARCH']['aic']),
            "bic": float(vol_results['EGARCH']['bic']),
            "forecast_horizon": len(vol_results['EGARCH']['forecast'])
        }
        
        # Model comparison
        model_comparison = {
            "best_price_model": "ARIMA",
            "best_volatility_model": "GARCH" if vol_results['GARCH']['aic'] < vol_results['EGARCH']['aic'] else "EGARCH",
            "volatility_model_comparison": {
                "garch_aic": float(vol_results['GARCH']['aic']),
                "egarch_aic": float(vol_results['EGARCH']['aic']),
                "aic_difference": float(vol_results['GARCH']['aic'] - vol_results['EGARCH']['aic'])
            }
        }
        
        return ModelMetricsResponse(
            cryptocurrency=cryptocurrency.upper(),
            arima_metrics=arima_metrics,
            garch_metrics=garch_metrics,
            egarch_metrics=egarch_metrics,
            model_comparison=model_comparison
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.get("/historical/{cryptocurrency}", response_model=HistoricalAnalysisResponse)
async def get_historical_analysis(cryptocurrency: str):
    """
    Get historical data analysis for a cryptocurrency
    """
    try:
        # Load data
        if cryptocurrency.upper() == "BTC":
            df = pd.read_csv('data/raw/btc_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        elif cryptocurrency.upper() == "ETH":
            df = pd.read_csv('data/raw/eth_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        else:
            raise HTTPException(status_code=400, detail="Cryptocurrency must be BTC or ETH")
        
        # Calculate price statistics
        price_stats = {
            "mean": float(df['price'].mean()),
            "std": float(df['price'].std()),
            "min": float(df['price'].min()),
            "max": float(df['price'].max()),
            "median": float(df['price'].median()),
            "skewness": float(df['price'].skew()),
            "kurtosis": float(df['price'].kurtosis())
        }
        
        # Calculate volatility statistics (using log returns)
        log_returns = np.log(df['price']).diff().dropna() * 100
        vol_stats = {
            "mean_volatility": float(log_returns.std()),
            "volatility_std": float(log_returns.rolling(window=30).std().std()),
            "min_volatility": float(log_returns.rolling(window=30).std().min()),
            "max_volatility": float(log_returns.rolling(window=30).std().max()),
            "volatility_skewness": float(log_returns.rolling(window=30).std().skew()),
            "volatility_kurtosis": float(log_returns.rolling(window=30).std().kurtosis())
        }
        
        return HistoricalAnalysisResponse(
            cryptocurrency=cryptocurrency.upper(),
            data_points=len(df),
            date_range={
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat()
            },
            price_statistics=price_stats,
            volatility_statistics=vol_stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical analysis failed: {str(e)}")

@app.get("/compare")
async def compare_models():
    """
    Compare models across both BTC and ETH
    """
    try:
        # Generate forecasts for both cryptocurrencies
        btc_request = ForecastRequest(cryptocurrency="BTC", forecast_days=7)
        eth_request = ForecastRequest(cryptocurrency="ETH", forecast_days=7)
        
        btc_forecast = await get_forecast(btc_request)
        eth_forecast = await get_forecast(eth_request)
        
        # Create comparison summary
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "comparison_summary": {
                "btc_best_price_model": btc_forecast.best_models["price"],
                "btc_best_volatility_model": btc_forecast.best_models["volatility"],
                "eth_best_price_model": eth_forecast.best_models["price"],
                "eth_best_volatility_model": eth_forecast.best_models["volatility"]
            },
            "performance_comparison": {
                "btc_arima_mape": btc_forecast.model_metrics["ARIMA"]["mape"],
                "eth_arima_mape": eth_forecast.model_metrics["ARIMA"]["mape"],
                "btc_garch_aic": btc_forecast.model_metrics["GARCH"]["aic"],
                "eth_garch_aic": eth_forecast.model_metrics["GARCH"]["aic"],
                "btc_egarch_aic": btc_forecast.model_metrics["EGARCH"]["aic"],
                "eth_egarch_aic": eth_forecast.model_metrics["EGARCH"]["aic"]
            },
            "forecast_comparison": {
                "btc_price_forecast": btc_forecast.price_forecast,
                "eth_price_forecast": eth_forecast.price_forecast,
                "btc_volatility_forecast": btc_forecast.volatility_forecast,
                "eth_volatility_forecast": eth_forecast.volatility_forecast
            }
        }
        
        return comparison
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@app.get("/cache/status")
async def get_cache_status():
    """
    Get the status of cached model results
    """
    cache_info = {}
    for key, value in model_cache.items():
        cache_info[key] = {
            "timestamp": value["timestamp"].isoformat(),
            "age_minutes": (datetime.now() - value["timestamp"]).total_seconds() / 60
        }
    
    return {
        "cached_models": list(model_cache.keys()),
        "cache_details": cache_info,
        "total_cached": len(model_cache)
    }

@app.delete("/cache/clear")
async def clear_cache():
    """
    Clear all cached model results
    """
    global model_cache
    model_cache.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
