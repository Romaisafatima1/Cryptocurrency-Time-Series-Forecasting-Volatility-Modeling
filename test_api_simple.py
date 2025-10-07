#!/usr/bin/env python3
"""
Simple Test API for Cryptocurrency Forecasting
This is a simplified version to test the API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json

app = FastAPI(
    title="Cryptocurrency Forecasting API - Test Version",
    description="Simplified API for testing cryptocurrency forecasting endpoints",
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

# Pydantic models
class ForecastRequest(BaseModel):
    cryptocurrency: str = "BTC"
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

# Global cache
model_cache = {}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cryptocurrency Forecasting API - Test Version",
        "version": "1.0.0",
        "endpoints": {
            "/forecast": "Get price and volatility forecasts",
            "/metrics/{crypto}": "Get model performance metrics",
            "/historical/{crypto}": "Get historical data analysis",
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
        "models_loaded": len(model_cache),
        "message": "API is running successfully!"
    }

@app.post("/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    """Get price and volatility forecasts for a cryptocurrency"""
    try:
        # Load data
        if request.cryptocurrency.upper() == "BTC":
            df = pd.read_csv('data/raw/btc_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        elif request.cryptocurrency.upper() == "ETH":
            df = pd.read_csv('data/raw/eth_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        else:
            raise HTTPException(status_code=400, detail="Cryptocurrency must be BTC or ETH")
        
        # Simple mock forecast (last price + small random variation)
        last_price = df['price'].iloc[-1]
        forecast_prices = []
        confidence_intervals = []
        
        for i in range(request.forecast_days):
            # Simple trend-based forecast
            trend = 1 + (i * 0.001)  # Small upward trend
            noise = np.random.normal(0, 0.02)  # 2% noise
            forecast_price = last_price * trend * (1 + noise)
            forecast_prices.append(float(forecast_price))
            
            # Confidence intervals
            confidence_intervals.append({
                "lower": float(forecast_price * 0.95),
                "upper": float(forecast_price * 1.05)
            })
        
        # Mock volatility forecast
        volatility_forecast = {
            "GARCH": [float(np.random.uniform(0.02, 0.05)) for _ in range(request.forecast_days)],
            "EGARCH": [float(np.random.uniform(0.02, 0.05)) for _ in range(request.forecast_days)]
        }
        
        # Mock metrics
        model_metrics = {
            "ARIMA": {
                "rmse": float(np.random.uniform(1000, 5000)),
                "mae": float(np.random.uniform(800, 4000)),
                "mape": float(np.random.uniform(5, 20)),
                "bic": float(np.random.uniform(4000, 6000)),
                "order": "(1,1,1)"
            },
            "GARCH": {
                "rmse": float(np.random.uniform(0.01, 0.03)),
                "mae": float(np.random.uniform(0.008, 0.025)),
                "aic": float(np.random.uniform(-200, -100)),
                "bic": float(np.random.uniform(-180, -80))
            },
            "EGARCH": {
                "rmse": float(np.random.uniform(0.01, 0.03)),
                "mae": float(np.random.uniform(0.008, 0.025)),
                "aic": float(np.random.uniform(-200, -100)),
                "bic": float(np.random.uniform(-180, -80))
            }
        }
        
        # Cache results
        cache_key = f"{request.cryptocurrency}_{request.forecast_days}"
        model_cache[cache_key] = {
            "forecast_prices": forecast_prices,
            "timestamp": datetime.now()
        }
        
        return ForecastResponse(
            cryptocurrency=request.cryptocurrency.upper(),
            forecast_days=request.forecast_days,
            price_forecast=forecast_prices,
            price_confidence_intervals=confidence_intervals,
            volatility_forecast=volatility_forecast,
            model_metrics=model_metrics,
            best_models={
                "price": "ARIMA",
                "volatility": "GARCH"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/metrics/{cryptocurrency}")
async def get_model_metrics(cryptocurrency: str):
    """Get detailed model performance metrics for a cryptocurrency"""
    try:
        # Check if we have cached results
        cache_key = f"{cryptocurrency}_7"
        if cache_key not in model_cache:
            # Generate forecasts if not cached
            request = ForecastRequest(cryptocurrency=cryptocurrency, forecast_days=7)
            await get_forecast(request)
        
        return {
            "cryptocurrency": cryptocurrency.upper(),
            "message": "Metrics retrieved successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@app.get("/historical/{cryptocurrency}")
async def get_historical_analysis(cryptocurrency: str):
    """Get historical data analysis for a cryptocurrency"""
    try:
        # Load data
        if cryptocurrency.upper() == "BTC":
            df = pd.read_csv('data/raw/btc_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        elif cryptocurrency.upper() == "ETH":
            df = pd.read_csv('data/raw/eth_data.csv', parse_dates=['timestamp'], index_col='timestamp')
        else:
            raise HTTPException(status_code=400, detail="Cryptocurrency must be BTC or ETH")
        
        # Calculate basic statistics
        price_stats = {
            "mean": float(df['price'].mean()),
            "std": float(df['price'].std()),
            "min": float(df['price'].min()),
            "max": float(df['price'].max()),
            "median": float(df['price'].median())
        }
        
        return {
            "cryptocurrency": cryptocurrency.upper(),
            "data_points": len(df),
            "date_range": {
                "start": df.index.min().isoformat(),
                "end": df.index.max().isoformat()
            },
            "price_statistics": price_stats,
            "message": "Historical analysis completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical analysis failed: {str(e)}")

@app.get("/compare")
async def compare_models():
    """Compare models across both BTC and ETH"""
    try:
        # Generate forecasts for both cryptocurrencies
        btc_request = ForecastRequest(cryptocurrency="BTC", forecast_days=7)
        eth_request = ForecastRequest(cryptocurrency="ETH", forecast_days=7)
        
        btc_forecast = await get_forecast(btc_request)
        eth_forecast = await get_forecast(eth_request)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "message": "Model comparison completed successfully",
            "btc_forecast_days": btc_forecast.forecast_days,
            "eth_forecast_days": eth_forecast.forecast_days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@app.get("/cache/status")
async def get_cache_status():
    """Get the status of cached model results"""
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
    """Clear all cached model results"""
    global model_cache
    model_cache.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simplified Cryptocurrency Forecasting API...")
    print("📊 Available endpoints:")
    print("  - GET  / : API information")
    print("  - GET  /health : Health check")
    print("  - POST /forecast : Get forecasts")
    print("  - GET  /metrics/{crypto} : Get model metrics")
    print("  - GET  /historical/{crypto} : Get historical analysis")
    print("  - GET  /compare : Compare models")
    print("  - GET  /cache/status : Cache status")
    print("  - DELETE /cache/clear : Clear cache")
    print("\n🌐 API will be available at: http://localhost:8001")
    print("📖 API documentation at: http://localhost:8001/docs")
    print("🔧 Interactive docs at: http://localhost:8001/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8001) 