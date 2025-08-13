#!/usr/bin/env python3
"""
Test script for the Cryptocurrency Forecasting API
"""

import requests
import json
import time
from datetime import datetime

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Cryptocurrency Forecasting API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test 3: Historical analysis
    print("\n3. Testing historical analysis...")
    try:
        response = requests.get(f"{base_url}/historical/BTC", timeout=30)
        if response.status_code == 200:
            print("✅ Historical analysis passed")
            data = response.json()
            print(f"   Data points: {data['data_points']}")
            print(f"   Date range: {data['date_range']['start']} to {data['date_range']['end']}")
        else:
            print(f"❌ Historical analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Historical analysis failed: {e}")
    
    # Test 4: Forecast generation
    print("\n4. Testing forecast generation...")
    try:
        payload = {
            "cryptocurrency": "BTC",
            "forecast_days": 3,  # Shorter forecast for faster testing
            "test_size": 0.2
        }
        response = requests.post(f"{base_url}/forecast", json=payload, timeout=60)
        if response.status_code == 200:
            print("✅ Forecast generation passed")
            data = response.json()
            print(f"   Cryptocurrency: {data['cryptocurrency']}")
            print(f"   Forecast days: {data['forecast_days']}")
            print(f"   Price forecast: {data['price_forecast']}")
            print(f"   Best models: {data['best_models']}")
        else:
            print(f"❌ Forecast generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Forecast generation failed: {e}")
    
    # Test 5: Model metrics
    print("\n5. Testing model metrics...")
    try:
        response = requests.get(f"{base_url}/metrics/BTC", timeout=30)
        if response.status_code == 200:
            print("✅ Model metrics passed")
            data = response.json()
            print(f"   ARIMA RMSE: {data['arima_metrics']['rmse']:.4f}")
            print(f"   ARIMA MAPE: {data['arima_metrics']['mape']:.2f}%")
            print(f"   Best volatility model: {data['model_comparison']['best_volatility_model']}")
        else:
            print(f"❌ Model metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model metrics failed: {e}")
    
    # Test 6: Cache status
    print("\n6. Testing cache status...")
    try:
        response = requests.get(f"{base_url}/cache/status", timeout=10)
        if response.status_code == 200:
            print("✅ Cache status passed")
            data = response.json()
            print(f"   Total cached: {data['total_cached']}")
            print(f"   Cached models: {data['cached_models']}")
        else:
            print(f"❌ Cache status failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cache status failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    return True

if __name__ == "__main__":
    # Wait a moment for the API to start
    print("⏳ Waiting for API to start...")
    time.sleep(5)
    
    # Run tests
    success = test_api()
    
    if success:
        print("\n✅ All tests passed! The API is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the API logs.")
