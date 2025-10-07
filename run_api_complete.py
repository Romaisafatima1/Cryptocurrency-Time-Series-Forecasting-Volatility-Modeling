#!/usr/bin/env python3
"""
Startup script for the Complete Cryptocurrency Forecasting API
"""

import uvicorn
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    print("🚀 Starting Complete Cryptocurrency Forecasting API...")
    print("📊 Models: ARIMA, LSTM, GARCH, EGARCH")
    print("\n🌐 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🔧 Interactive docs at: http://localhost:8000/redoc")
    print("\n" + "="*60)
    
    # Run the complete API server
    uvicorn.run(
        "api_complete:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )