#!/usr/bin/env python3
"""
Startup script for the Simplified Cryptocurrency Forecasting API
"""

import uvicorn
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
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
    print("\n🌐 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🔧 Interactive docs at: http://localhost:8000/redoc")
    print("\n" + "="*60)
    
    # Run the simplified API server
    uvicorn.run(
        "src.api_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
