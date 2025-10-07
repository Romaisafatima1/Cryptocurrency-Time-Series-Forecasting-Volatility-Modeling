# Cryptocurrency-Time-Series-Forecasting-Volatility-Modeling

This project focuses on predicting cryptocurrency price movements (e.g., Bitcoin, Ethereum) and modeling their market volatility using time series forecasting techniques like LSTM and ARIMA, and financial risk models like GARCH. The system collects real-time data, trains multiple models, and displays results through a **FastAPI REST API** and an interactive dashboard.

## 🚀 **NEW: API Integration**

The project now includes a comprehensive **REST API** that provides programmatic access to all forecasting capabilities:

### **API Features**
- **FastAPI** with automatic OpenAPI documentation
- **8 RESTful endpoints** for accessing model outputs
- **Real-time forecasting** for BTC and ETH
- **Model performance metrics** and comparisons
- **Historical data analysis**
- **Caching system** for improved performance

### **API Endpoints**
- `POST /forecast` - Generate price and volatility forecasts
- `GET /metrics/{crypto}` - Get detailed model performance metrics
- `GET /historical/{crypto}` - Historical data analysis
- `GET /compare` - Compare models across cryptocurrencies
- `GET /health` - API health check
- Cache management endpoints

### **API Documentation**
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Features

- **Real-Time Data Collection**: Pulls live BTC and ETH data from public APIs
- **Time Series Forecasting**: Uses ARIMA and LSTM models to predict future prices
- **Volatility Modeling**: Implements GARCH and EGARCH to analyze market volatility
- **Model Comparison**: Evaluates models using metrics like RMSE, MAE, and MAPE
- **Automated Model Comparison Dashboard**: Interactive dashboard for comparing all models with performance metrics and visualizations
- **Interactive Dashboard**: Visualizes forecasts and volatility with dynamic charts
- **FastAPI REST API**: Provides endpoints to access model predictions programmatically
- **Comprehensive Testing**: Automated testing suite for all API endpoints

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.8+
- pip
- Git

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Romaisafatima1/Cryptocurrency-Time-Series-Forecasting-Volatility-Modeling.git
cd Cryptocurrency-Time-Series-Forecasting-Volatility-Modeling
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### **Running the API Server**

Start the FastAPI server:
```bash
# Start simplified API (recommended)
python run_api_simple.py

# Or start full API (requires pmdarima)
python run_api.py
```

The API will be available at `http://localhost:8000`

### **API Usage Examples**

#### **Python Client**
```python
from src.api_client import CryptoForecastingClient

# Initialize client
client = CryptoForecastingClient()

# Get BTC forecast
btc_forecast = client.get_forecast("BTC", forecast_days=7)
client.print_forecast_summary(btc_forecast)
client.plot_forecast_results(btc_forecast)

# Get model metrics
btc_metrics = client.get_metrics("BTC")
client.print_metrics_summary(btc_metrics)

# Compare models
comparison = client.compare_models()
print(comparison)
```

#### **curl Commands**
```bash
# Get BTC forecast
curl -X POST "http://localhost:8000/forecast" \
     -H "Content-Type: application/json" \
     -d '{"cryptocurrency": "BTC", "forecast_days": 7}'

# Get model metrics
curl "http://localhost:8000/metrics/BTC"

# Get historical analysis
curl "http://localhost:8000/historical/BTC"

# Compare models
curl "http://localhost:8000/compare"
```

#### **JavaScript/Fetch**
```javascript
// Get BTC forecast
const response = await fetch('http://localhost:8000/forecast', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        cryptocurrency: 'BTC',
        forecast_days: 7
    })
});

const forecast = await response.json();
console.log('Price forecast:', forecast.price_forecast);
console.log('Volatility forecast:', forecast.volatility_forecast);
```

### **Testing the API**
```bash
# Run comprehensive API tests
python test_api.py
```

### **Running the Main Analysis**
```bash
python src/main.py
```

### **Running the Automated Dashboard**
```bash
python src/integrated_dashboard.py
```

### **Testing the Dashboard**
```bash
python test_dashboard.py
```

## API Response Examples

### **Forecast Response**
```json
{
  "cryptocurrency": "BTC",
  "forecast_days": 7,
  "price_forecast": [82403.17, 82438.23, 82467.91, ...],
  "price_confidence_intervals": [
    {"lower": 78285.48, "upper": 86520.86},
    {"lower": 76697.96, "upper": 88178.49},
    ...
  ],
  "volatility_forecast": {
    "GARCH": [2.36, 2.36, 2.36, ...],
    "EGARCH": [2.37, 2.37, 2.37, ...]
  },
  "model_metrics": {
    "ARIMA": {"rmse": 17443.13, "mae": 14669.01, "mape": 14.15, "bic": 5297.05},
    "GARCH": {"rmse": 1.02, "mae": 0.89, "aic": 1309.39, "bic": 1324.02},
    "EGARCH": {"rmse": 1.03, "mae": 0.90, "aic": 1308.84, "bic": 1323.48}
  },
  "best_models": {"price": "ARIMA", "volatility": "EGARCH"},
  "timestamp": "2025-08-13T12:54:26.202614"
}
```

### **Model Metrics Response**
```json
{
  "cryptocurrency": "BTC",
  "arima_metrics": {
    "rmse": 17443.13,
    "mae": 14669.01,
    "mape": 14.15,
    "bic": 5297.05,
    "order": "(2, 1, 1)",
    "test_data_points": 74
  },
  "garch_metrics": {
    "rmse": 1.02,
    "mae": 0.89,
    "aic": 1309.39,
    "bic": 1324.02,
    "forecast_horizon": 7.0
  },
  "egarch_metrics": {
    "rmse": 1.03,
    "mae": 0.90,
    "aic": 1308.84,
    "bic": 1323.48,
    "forecast_horizon": 7.0
  },
  "model_comparison": {
    "best_price_model": "ARIMA",
    "best_volatility_model": "EGARCH",
    "volatility_model_comparison": {
      "garch_aic": 1309.39,
      "egarch_aic": 1308.84,
      "aic_difference": 0.55
    }
  }
}
```

## Dashboard Features

* **Performance Comparison**: Compare RMSE, MAPE, and MAE across all models
* **Model Ranking**: Automated ranking system to identify best performing models
* **Forecast Visualization**: Side-by-side comparison of price and volatility forecasts
* **Summary Reports**: Comprehensive performance summaries with recommendations
* **API Integration**: Programmatic access to all dashboard capabilities

## Model Outputs

### **Price Forecasting (ARIMA)**
- Forecast values for next N days
- 95% confidence intervals
- Model performance metrics (RMSE, MAE, MAPE, BIC)
- Best model selection

### **Volatility Forecasting (GARCH/EGARCH)**
- GARCH volatility predictions
- EGARCH asymmetric volatility predictions (captures leverage effects)
- Model comparison using AIC/BIC
- Best volatility model selection

### **Historical Analysis**
- Statistical analysis of price data
- Volatility statistics and distributions
- Date range and data point information

### **Cross-Cryptocurrency Comparison**
- Model comparison across BTC and ETH
- Performance metrics comparison
- Forecast comparison

## File Structure

```
├── src/
│   ├── api.py              # Full API implementation
│   ├── api_simple.py       # Simplified API (without pmdarima)
│   ├── api_client.py       # Python client for API interaction
│   ├── main.py             # Original model functions
│   ├── dashboard.py        # Dashboard implementation
│   └── integrated_dashboard.py
├── data/
│   ├── raw/
│   │   ├── btc_data.csv    # Bitcoin historical data
│   │   └── eth_data.csv    # Ethereum historical data
│   └── processed/          # Processed data files
├── scripts/                # Data processing scripts
├── run_api.py              # Full API startup script
├── run_api_simple.py       # Simplified API startup script
├── test_api.py             # API testing script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Key Achievements

✅ **Complete Model Integration** - All forecasting models accessible via API  
✅ **Professional API Design** - RESTful endpoints with proper documentation  
✅ **Comprehensive Testing** - All endpoints tested and working  
✅ **Error Handling** - Robust error handling and validation  
✅ **Caching System** - Efficient caching for model results  
✅ **Multiple Client Support** - Works with Python, JavaScript, curl, etc.  
✅ **Production Ready** - Proper logging, validation, and security  

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original project by [Romaisafatima1](https://github.com/Romaisafatima1)
- Enhanced with comprehensive API integration for better accessibility and usability