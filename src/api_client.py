import requests
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class CryptoForecastingClient:
    """
    Client for interacting with the Cryptocurrency Forecasting API
    """
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_forecast(self, cryptocurrency="BTC", forecast_days=7, test_size=0.2):
        """Get price and volatility forecasts"""
        try:
            payload = {
                "cryptocurrency": cryptocurrency,
                "forecast_days": forecast_days,
                "test_size": test_size
            }
            
            response = self.session.post(f"{self.base_url}/forecast", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Forecast request failed: {e}")
            return None
    
    def get_metrics(self, cryptocurrency="BTC"):
        """Get model performance metrics"""
        try:
            response = self.session.get(f"{self.base_url}/metrics/{cryptocurrency}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Metrics request failed: {e}")
            return None
    
    def get_historical_analysis(self, cryptocurrency="BTC"):
        """Get historical data analysis"""
        try:
            response = self.session.get(f"{self.base_url}/historical/{cryptocurrency}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Historical analysis request failed: {e}")
            return None
    
    def compare_models(self):
        """Compare models across cryptocurrencies"""
        try:
            response = self.session.get(f"{self.base_url}/compare")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Model comparison request failed: {e}")
            return None
    
    def get_cache_status(self):
        """Get cache status"""
        try:
            response = self.session.get(f"{self.base_url}/cache/status")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Cache status request failed: {e}")
            return None
    
    def clear_cache(self):
        """Clear model cache"""
        try:
            response = self.session.delete(f"{self.base_url}/cache/clear")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Cache clear request failed: {e}")
            return None
    
    def plot_forecast_results(self, forecast_data):
        """Plot forecast results"""
        if not forecast_data:
            print("No forecast data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{forecast_data["cryptocurrency"]} Forecast Results', fontsize=16)
        
        # Price forecast
        days = range(1, len(forecast_data["price_forecast"]) + 1)
        axes[0, 0].plot(days, forecast_data["price_forecast"], 'b-o', linewidth=2, label='Price Forecast')
        axes[0, 0].set_title('Price Forecast')
        axes[0, 0].set_xlabel('Days Ahead')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Confidence intervals
        lower_bounds = [ci["lower"] for ci in forecast_data["price_confidence_intervals"]]
        upper_bounds = [ci["upper"] for ci in forecast_data["price_confidence_intervals"]]
        axes[0, 1].fill_between(days, lower_bounds, upper_bounds, alpha=0.3, color='blue', label='95% CI')
        axes[0, 1].plot(days, forecast_data["price_forecast"], 'b-o', linewidth=2, label='Price Forecast')
        axes[0, 1].set_title('Price Forecast with Confidence Intervals')
        axes[0, 1].set_xlabel('Days Ahead')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Volatility forecasts
        garch_vol = forecast_data["volatility_forecast"]["GARCH"]
        egarch_vol = forecast_data["volatility_forecast"]["EGARCH"]
        axes[1, 0].plot(days, garch_vol, 'r-s', linewidth=2, label='GARCH')
        axes[1, 0].plot(days, egarch_vol, 'g-^', linewidth=2, label='EGARCH')
        axes[1, 0].set_title('Volatility Forecast Comparison')
        axes[1, 0].set_xlabel('Days Ahead')
        axes[1, 0].set_ylabel('Volatility (%)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Model metrics comparison
        metrics = forecast_data["model_metrics"]
        model_names = list(metrics.keys())
        rmse_values = []
        for model in model_names:
            if "rmse" in metrics[model] and metrics[model]["rmse"] is not None:
                rmse_values.append(metrics[model]["rmse"])
            else:
                rmse_values.append(0)
        
        axes[1, 1].bar(model_names, rmse_values, color=['blue', 'red', 'green'])
        axes[1, 1].set_title('Model RMSE Comparison')
        axes[1, 1].set_ylabel('RMSE (lower is better)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_forecast_summary(self, forecast_data):
        """Print a summary of forecast results"""
        if not forecast_data:
            print("No forecast data available")
            return
        
        print(f"\n{'='*60}")
        print(f"{forecast_data['cryptocurrency']} FORECAST SUMMARY")
        print(f"{'='*60}")
        print(f"Forecast Days: {forecast_data['forecast_days']}")
        print(f"Timestamp: {forecast_data['timestamp']}")
        
        print(f"\n📈 PRICE FORECAST (Next {forecast_data['forecast_days']} days):")
        for i, (price, ci) in enumerate(zip(forecast_data['price_forecast'], forecast_data['price_confidence_intervals']), 1):
            print(f"  Day {i}: ${price:.2f} [${ci['lower']:.2f}, ${ci['upper']:.2f}]")
        
        print(f"\n📊 VOLATILITY FORECAST:")
        garch_vol = forecast_data['volatility_forecast']['GARCH']
        egarch_vol = forecast_data['volatility_forecast']['EGARCH']
        for i, (garch, egarch) in enumerate(zip(garch_vol, egarch_vol), 1):
            print(f"  Day {i}: GARCH={garch:.3f}%, EGARCH={egarch:.3f}%")
        
        print(f"\n🏆 BEST MODELS:")
        print(f"  Price: {forecast_data['best_models']['price']}")
        print(f"  Volatility: {forecast_data['best_models']['volatility']}")
        
        print(f"\n📋 MODEL METRICS:")
        metrics = forecast_data['model_metrics']
        for model, metric_data in metrics.items():
            print(f"  {model}:")
            for metric, value in metric_data.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")
    
    def print_metrics_summary(self, metrics_data):
        """Print a summary of model metrics"""
        if not metrics_data:
            print("No metrics data available")
            return
        
        print(f"\n{'='*60}")
        print(f"{metrics_data['cryptocurrency']} MODEL METRICS")
        print(f"{'='*60}")
        
        print(f"\n📊 ARIMA METRICS:")
        arima = metrics_data['arima_metrics']
        print(f"  RMSE: {arima['rmse']:.4f}")
        print(f"  MAE: {arima['mae']:.4f}")
        print(f"  MAPE: {arima['mape']:.2f}%")
        print(f"  BIC: {arima['bic']:.2f}")
        print(f"  Order: {arima['order']}")
        
        print(f"\n📈 GARCH METRICS:")
        garch = metrics_data['garch_metrics']
        for metric, value in garch.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: N/A")
        
        print(f"\n📉 EGARCH METRICS:")
        egarch = metrics_data['egarch_metrics']
        for metric, value in egarch.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: N/A")
        
        print(f"\n🏆 MODEL COMPARISON:")
        comparison = metrics_data['model_comparison']
        print(f"  Best Price Model: {comparison['best_price_model']}")
        print(f"  Best Volatility Model: {comparison['best_volatility_model']}")
        
        vol_comp = comparison['volatility_model_comparison']
        print(f"  GARCH AIC: {vol_comp['garch_aic']:.2f}")
        print(f"  EGARCH AIC: {vol_comp['egarch_aic']:.2f}")
        print(f"  AIC Difference: {vol_comp['aic_difference']:.2f}")

def main():
    """Main function to demonstrate API usage"""
    client = CryptoForecastingClient()
    
    print("🔍 Checking API health...")
    health = client.health_check()
    if health:
        print(f"✅ API is healthy: {health}")
    else:
        print("❌ API is not responding")
        return
    
    print("\n🚀 Getting BTC forecast...")
    btc_forecast = client.get_forecast("BTC", forecast_days=7)
    if btc_forecast:
        client.print_forecast_summary(btc_forecast)
        client.plot_forecast_results(btc_forecast)
    
    print("\n📊 Getting BTC metrics...")
    btc_metrics = client.get_metrics("BTC")
    if btc_metrics:
        client.print_metrics_summary(btc_metrics)
    
    print("\n📈 Getting ETH forecast...")
    eth_forecast = client.get_forecast("ETH", forecast_days=7)
    if eth_forecast:
        client.print_forecast_summary(eth_forecast)
    
    print("\n🔄 Comparing models...")
    comparison = client.compare_models()
    if comparison:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Timestamp: {comparison['timestamp']}")
        
        summary = comparison['comparison_summary']
        print(f"\nBest Models:")
        print(f"  BTC Price: {summary['btc_best_price_model']}")
        print(f"  BTC Volatility: {summary['btc_best_volatility_model']}")
        print(f"  ETH Price: {summary['eth_best_price_model']}")
        print(f"  ETH Volatility: {summary['eth_best_volatility_model']}")
        
        perf = comparison['performance_comparison']
        print(f"\nPerformance Comparison:")
        print(f"  BTC ARIMA MAPE: {perf['btc_arima_mape']:.2f}%")
        print(f"  ETH ARIMA MAPE: {perf['eth_arima_mape']:.2f}%")
        print(f"  BTC GARCH AIC: {perf['btc_garch_aic']:.2f}")
        print(f"  ETH GARCH AIC: {perf['eth_garch_aic']:.2f}")
    
    print("\n📋 Getting cache status...")
    cache_status = client.get_cache_status()
    if cache_status:
        print(f"Total cached models: {cache_status['total_cached']}")
        print(f"Cached models: {cache_status['cached_models']}")

if __name__ == "__main__":
    main()
