#!/usr/bin/env python3
"""
Test script for the Automated Model Comparison Dashboard
"""

import pandas as pd
import numpy as np
from src.dashboard import ModelComparisonDashboard

def test_dashboard():
    """
    Test the dashboard with sample data
    """
    print("🧪 Testing Automated Model Comparison Dashboard")
    print("="*50)
    
    # Create dashboard instance
    dashboard = ModelComparisonDashboard()
    
    # Add sample BTC results
    print("📊 Adding Bitcoin (BTC) model results...")
    dashboard.add_model_results("ARIMA", "BTC", {
        'rmse': 1250.45, 'mae': 980.32, 'mape': 2.34
    })
    dashboard.add_model_results("LSTM", "BTC", {
        'rmse': 1180.67, 'mae': 920.15, 'mape': 2.18
    })
    dashboard.add_model_results("GARCH", "BTC", {
        'rmse': 0.045, 'mae': 0.038, 'aic': -1250.67, 'bic': -1240.23
    })
    dashboard.add_model_results("EGARCH", "BTC", {
        'rmse': 0.042, 'mae': 0.035, 'aic': -1260.45, 'bic': -1249.89
    })
    
    # Add sample ETH results
    print("📊 Adding Ethereum (ETH) model results...")
    dashboard.add_model_results("ARIMA", "ETH", {
        'rmse': 95.23, 'mae': 75.45, 'mape': 1.85
    })
    dashboard.add_model_results("LSTM", "ETH", {
        'rmse': 88.67, 'mae': 68.15, 'mape': 1.72
    })
    dashboard.add_model_results("GARCH", "ETH", {
        'rmse': 0.038, 'mae': 0.032, 'aic': -1180.45, 'bic': -1170.12
    })
    dashboard.add_model_results("EGARCH", "ETH", {
        'rmse': 0.035, 'mae': 0.029, 'aic': -1190.23, 'bic': -1179.89
    })
    
    # Add sample forecast data
    print("📈 Adding sample forecast data...")
    dashboard.add_model_results("ARIMA", "BTC", {
        'rmse': 1250.45, 'mae': 980.32, 'mape': 2.34
    }, forecasts=[45000, 45200, 45400, 45600, 45800, 46000, 46200])
    
    dashboard.add_model_results("LSTM", "BTC", {
        'rmse': 1180.67, 'mae': 920.15, 'mape': 2.18
    }, forecasts=[44800, 45000, 45200, 45400, 45600, 45800, 46000])
    
    # Generate dashboard
    print("\n🎨 Generating dashboard visualizations...")
    metrics_df = dashboard.generate_performance_comparison()
    dashboard.generate_forecast_comparison()
    dashboard.generate_summary_report()
    
    # Save results
    metrics_df.to_csv('test_dashboard_results.csv', index=False)
    print(f"\n💾 Test results saved to 'test_dashboard_results.csv'")
    
    print("\n✅ Dashboard test completed successfully!")
    return dashboard, metrics_df

if __name__ == "__main__":
    dashboard, metrics_df = test_dashboard() 