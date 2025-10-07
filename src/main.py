import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load local BTC and ETH data
btc_df = pd.read_csv('data/raw/btc_data.csv', parse_dates=['timestamp'], index_col='timestamp')
eth_df = pd.read_csv('data/raw/eth_data.csv', parse_dates=['timestamp'], index_col='timestamp')

# Helper: Forecast price with Auto-ARIMA (enhanced with BIC)
def forecast_auto_arima(series, steps=7, test_size=0.2):
    """
    Forecast prices using Auto-ARIMA with proper train/test split
    """
    print(f"Training Auto-ARIMA model...")
    
    # Split data into train/test
    split_idx = int(len(series) * (1 - test_size))
    train_data = series[:split_idx]
    test_data = series[split_idx:]
    
    # Find best ARIMA parameters automatically
    auto_model = auto_arima(
        train_data,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_p=5, max_d=2, max_q=5,
        information_criterion='aic'
    )
    
    print(f"Best ARIMA order found: {auto_model.order}")
    
    # Fit final model on training data
    model = ARIMA(train_data, order=auto_model.order)
    model_fit = model.fit()
    
    # Evaluate on test data
    test_forecast = model_fit.forecast(steps=len(test_data))
    
    # Calculate evaluation metrics
    mse = mean_squared_error(test_data, test_forecast)
    mae = mean_absolute_error(test_data, test_forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_data - test_forecast) / test_data)) * 100
    bic = model_fit.bic
    
    print(f"Model Evaluation on Test Data:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  BIC: {bic:.4f}")
    
    # Generate future forecast
    future_forecast = model_fit.forecast(steps=steps)
    future_ci = model_fit.get_forecast(steps=steps).conf_int()
    
    return {
        'model': model_fit,
        'order': auto_model.order,
        'forecast': future_forecast,
        'confidence_interval': future_ci,
        'metrics': {'rmse': rmse, 'mae': mae, 'mape': mape, 'bic': bic},
        'test_data': test_data,
        'test_forecast': test_forecast
    }

# Helper: Forecast volatility with GARCH and EGARCH (enhanced version)
def forecast_garch_models(series, steps=7, test_size=0.2):
    """
    Forecast volatility using both GARCH and EGARCH models
    """
    print(f"Training GARCH and EGARCH models...")
    
    # Calculate log returns
    log_returns = np.log(series).diff().dropna() * 100  # Convert to percentage
    
    # Remove extreme outliers
    log_returns_clean = log_returns[np.abs(log_returns - log_returns.mean()) <= (3 * log_returns.std())]
    
    # Split data
    split_idx = int(len(log_returns_clean) * (1 - test_size))
    train_returns = log_returns_clean[:split_idx]
    test_returns = log_returns_clean[split_idx:]
    
    results = {}
    
    # Model 1: Standard GARCH(1,1)
    print("  Training GARCH(1,1)...")
    garch_model = arch_model(train_returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    
    # GARCH forecast
    garch_forecast = garch_fit.forecast(horizon=steps)
    garch_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
    
    # GARCH evaluation on test data
    garch_test_forecast = garch_fit.forecast(horizon=len(test_returns), reindex=False)
    garch_test_vol = np.sqrt(garch_test_forecast.variance.values[-1, :])
    realized_vol = test_returns.rolling(window=5).std().dropna()[:len(garch_test_vol)]
    
    if len(realized_vol) > 0:
        garch_rmse = np.sqrt(mean_squared_error(realized_vol, garch_test_vol[:len(realized_vol)]))
        garch_mae = mean_absolute_error(realized_vol, garch_test_vol[:len(realized_vol)])
    else:
        garch_rmse = garch_mae = np.nan
    
    results['GARCH'] = {
        'model': garch_fit,
        'forecast': garch_volatility,
        'rmse': garch_rmse,
        'mae': garch_mae,
        'aic': garch_fit.aic,
        'bic': garch_fit.bic
    }
    
    # Model 2: EGARCH(1,1) - captures asymmetric volatility effects
    print("  Training EGARCH(1,1)...")
    egarch_model = arch_model(train_returns, vol='EGARCH', p=1, q=1)
    egarch_fit = egarch_model.fit(disp='off')
    
    # EGARCH forecast
    egarch_forecast = egarch_fit.forecast(horizon=steps)
    egarch_volatility = np.sqrt(egarch_forecast.variance.values[-1, :])
    
    # EGARCH evaluation on test data
    egarch_test_forecast = egarch_fit.forecast(horizon=len(test_returns), reindex=False)
    egarch_test_vol = np.sqrt(egarch_test_forecast.variance.values[-1, :])
    
    if len(realized_vol) > 0:
        egarch_rmse = np.sqrt(mean_squared_error(realized_vol, egarch_test_vol[:len(realized_vol)]))
        egarch_mae = mean_absolute_error(realized_vol, egarch_test_vol[:len(realized_vol)])
    else:
        egarch_rmse = egarch_mae = np.nan
    
    results['EGARCH'] = {
        'model': egarch_fit,
        'forecast': egarch_volatility,
        'rmse': egarch_rmse,
        'mae': egarch_mae,
        'aic': egarch_fit.aic,
        'bic': egarch_fit.bic
    }
    
    # Print comparison
    print(f"Model Comparison:")
    print(f"  GARCH  - AIC: {garch_fit.aic:.2f}, BIC: {garch_fit.bic:.2f}, RMSE: {garch_rmse:.4f}")
    print(f"  EGARCH - AIC: {egarch_fit.aic:.2f}, BIC: {egarch_fit.bic:.2f}, RMSE: {egarch_rmse:.4f}")
    
    # Determine best model
    best_model = 'GARCH' if garch_fit.aic < egarch_fit.aic else 'EGARCH'
    print(f"  Best model (by AIC): {best_model}")
    
    return results, log_returns_clean

# NEW FUNCTION: Plot Actual vs Predicted for ARIMA Models
def plot_actual_vs_predicted(test_data, predicted_data, coin_name, model_name):
    """
    Plot actual vs predicted values for model evaluation
    """
    plt.figure(figsize=(12, 6))
    
    # Main plot
    plt.plot(test_data.index, test_data.values, label='Actual', marker='o', linewidth=2, color='blue')
    plt.plot(test_data.index, predicted_data, label=f'{model_name} Predicted', marker='x', linewidth=2, color='red')
    
    # Calculate residuals and add them as shaded area
    residuals = test_data.values - predicted_data
    plt.fill_between(test_data.index, test_data.values, predicted_data, alpha=0.3, color='gray', label='Residuals')
    
    plt.title(f'{coin_name} - {model_name}: Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add performance metrics as text box
    mse = mean_squared_error(test_data.values, predicted_data)
    mae = mean_absolute_error(test_data.values, predicted_data)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_data.values - predicted_data) / test_data.values)) * 100
    
    textstr = f'RMSE: ${rmse:.2f}\nMAE: ${mae:.2f}\nMAPE: {mape:.2f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.show()

# NEW FUNCTION: Build Comprehensive Comparison Table
def build_comparison_table(btc_price_results, eth_price_results, btc_vol_results, eth_vol_results):
    """
    Build a comprehensive table comparing all models for both coins
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON TABLE")
    print("="*80)
    
    # Prepare data for the comparison table
    comparison_data = []
    
    # BTC ARIMA
    comparison_data.append({
        'Coin': 'BTC',
        'Model': 'ARIMA',
        'RMSE': f"{btc_price_results['metrics']['rmse']:.4f}",
        'MAE': f"{btc_price_results['metrics']['mae']:.4f}",
        'MAPE (%)': f"{btc_price_results['metrics']['mape']:.2f}",
        'BIC': f"{btc_price_results['metrics']['bic']:.2f}",
        'Parameters': str(btc_price_results['order'])
    })
    
    # ETH ARIMA
    comparison_data.append({
        'Coin': 'ETH',
        'Model': 'ARIMA',
        'RMSE': f"{eth_price_results['metrics']['rmse']:.4f}",
        'MAE': f"{eth_price_results['metrics']['mae']:.4f}",
        'MAPE (%)': f"{eth_price_results['metrics']['mape']:.2f}",
        'BIC': f"{eth_price_results['metrics']['bic']:.2f}",
        'Parameters': str(eth_price_results['order'])
    })
    
    # BTC GARCH
    comparison_data.append({
        'Coin': 'BTC',
        'Model': 'GARCH',
        'RMSE': f"{btc_vol_results['GARCH']['rmse']:.4f}" if not np.isnan(btc_vol_results['GARCH']['rmse']) else 'N/A',
        'MAE': f"{btc_vol_results['GARCH']['mae']:.4f}" if not np.isnan(btc_vol_results['GARCH']['mae']) else 'N/A',
        'MAPE (%)': 'N/A',
        'BIC': f"{btc_vol_results['GARCH']['bic']:.2f}",
        'Parameters': 'p=1, q=1'
    })
    
    # BTC EGARCH
    comparison_data.append({
        'Coin': 'BTC',
        'Model': 'EGARCH',
        'RMSE': f"{btc_vol_results['EGARCH']['rmse']:.4f}" if not np.isnan(btc_vol_results['EGARCH']['rmse']) else 'N/A',
        'MAE': f"{btc_vol_results['EGARCH']['mae']:.4f}" if not np.isnan(btc_vol_results['EGARCH']['mae']) else 'N/A',
        'MAPE (%)': 'N/A',
        'BIC': f"{btc_vol_results['EGARCH']['bic']:.2f}",
        'Parameters': 'p=1, q=1'
    })
    
    # ETH GARCH
    comparison_data.append({
        'Coin': 'ETH',
        'Model': 'GARCH',
        'RMSE': f"{eth_vol_results['GARCH']['rmse']:.4f}" if not np.isnan(eth_vol_results['GARCH']['rmse']) else 'N/A',
        'MAE': f"{eth_vol_results['GARCH']['mae']:.4f}" if not np.isnan(eth_vol_results['GARCH']['mae']) else 'N/A',
        'MAPE (%)': 'N/A',
        'BIC': f"{eth_vol_results['GARCH']['bic']:.2f}",
        'Parameters': 'p=1, q=1'
    })
    
    # ETH EGARCH
    comparison_data.append({
        'Coin': 'ETH',
        'Model': 'EGARCH',
        'RMSE': f"{eth_vol_results['EGARCH']['rmse']:.4f}" if not np.isnan(eth_vol_results['EGARCH']['rmse']) else 'N/A',
        'MAE': f"{eth_vol_results['EGARCH']['mae']:.4f}" if not np.isnan(eth_vol_results['EGARCH']['mae']) else 'N/A',
        'MAPE (%)': 'N/A',
        'BIC': f"{eth_vol_results['EGARCH']['bic']:.2f}",
        'Parameters': 'p=1, q=1'
    })
    
    # Create and display the comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    print(comparison_df.to_string(index=False))
    
    # Save to CSV for future reference
    comparison_df.to_csv('data/processed/model_comparison_table.csv', index=False)
    print(f"\nComparison table saved to: data/processed/model_comparison_table.csv")
    
    # Best Model Analysis
    print("\n" + "="*60)
    print("BEST MODEL ANALYSIS")
    print("="*60)
    
    # Best ARIMA model (lowest BIC)
    arima_models = comparison_df[comparison_df['Model'] == 'ARIMA'].copy()
    arima_models['BIC_float'] = pd.to_numeric(arima_models['BIC'])
    best_arima = arima_models.loc[arima_models['BIC_float'].idxmin()]
    print(f"Best ARIMA Model: {best_arima['Coin']} (BIC: {best_arima['BIC']})")
    
    # Best volatility models
    for coin in ['BTC', 'ETH']:
        coin_vol = comparison_df[(comparison_df['Coin'] == coin) & (comparison_df['Model'].isin(['GARCH', 'EGARCH']))].copy()
        coin_vol['BIC_float'] = pd.to_numeric(coin_vol['BIC'])
        best_vol = coin_vol.loc[coin_vol['BIC_float'].idxmin()]
        print(f"Best Volatility Model for {coin}: {best_vol['Model']} (BIC: {best_vol['BIC']})")
    
    return comparison_df

def plot_forecasts(price_results, vol_results, crypto_name):
    """
    Plot price and volatility forecasts
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{crypto_name} Forecasting Results', fontsize=16)
    
    # Price forecast plot
    forecast = price_results['forecast']
    ci = price_results['confidence_interval']
    test_data = price_results['test_data']
    test_forecast = price_results['test_forecast']
    
    # Plot test data vs forecast
    axes[0, 0].plot(test_data.index, test_data.values, label='Actual', color='blue', linewidth=2)
    axes[0, 0].plot(test_data.index, test_forecast, label='ARIMA Forecast', color='red', linewidth=2)
    axes[0, 0].set_title('Price Forecast vs Actual (Test Data)')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Future price forecast
    future_dates = pd.date_range(start=test_data.index[-1], periods=len(forecast)+1, freq='D')[1:]
    axes[0, 1].plot(future_dates, forecast, label='Future Forecast', color='green', linewidth=2)
    axes[0, 1].fill_between(future_dates, ci.iloc[:, 0], ci.iloc[:, 1], 
                           alpha=0.3, color='green', label='95% Confidence Interval')
    axes[0, 1].set_title('Future Price Forecast (Next 7 Days)')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Volatility comparison
    garch_vol = vol_results['GARCH']['forecast']
    egarch_vol = vol_results['EGARCH']['forecast']
    
    days = range(1, len(garch_vol) + 1)
    axes[1, 0].plot(days, garch_vol, label='GARCH', marker='o', linewidth=2)
    axes[1, 0].plot(days, egarch_vol, label='EGARCH', marker='s', linewidth=2)
    axes[1, 0].set_title('Volatility Forecast Comparison')
    axes[1, 0].set_xlabel('Days Ahead')
    axes[1, 0].set_ylabel('Volatility (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Model performance comparison
    models = ['GARCH', 'EGARCH']
    aics = [vol_results['GARCH']['aic'], vol_results['EGARCH']['aic']]
    
    axes[1, 1].bar(models, aics, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_title('Volatility Model Comparison (AIC)')
    axes[1, 1].set_ylabel('AIC (lower is better)')
    
    # Add AIC values on bars
    for i, v in enumerate(aics):
        axes[1, 1].text(i, v + max(aics)*0.01, f'{v:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Main execution
print("="*60)
print("CRYPTOCURRENCY FORECASTING WITH AUTO-ARIMA AND GARCH/EGARCH")
print("="*60)

# Analyze BTC
print("\n" + "="*30)
print("ANALYZING BITCOIN (BTC)")
print("="*30)

btc_price_results = forecast_auto_arima(btc_df['price'])
btc_vol_results, btc_returns = forecast_garch_models(btc_df['price'])

print('\nBTC Results Summary:')
print('='*40)
print(f'Price Forecast (next 7 days):')
for i, price in enumerate(btc_price_results['forecast'], 1):
    ci_lower = btc_price_results['confidence_interval'].iloc[i-1, 0]
    ci_upper = btc_price_results['confidence_interval'].iloc[i-1, 1]
    print(f'  Day {i}: ${price:.2f} [${ci_lower:.2f}, ${ci_upper:.2f}]')

print(f'\nVolatility Forecast (next 7 days):')
print(f'  GARCH:  {btc_vol_results["GARCH"]["forecast"]}')
print(f'  EGARCH: {btc_vol_results["EGARCH"]["forecast"]}')

# PLOT ACTUAL vs PREDICTED for BTC
plot_actual_vs_predicted(btc_price_results['test_data'], btc_price_results['test_forecast'], 'Bitcoin (BTC)', 'ARIMA')

# Plot BTC results
plot_forecasts(btc_price_results, btc_vol_results, 'Bitcoin (BTC)')

# Analyze ETH
print("\n" + "="*30)
print("ANALYZING ETHEREUM (ETH)")
print("="*30)

eth_price_results = forecast_auto_arima(eth_df['price'])
eth_vol_results, eth_returns = forecast_garch_models(eth_df['price'])

print('\nETH Results Summary:')
print('='*40)
print(f'Price Forecast (next 7 days):')
for i, price in enumerate(eth_price_results['forecast'], 1):
    ci_lower = eth_price_results['confidence_interval'].iloc[i-1, 0]
    ci_upper = eth_price_results['confidence_interval'].iloc[i-1, 1]
    print(f'  Day {i}: ${price:.2f} [${ci_lower:.2f}, ${ci_upper:.2f}]')

print(f'\nVolatility Forecast (next 7 days):')
print(f'  GARCH:  {eth_vol_results["GARCH"]["forecast"]}')
print(f'  EGARCH: {eth_vol_results["EGARCH"]["forecast"]}')

# PLOT ACTUAL vs PREDICTED for ETH
plot_actual_vs_predicted(eth_price_results['test_data'], eth_price_results['test_forecast'], 'Ethereum (ETH)', 'ARIMA')

# Plot ETH results
plot_forecasts(eth_price_results, eth_vol_results, 'Ethereum (ETH)')

# BUILD COMPREHENSIVE COMPARISON TABLE
comparison_table = build_comparison_table(btc_price_results, eth_price_results, btc_vol_results, eth_vol_results)

# Overall comparison (keeping original summary)
print("\n" + "="*50)
print("OVERALL MODEL PERFORMANCE SUMMARY")
print("="*50)

comparison_data = {
    'Cryptocurrency': ['BTC', 'ETH'],
    'ARIMA Order': [str(btc_price_results['order']), str(eth_price_results['order'])],
    'Price RMSE': [btc_price_results['metrics']['rmse'], eth_price_results['metrics']['rmse']],
    'Price MAPE (%)': [btc_price_results['metrics']['mape'], eth_price_results['metrics']['mape']],
    'Best Vol Model': [
        'GARCH' if btc_vol_results['GARCH']['aic'] < btc_vol_results['EGARCH']['aic'] else 'EGARCH',
        'GARCH' if eth_vol_results['GARCH']['aic'] < eth_vol_results['EGARCH']['aic'] else 'EGARCH'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("- All models evaluated using MAE, RMSE, and BIC")
print("- Actual vs predicted plots generated")
print("- Comprehensive comparison table built")
print("- Model evaluation and comparison completed")
print("="*50)
