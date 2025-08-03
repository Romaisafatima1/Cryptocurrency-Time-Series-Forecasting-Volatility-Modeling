import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ModelComparisonDashboard:
    """
    Automated Model Comparison Dashboard for Cryptocurrency Forecasting
    """
    
    def __init__(self):
        self.models_data = {}
        self.performance_metrics = {}
        self.forecast_results = {}
        
    def add_model_results(self, model_name, crypto_name, metrics, forecasts=None):
        """
        Add model results to the dashboard
        """
        if crypto_name not in self.models_data:
            self.models_data[crypto_name] = {}
            self.performance_metrics[crypto_name] = {}
            self.forecast_results[crypto_name] = {}
            
        self.models_data[crypto_name][model_name] = metrics
        if forecasts is not None:
            self.forecast_results[crypto_name][model_name] = forecasts
            
    def calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    def generate_performance_comparison(self):
        """
        Generate comprehensive performance comparison across all models
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison Dashboard', fontsize=16, fontweight='bold')
        
        # Collect all metrics for comparison
        all_metrics = []
        crypto_names = list(self.models_data.keys())
        
        for crypto in crypto_names:
            for model, metrics in self.models_data[crypto].items():
                all_metrics.append({
                    'Crypto': crypto,
                    'Model': model,
                    'RMSE': metrics.get('rmse', np.nan),
                    'MAE': metrics.get('mae', np.nan),
                    'MAPE': metrics.get('mape', np.nan),
                    'AIC': metrics.get('aic', np.nan),
                    'BIC': metrics.get('bic', np.nan)
                })
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # 1. RMSE Comparison
        if not metrics_df.empty and 'RMSE' in metrics_df.columns:
            rmse_pivot = metrics_df.pivot(index='Model', columns='Crypto', values='RMSE')
            rmse_pivot.plot(kind='bar', ax=axes[0,0], color=['#FF6B6B', '#4ECDC4'])
            axes[0,0].set_title('RMSE Comparison Across Models')
            axes[0,0].set_ylabel('RMSE (lower is better)')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].legend(title='Cryptocurrency')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. MAPE Comparison
        if not metrics_df.empty and 'MAPE' in metrics_df.columns:
            mape_pivot = metrics_df.pivot(index='Model', columns='Crypto', values='MAPE')
            mape_pivot.plot(kind='bar', ax=axes[0,1], color=['#FF6B6B', '#4ECDC4'])
            axes[0,1].set_title('MAPE Comparison Across Models')
            axes[0,1].set_ylabel('MAPE % (lower is better)')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].legend(title='Cryptocurrency')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Model Ranking Heatmap
        if not metrics_df.empty:
            # Create a ranking system (1 = best, higher = worse)
            ranking_df = metrics_df.copy()
            for metric in ['RMSE', 'MAE', 'MAPE']:
                if metric in ranking_df.columns:
                    ranking_df[f'{metric}_Rank'] = ranking_df.groupby('Crypto')[metric].rank()
            
            # Create heatmap of rankings
            rank_columns = [col for col in ranking_df.columns if 'Rank' in col]
            if rank_columns:
                rank_pivot = ranking_df.pivot(index='Model', columns='Crypto', values=rank_columns[0])
                sns.heatmap(rank_pivot, annot=True, cmap='RdYlGn_r', ax=axes[1,0], cbar_kws={'label': 'Rank (1=Best)'})
                axes[1,0].set_title('Model Performance Ranking')
        
        # 4. AIC/BIC Comparison (for volatility models)
        volatility_models = [model for model in metrics_df['Model'].unique() if 'GARCH' in model or 'EGARCH' in model]
        if volatility_models:
            vol_metrics = metrics_df[metrics_df['Model'].isin(volatility_models)]
            if 'AIC' in vol_metrics.columns:
                aic_pivot = vol_metrics.pivot(index='Model', columns='Crypto', values='AIC')
                aic_pivot.plot(kind='bar', ax=axes[1,1], color=['#FF6B6B', '#4ECDC4'])
                axes[1,1].set_title('Volatility Model AIC Comparison')
                axes[1,1].set_ylabel('AIC (lower is better)')
                axes[1,1].tick_params(axis='x', rotation=45)
                axes[1,1].legend(title='Cryptocurrency')
                axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return metrics_df
    
    def generate_forecast_comparison(self):
        """
        Generate forecast comparison visualization
        """
        if not self.forecast_results:
            print("No forecast data available for comparison")
            return
            
        for crypto in self.forecast_results.keys():
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle(f'{crypto} Forecast Comparison', fontsize=14, fontweight='bold')
            
            # Price forecasts
            price_models = [model for model in self.forecast_results[crypto].keys() 
                          if 'ARIMA' in model or 'LSTM' in model]
            
            if price_models:
                for i, model in enumerate(price_models):
                    forecast = self.forecast_results[crypto][model]
                    days = range(1, len(forecast) + 1)
                    axes[0].plot(days, forecast, marker='o', label=model, linewidth=2)
                
                axes[0].set_title('Price Forecast Comparison')
                axes[0].set_xlabel('Days Ahead')
                axes[0].set_ylabel('Price ($)')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
            
            # Volatility forecasts
            vol_models = [model for model in self.forecast_results[crypto].keys() 
                         if 'GARCH' in model or 'EGARCH' in model]
            
            if vol_models:
                for i, model in enumerate(vol_models):
                    forecast = self.forecast_results[crypto][model]
                    days = range(1, len(forecast) + 1)
                    axes[1].plot(days, forecast, marker='s', label=model, linewidth=2)
                
                axes[1].set_title('Volatility Forecast Comparison')
                axes[1].set_xlabel('Days Ahead')
                axes[1].set_ylabel('Volatility (%)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("="*60)
        print("AUTOMATED MODEL COMPARISON DASHBOARD")
        print("="*60)
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for crypto in self.models_data.keys():
            print(f"📊 {crypto} ANALYSIS")
            print("-" * 40)
            
            # Find best model for each metric
            metrics_df = pd.DataFrame([
                {'Model': model, **metrics}
                for model, metrics in self.models_data[crypto].items()
            ])
            
            if not metrics_df.empty:
                # Best RMSE
                if 'rmse' in metrics_df.columns:
                    best_rmse = metrics_df.loc[metrics_df['rmse'].idxmin()]
                    print(f"🏆 Best RMSE: {best_rmse['Model']} ({best_rmse['rmse']:.4f})")
                
                # Best MAPE
                if 'mape' in metrics_df.columns:
                    best_mape = metrics_df.loc[metrics_df['mape'].idxmin()]
                    print(f"🎯 Best MAPE: {best_mape['Model']} ({best_mape['mape']:.2f}%)")
                
                # Best AIC (for volatility models)
                volatility_models = metrics_df[metrics_df['Model'].str.contains('GARCH', case=False)]
                if not volatility_models.empty and 'aic' in volatility_models.columns:
                    best_aic = volatility_models.loc[volatility_models['aic'].idxmin()]
                    print(f"📈 Best Volatility Model: {best_aic['Model']} (AIC: {best_aic['aic']:.2f})")
            
            print()
        
        print("="*60)
        print("DASHBOARD SUMMARY")
        print("="*60)
        print("✅ Performance metrics calculated (RMSE, MAPE, MAE)")
        print("✅ Model comparison visualizations generated")
        print("✅ Forecast comparisons available")
        print("✅ Automated ranking system implemented")
        print("="*60)

# Example usage and integration with existing code
def integrate_with_existing_models():
    """
    Example of how to integrate the dashboard with your existing models
    """
    dashboard = ModelComparisonDashboard()
    
    # Example: Add ARIMA results (from your main.py)
    # dashboard.add_model_results(
    #     model_name="ARIMA",
    #     crypto_name="BTC",
    #     metrics={
    #         'rmse': btc_price_results['metrics']['rmse'],
    #         'mae': btc_price_results['metrics']['mae'],
    #         'mape': btc_price_results['metrics']['mape']
    #     },
    #     forecasts=btc_price_results['forecast']
    # )
    
    # Example: Add GARCH results
    # dashboard.add_model_results(
    #     model_name="GARCH",
    #     crypto_name="BTC",
    #     metrics={
    #         'rmse': btc_vol_results['GARCH']['rmse'],
    #         'mae': btc_vol_results['GARCH']['mae'],
    #         'aic': btc_vol_results['GARCH']['aic'],
    #         'bic': btc_vol_results['GARCH']['bic']
    #     },
    #     forecasts=btc_vol_results['GARCH']['forecast']
    # )
    
    return dashboard

if __name__ == "__main__":
    # Create and demonstrate the dashboard
    dashboard = ModelComparisonDashboard()
    
    # Add sample data for demonstration
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
    
    # Generate dashboard
    metrics_df = dashboard.generate_performance_comparison()
    dashboard.generate_forecast_comparison()
    dashboard.generate_summary_report() 