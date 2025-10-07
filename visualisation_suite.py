#!/usr/bin/env python3
"""
Advanced Visualization Suite for Cryptocurrency Forecasting
Week 7: Interpretability & UI Planning
Includes: Confidence intervals, prediction bands, residual analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class ForecastVisualizer:
    """Comprehensive visualization suite for forecast analysis"""
    
    def __init__(self, actual, predicted, residuals=None):
        """
        Initialize visualizer with forecast results
        
        Parameters:
        actual: array-like, actual values
        predicted: array-like, predicted values
        residuals: array-like, optional residuals (will compute if not provided)
        """
        self.actual = np.array(actual)
        self.predicted = np.array(predicted)
        self.residuals = residuals if residuals is not None else self.actual - self.predicted
        
    def plot_forecast_with_confidence(self, confidence_level=0.95, 
                                     title="Forecast with Confidence Intervals"):
        """
        Plot forecast with confidence intervals and prediction bands
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Calculate confidence intervals
        std_error = np.std(self.residuals)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        confidence_interval = z_score * std_error
        upper_bound = self.predicted + confidence_interval
        lower_bound = self.predicted - confidence_interval
        
        # Time axis
        x = np.arange(len(self.actual))
        
        # Plot actual values
        ax.plot(x, self.actual, 'o-', label='Actual', color='#2C3E50', 
                linewidth=2, markersize=4, alpha=0.8)
        
        # Plot predicted values
        ax.plot(x, self.predicted, 's-', label='Predicted', color='#E74C3C', 
                linewidth=2, markersize=4, alpha=0.8)
        
        # Plot confidence bands
        ax.fill_between(x, lower_bound, upper_bound, 
                        alpha=0.2, color='#3498DB', 
                        label=f'{int(confidence_level*100)}% Confidence Interval')
        
        # Styling
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/forecast_confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return upper_bound, lower_bound
    
    def plot_residual_diagnostics(self, title="Residual Diagnostics"):
        """
        Comprehensive residual analysis with 4 diagnostic plots
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Residuals over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.residuals, 'o-', color='#3498DB', alpha=0.7, markersize=3)
        ax1.axhline(y=0, color='#E74C3C', linestyle='--', linewidth=2)
        ax1.fill_between(range(len(self.residuals)), 
                         -2*np.std(self.residuals), 
                         2*np.std(self.residuals),
                         alpha=0.2, color='#E74C3C')
        ax1.set_xlabel('Time', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax1.set_title('Residuals Over Time (±2σ bands)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram of residuals with normal distribution overlay
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(self.residuals, bins=30, density=True, alpha=0.7, 
                color='#3498DB', edgecolor='black')
        
        # Fit normal distribution
        mu, sigma = np.mean(self.residuals), np.std(self.residuals)
        x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
        ax2.set_xlabel('Residual Value', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=11, fontweight='bold')
        ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot
        ax3 = fig.add_subplot(gs[1, 1])
        stats.probplot(self.residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. ACF Plot
        ax4 = fig.add_subplot(gs[2, 0])
        plot_acf(self.residuals, lags=30, ax=ax4, alpha=0.05)
        ax4.set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals vs Fitted
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.scatter(self.predicted, self.residuals, alpha=0.6, color='#3498DB', s=30)
        ax5.axhline(y=0, color='#E74C3C', linestyle='--', linewidth=2)
        ax5.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Residuals', fontsize=11, fontweight='bold')
        ax5.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('visualizations/residual_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def statistical_tests(self):
        """
        Perform statistical tests on residuals
        """
        print("=" * 60)
        print("RESIDUAL STATISTICAL TESTS")
        print("=" * 60)
        
        # Normality tests
        print("\n1. NORMALITY TESTS:")
        shapiro_stat, shapiro_p = stats.shapiro(self.residuals)
        print(f"   Shapiro-Wilk Test:")
        print(f"   - Statistic: {shapiro_stat:.4f}")
        print(f"   - P-value: {shapiro_p:.4f}")
        print(f"   - Result: {'Normal' if shapiro_p > 0.05 else 'Not Normal'} (α=0.05)")
        
        ks_stat, ks_p = stats.kstest(self.residuals, 'norm', 
                                     args=(np.mean(self.residuals), np.std(self.residuals)))
        print(f"\n   Kolmogorov-Smirnov Test:")
        print(f"   - Statistic: {ks_stat:.4f}")
        print(f"   - P-value: {ks_p:.4f}")
        print(f"   - Result: {'Normal' if ks_p > 0.05 else 'Not Normal'} (α=0.05)")
        
        # Autocorrelation test (Ljung-Box)
        print("\n2. AUTOCORRELATION TEST (Ljung-Box):")
        lb_test = acorr_ljungbox(self.residuals, lags=10, return_df=True)
        print(f"   First 3 lags:")
        for i in range(min(3, len(lb_test))):
            print(f"   - Lag {i+1}: P-value = {lb_test['lb_pvalue'].iloc[i]:.4f}")
        significant_lags = (lb_test['lb_pvalue'] < 0.05).sum()
        print(f"   - Significant lags (p < 0.05): {significant_lags}/{len(lb_test)}")
        print(f"   - Result: {'No significant autocorrelation' if significant_lags == 0 else 'Autocorrelation detected'}")
        
        # Descriptive statistics
        print("\n3. DESCRIPTIVE STATISTICS:")
        print(f"   - Mean: {np.mean(self.residuals):.4f}")
        print(f"   - Std Dev: {np.std(self.residuals):.4f}")
        print(f"   - Skewness: {stats.skew(self.residuals):.4f}")
        print(f"   - Kurtosis: {stats.kurtosis(self.residuals):.4f}")
        print(f"   - Min: {np.min(self.residuals):.4f}")
        print(f"   - Max: {np.max(self.residuals):.4f}")
        
        print("\n" + "=" * 60)
        
    def plot_prediction_bands(self, future_steps=30, title="Multi-Step Forecast with Prediction Bands"):
        """
        Plot expanding prediction bands for multi-step forecasting
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical data
        historical_x = np.arange(len(self.actual))
        ax.plot(historical_x, self.actual, 'o-', label='Historical', 
                color='#2C3E50', linewidth=2, markersize=4)
        
        # Future predictions with expanding bands
        future_x = np.arange(len(self.actual), len(self.actual) + future_steps)
        
        # Simple forecast (using last value trend)
        last_value = self.actual[-1]
        trend = (self.actual[-1] - self.actual[-30]) / 30
        future_pred = [last_value + trend * i for i in range(1, future_steps + 1)]
        
        # Expanding confidence bands (increase with forecast horizon)
        std_error = np.std(self.residuals)
        expanding_std = [std_error * np.sqrt(i) for i in range(1, future_steps + 1)]
        
        upper_band = [future_pred[i] + 1.96 * expanding_std[i] for i in range(future_steps)]
        lower_band = [future_pred[i] - 1.96 * expanding_std[i] for i in range(future_steps)]
        
        # Plot future forecast
        ax.plot(future_x, future_pred, 's-', label='Forecast', 
                color='#E74C3C', linewidth=2, markersize=4)
        
        # Plot expanding bands
        ax.fill_between(future_x, lower_band, upper_band, 
                        alpha=0.2, color='#3498DB', 
                        label='95% Prediction Band')
        
        # Add vertical line at forecast start
        ax.axvline(x=len(self.actual)-1, color='gray', linestyle='--', 
                   linewidth=2, alpha=0.5, label='Forecast Start')
        
        # Styling
        ax.set_xlabel('Time Period', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/prediction_bands.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_comprehensive_report(btc_actual, btc_pred, eth_actual, eth_pred):
    """
    Generate comprehensive visualization report for both BTC and ETH
    """
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE VISUALIZATION SUITE")
    print("="*70)
    
    # BTC Analysis
    print("\n📊 Analyzing Bitcoin (BTC)...")
    btc_viz = ForecastVisualizer(btc_actual, btc_pred)
    btc_viz.plot_forecast_with_confidence(title="BTC: Forecast with 95% Confidence Intervals")
    btc_viz.plot_residual_diagnostics(title="BTC: Residual Diagnostics")
    btc_viz.statistical_tests()
    btc_viz.plot_prediction_bands(future_steps=30, title="BTC: 30-Day Forecast with Prediction Bands")
    
    # ETH Analysis
    print("\n📊 Analyzing Ethereum (ETH)...")
    eth_viz = ForecastVisualizer(eth_actual, eth_pred)
    eth_viz.plot_forecast_with_confidence(title="ETH: Forecast with 95% Confidence Intervals")
    eth_viz.plot_residual_diagnostics(title="ETH: Residual Diagnostics")
    eth_viz.statistical_tests()
    eth_viz.plot_prediction_bands(future_steps=30, title="ETH: 30-Day Forecast with Prediction Bands")
    
    print("\n✅ All visualizations saved to 'visualizations/' directory")
    print("="*70)

if __name__ == "__main__":
    # Example usage with sample data
    print("📈 Cryptocurrency Forecast Visualization Suite")
    print("This module provides advanced visualization capabilities for forecast analysis")
    print("\nFeatures:")
    print("  ✓ Confidence intervals and prediction bands")
    print("  ✓ Comprehensive residual diagnostics")
    print("  ✓ Statistical tests for model validation")
    print("  ✓ Multi-step forecast visualization")
    print("\nTo use: Import and call create_comprehensive_report() with your forecast data")