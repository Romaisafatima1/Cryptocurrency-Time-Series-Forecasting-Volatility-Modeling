import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf
import plotly.graph_objects as go

def compute_residuals(y_true, y_pred) -> pd.Series:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    return y_true - y_pred

def coverage_rate(y_true, lower, upper) -> float:
    y_true = pd.Series(y_true).reset_index(drop=True)
    lower = pd.Series(lower).reset_index(drop=True)
    upper = pd.Series(upper).reset_index(drop=True)
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())

def empirical_error_quantiles(residuals: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    q_low = float(residuals.quantile(alpha / 2.0))
    q_high = float(residuals.quantile(1.0 - alpha / 2.0))
    return q_low, q_high

def apply_empirical_pi(y_pred, q_low: float, q_high: float):
    y_pred = pd.Series(y_pred)
    return (y_pred + q_low, y_pred + q_high)

def normal_pi_from_residuals(y_pred, residuals: pd.Series, alpha: float = 0.05):
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    std = float(residuals.std(ddof=1))
    y_pred = pd.Series(y_pred)
    return (y_pred - z * std, y_pred + z * std)

def arima_conf_int(fitted_results, steps: int, alpha: float = 0.05):
    fc = fitted_results.get_forecast(steps=steps)
    pred_mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    return pred_mean, lower, upper

def mc_dropout_intervals(keras_model, X, passes: int = 100, alpha: float = 0.05):
    preds = []
    for _ in range(passes):
        preds.append(keras_model(X, training=True).numpy().squeeze())
    P = np.vstack(preds)
    mean = P.mean(axis=0)
    lower = np.quantile(P, alpha / 2.0, axis=0)
    upper = np.quantile(P, 1.0 - alpha / 2.0, axis=0)
    return pd.Series(mean), pd.Series(lower), pd.Series(upper)

def forecast_with_band_figure(df, time_col, y_col, yhat_col, lower_col, upper_col, title="Forecast with Prediction Band"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[y_col], name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=df[time_col], y=df[upper_col], name="Upper", mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df[time_col], y=df[lower_col], name="Prediction band", mode="lines", fill='tonexty', line=dict(width=0)))
    fig.add_trace(go.Scatter(x=df[time_col], y=df[yhat_col], name="Prediction", mode="lines"))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title=y_col)
    return fig

def residual_timeseries_figure(residuals: pd.Series, time_index=None, title="Residuals (y - ŷ)"):
    import numpy as _np
    x = time_index if time_index is not None else _np.arange(len(residuals))
    fig = go.Figure(go.Scatter(x=x, y=residuals, mode="lines", name="Residuals"))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Residual")
    return fig

def residual_hist_figure(residuals: pd.Series, nbins: int = 40, title="Residual Distribution"):
    fig = go.Figure(go.Histogram(x=residuals, nbinsx=nbins, name="Residuals"))
    fig.update_layout(title=title, xaxis_title="Residual", yaxis_title="Count")
    return fig

def residual_acf_figure(residuals: pd.Series, nlags: int = 40, title="Residual ACF"):
    acf_vals = acf(residuals, nlags=nlags, fft=True)
    lags = np.arange(len(acf_vals))
    acf_vals = acf_vals[1:]
    lags = lags[1:]
    fig = go.Figure(go.Bar(x=lags, y=acf_vals, name="ACF"))
    fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="ACF")
    return fig
