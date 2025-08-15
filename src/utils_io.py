import pandas as pd

def build_forecast_df(timestamps, y_true, y_pred, lower, upper,
                      time_col="ds", y_col="y", yhat_col="yhat",
                      lower_col="yhat_lower", upper_col="yhat_upper"):
    return pd.DataFrame({
        time_col: timestamps,
        y_col: y_true,
        yhat_col: y_pred,
        lower_col: lower,
        upper_col: upper
    })
