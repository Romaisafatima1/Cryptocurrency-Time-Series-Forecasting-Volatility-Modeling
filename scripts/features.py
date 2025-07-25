import pandas as pd
import numpy as np
import os
from pathlib import Path

def add_derived_features(df):
    """
    Add derived features to cryptocurrency data
    
    Parameters:
    df (pd.DataFrame): Raw crypto data with OHLCV columns
    
    Returns:
    pd.DataFrame: DataFrame with added derived features
    """
    
    # Make a copy to avoid modifying original data
    df_features = df.copy()
    
    # Ensure the data is sorted by date
    df_features = df_features.sort_values('timestamp')
    
    # If 'close' not in columns but 'price' is, map 'price' to 'close'
    if 'close' not in df_features.columns and 'price' in df_features.columns:
        df_features['close'] = df_features['price']
    # Fill missing OHLCV columns with 'close' or NaN
    for col in ['open', 'high', 'low', 'volume']:
        if col not in df_features.columns:
            df_features[col] = df_features['close'] if col != 'volume' else np.nan
    
    # 1. RETURNS CALCULATION
    # Simple returns (percentage change)
    df_features['daily_return'] = df_features['close'].pct_change()
    
    # Log returns (more suitable for financial analysis)
    df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))
    
    # Multi-period returns
    df_features['weekly_return'] = df_features['close'].pct_change(periods=7)
    df_features['monthly_return'] = df_features['close'].pct_change(periods=30)
    
    # 2. ROLLING MEANS (MOVING AVERAGES)
    # Short-term moving averages
    df_features['ma_7'] = df_features['close'].rolling(window=7).mean()
    df_features['ma_14'] = df_features['close'].rolling(window=14).mean()
    df_features['ma_30'] = df_features['close'].rolling(window=30).mean()
    
    # Long-term moving averages
    df_features['ma_50'] = df_features['close'].rolling(window=50).mean()
    df_features['ma_200'] = df_features['close'].rolling(window=200).mean()
    # Add missing ma_20 for Bollinger Bands
    df_features['ma_20'] = df_features['close'].rolling(window=20).mean()
    
    # Exponential moving averages (gives more weight to recent prices)
    df_features['ema_12'] = df_features['close'].ewm(span=12).mean()
    df_features['ema_26'] = df_features['close'].ewm(span=26).mean()
    
    # 3. VOLATILITY MEASURES
    # Rolling standard deviation of returns (volatility)
    df_features['volatility_7'] = df_features['daily_return'].rolling(window=7).std()
    df_features['volatility_14'] = df_features['daily_return'].rolling(window=14).std()
    df_features['volatility_30'] = df_features['daily_return'].rolling(window=30).std()
    
    # Annualized volatility (multiply by sqrt(365) for daily data)
    df_features['annualized_volatility'] = df_features['volatility_30'] * np.sqrt(365)
    
    # Price volatility (rolling std of prices)
    df_features['price_volatility_7'] = df_features['close'].rolling(window=7).std()
    df_features['price_volatility_30'] = df_features['close'].rolling(window=30).std()
    
    # 4. ADDITIONAL TECHNICAL INDICATORS
    # Bollinger Bands components
    df_features['bb_upper'] = df_features['ma_20'] + (df_features['close'].rolling(window=20).std() * 2)
    df_features['bb_lower'] = df_features['ma_20'] - (df_features['close'].rolling(window=20).std() * 2)
    df_features['bb_width'] = df_features['bb_upper'] - df_features['bb_lower']
    
    # Price relative to moving averages
    df_features['price_to_ma_7'] = df_features['close'] / df_features['ma_7']
    df_features['price_to_ma_30'] = df_features['close'] / df_features['ma_30']
    
    # MACD (Moving Average Convergence Divergence)
    df_features['macd'] = df_features['ema_12'] - df_features['ema_26']
    df_features['macd_signal'] = df_features['macd'].ewm(span=9).mean()
    df_features['macd_histogram'] = df_features['macd'] - df_features['macd_signal']
    
    # RSI (Relative Strength Index)
    delta = df_features['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_features['rsi'] = 100 - (100 / (1 + rs))
    
    # Volume-based features
    df_features['volume_ma_7'] = df_features['volume'].rolling(window=7).mean()
    df_features['volume_ratio'] = df_features['volume'] / df_features['volume_ma_7']
    
    # Price range features
    df_features['daily_range'] = df_features['high'] - df_features['low']
    df_features['range_to_close'] = df_features['daily_range'] / df_features['close']
    
    # Lag features (previous day values)
    df_features['close_lag_1'] = df_features['close'].shift(1)
    df_features['close_lag_7'] = df_features['close'].shift(7)
    df_features['volume_lag_1'] = df_features['volume'].shift(1)
    
    return df_features

def process_crypto_data(file_path, output_path):
    """
    Process a single cryptocurrency data file and add derived features
    """
    print(f"Processing {file_path}...")
    
    # Read the raw data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if it's not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Add derived features
    df_with_features = add_derived_features(df)
    
    # Save the processed data
    df_with_features.to_csv(output_path)
    print(f"Saved processed data to {output_path}")
    
    # Print summary statistics
    print(f"Original shape: {df.shape}")
    print(f"With features shape: {df_with_features.shape}")
    print(f"Added {df_with_features.shape[1] - df.shape[1]} new features")
    
    return df_with_features

def main():
    """
    Main function to process all cryptocurrency data files
    """
    # Define paths
    raw_data_path = Path("data/raw")
    processed_data_path = Path("data/processed")
    
    # Create processed data directory if it doesn't exist
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    # Process each cryptocurrency file
    crypto_files = ['btc_data.csv', 'eth_data.csv']
    
    for file_name in crypto_files:
        input_file = raw_data_path / file_name
        output_file = processed_data_path / file_name.replace('.csv', '_with_features.csv')
        
        if input_file.exists():
            df_processed = process_crypto_data(input_file, output_file)
            
            # Display some key statistics
            print(f"\n=== {file_name} Feature Summary ===")
            print("Sample of new features:")
            feature_cols = ['daily_return', 'ma_7', 'ma_30', 'volatility_7', 'volatility_30']
            print(df_processed[feature_cols].describe())
            print("\n" + "="*50 + "\n")
        else:
            print(f"Warning: {input_file} not found!")

if __name__ == "__main__":
    main()
