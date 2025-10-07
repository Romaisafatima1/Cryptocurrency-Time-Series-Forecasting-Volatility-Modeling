import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# Load your processed data
df = pd.read_csv('data/processed/btc_spark_output.csv/part-00000-f9588ca1-3f4c-453b-8352-0fadd4b9a3ab-c000.csv')  # adjust if file name is different
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Normalize the 'price' column
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['price']])

# Create sequences
sequence_length = 60
X, y = [], []
for i in range(len(scaled_data) - sequence_length):
    X.append(scaled_data[i:i + sequence_length])
    y.append(scaled_data[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Save to .npy files
os.makedirs('data/lstm/', exist_ok=True)
np.save('data/lstm/X.npy', X)
np.save('data/lstm/y.npy', y)
print("LSTM dataset created and saved to data/lstm/")
