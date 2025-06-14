from pycoingecko import CoinGeckoAPI
import pandas as pd
import os

cg = CoinGeckoAPI()

def fetch_data(coin_id, days=365):
    print(f"Fetching last {days} days of data for {coin_id}")
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
    df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def save_csv(df, filename):
    os.makedirs('data/raw', exist_ok=True)
    filepath = os.path.join('data/raw', filename)
    df.to_csv(filepath, index=False)
    print(f"Saved to {filepath}")

# Fetch and save BTC and ETH data
btc = fetch_data('bitcoin', days=365)
save_csv(btc, 'btc_data.csv')

eth = fetch_data('ethereum', days=365)
save_csv(eth, 'eth_data.csv')