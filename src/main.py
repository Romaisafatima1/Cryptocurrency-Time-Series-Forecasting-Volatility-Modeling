import kagglehub

btc_df = kagglehub.dataset_load(
    kagglehub.KaggleDatasetAdapter.PANDAS,
    "imranbukhari/comprehensive-btcusd-1m-data",
    "BTCUSD_1m_Combined_Index.csv"
)

eth_df = kagglehub.dataset_load(
    kagglehub.KaggleDatasetAdapter.PANDAS,
    "imranbukhari/comprehensive-ethusd-1m-data",
    "ETHUSD_1m_Combined_Index.csv"
)
