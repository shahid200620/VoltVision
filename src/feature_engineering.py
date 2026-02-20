import pandas as pd
import os

def run_feature_engineering():
    path = "data/processed/processed.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    target = df.columns[0]

    df["lag_1"] = df[target].shift(1)
    df["lag_24"] = df[target].shift(24)

    df["rolling_mean_24"] = df[target].rolling(24).mean()

    df = df.dropna()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/features.csv")

if __name__ == "__main__":
    run_feature_engineering()