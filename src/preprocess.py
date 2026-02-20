import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def run_preprocessing():
    path = "data/raw/household_power_consumption.txt"
    df = pd.read_csv(path, sep=";", low_memory=False)

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime")

    df = df.replace("?", None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill()

    df_hourly = df.resample("h").mean()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_hourly)

    df_scaled = pd.DataFrame(scaled, index=df_hourly.index, columns=df_hourly.columns)

    os.makedirs("data/processed", exist_ok=True)
    df_scaled.to_csv("data/processed/processed.csv")

if __name__ == "__main__":
    run_preprocessing()