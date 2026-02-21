import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
import wandb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models import LSTMModel
from feature_engineering import run_feature_engineering
from preprocess import run_preprocessing

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/training.log", level=logging.INFO)

def create_sequences(data, target_col, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][target_col])
    return np.array(X), np.array(y)

def walk_forward_validation(df, window, hidden_size):
    target_col = 0
    split_size = int(len(df) * 0.7)
    train_data = df.iloc[:split_size].values
    test_data = df.iloc[split_size:].values

    X_train, y_train = create_sequences(train_data, target_col, window)
    X_test, y_test = create_sequences(test_data, target_col, window)

    if len(X_train) == 0 or len(X_test) == 0:
        return 999, 999, 999

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    model = LSTMModel(X_train.shape[2], hidden_size, 1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    logging.info("Starting walk-forward fold 1")

    for epoch in range(5):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output.squeeze(), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = model(X_test).detach().numpy().flatten()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

    return mae, rmse, mape

def objective(trial):
    window = trial.suggest_int("window_size", 12, 48)
    hidden_size = trial.suggest_int("hidden_size", 16, 64)

    df = pd.read_csv("data/processed/features.csv", index_col=0)
    mae, rmse, mape = walk_forward_validation(df, window, hidden_size)

    wandb.log({"mae": mae, "rmse": rmse, "mape": mape})

    return rmse

def main():
    run_preprocessing()
    run_feature_engineering()

    wandb.init(project="VoltVision")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    best_params = study.best_params

    with open("results/best_params.json", "w") as f:
        json.dump(best_params, f)

    wandb.finish()

if __name__ == "__main__":
    main()