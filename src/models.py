import torch
import torch.nn as nn
from prophet import Prophet
import pandas as pd

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def train_prophet(train_df):
    df = train_df.reset_index()
    df.columns = ["ds", "y"]
    model = Prophet()
    model.fit(df)
    return model

def prophet_predict(model, future_df):
    forecast = model.predict(future_df)
    return forecast