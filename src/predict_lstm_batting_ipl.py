import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import shap
import argparse
import os

# Define model
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

# Parse input
parser = argparse.ArgumentParser()
parser.add_argument("--player", type=str, required=True)
args = parser.parse_args()
player_name = args.player

# Load data
df = pd.read_csv("data/cleaned_data/cleaned_ipl.csv")
df = df.dropna(subset=["batsman", "venue", "bowl_team", "runs", "recent_form_avg_runs", "date"])
df = df.rename(columns={"batsman": "player", "bowl_team": "opposition"})
df["date"] = pd.to_datetime(df["date"])
df["venue_enc"] = LabelEncoder().fit_transform(df["venue"])
df["oppo_enc"] = LabelEncoder().fit_transform(df["opposition"])
df = df.sort_values(by=["player", "date"])

# Player sequence
player_df = df[df["player"].str.lower() == player_name.lower()]
if len(player_df) < 5:
    print(f" Not enough recent data for SHAP on: {player_name}")
    exit()

# Input sequence
input_seq = player_df.iloc[-5:][["recent_form_avg_runs", "venue_enc", "oppo_enc"]].values.astype(np.float32)
flat_input = input_seq.flatten().reshape(1, -1)  # shape (1, 15)

# Load model
model = LSTMPredictor(input_dim=3)
model.load_state_dict(torch.load("models/lstm_batting_ipl.pt"))
model.eval()

# Define prediction wrapper for SHAP
def predict_fn(X_flat):
    X_seq = X_flat.reshape((-1, 5, 3))
    x_tensor = torch.tensor(X_seq, dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).numpy()

# SHAP explainer using sample background
background = np.tile(np.mean(input_seq, axis=0), 5).reshape(1, -1)  # shape (1, 15)
explainer = shap.Explainer(predict_fn, background)
shap_values = explainer(flat_input)

# Display results
prediction = predict_fn(flat_input)[0]
print(f"\n Predicted Runs for {player_name}: {round(prediction, 2)}")
print(" SHAP Feature Impact:")

features = ["recent_form_avg_runs", "venue_enc", "oppo_enc"]
for i in range(5):
    for j, feat in enumerate(features):
        val = shap_values.values[0][i * 3 + j]
        print(f"Match {-5+i+1:>2} | {feat:25s} → {round(val, 3)}")
