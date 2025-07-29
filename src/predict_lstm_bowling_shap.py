import pandas as pd
import torch
import torch.nn as nn
import shap
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse

# -----------------------------
# Step 1: Model definition
# -----------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

# -----------------------------
# Step 2: Load data and args
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--player", required=True)
args = parser.parse_args()

df = pd.read_csv("data/cleaned_data/cleaned_odi.csv")
df = df.rename(columns={"bowler": "player", "bat_team": "opposition"})
df["date"] = pd.to_datetime(df["date"])
df = df.dropna(subset=["player", "venue", "opposition", "wickets", "recent_form_avg_wickets", "date"])

# Encode categorical
df["venue_enc"] = LabelEncoder().fit_transform(df["venue"])
df["oppo_enc"] = LabelEncoder().fit_transform(df["opposition"])

# -----------------------------
# Step 3: Extract player sequence
# -----------------------------
df_player = df[df["player"].str.lower() == args.player.lower()].sort_values(by="date")
if len(df_player) < 5:
    print(f" Not enough recent data for SHAP on: {args.player}")
    exit()

last_5 = df_player.tail(5)
input_seq = last_5[["recent_form_avg_wickets", "venue_enc", "oppo_enc"]].values.astype(np.float32)

# -----------------------------
# Step 4: Load trained model
# -----------------------------
model = LSTMPredictor(input_dim=3)
model.load_state_dict(torch.load("models/lstm_bowling_odi.pt"))
model.eval()

# -----------------------------
# Step 5: SHAP compatibility
# -----------------------------
def predict_fn(X_flat):
    X_tensor = torch.tensor(X_flat.reshape(-1, 5, 3), dtype=torch.float32)
    return model(X_tensor).detach().numpy()

background = np.tile(np.mean(input_seq, axis=0), 5).reshape(1, -1)
explainer = shap.Explainer(predict_fn, background)

flat_input = input_seq.flatten().reshape(1, -1)
shap_values = explainer(flat_input)

# -----------------------------
# Step 6: Display SHAP output
# -----------------------------
print(" SHAP values shape:", shap_values.values.shape)
shap.plots.waterfall(shap_values[0])
