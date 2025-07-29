import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import os

print(" Loading and preparing data...")

class CricketDataset(Dataset):
    def __init__(self, sequences, targets):
        self.X = sequences
        self.y = targets

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

# Load data
df = pd.read_csv("data/cleaned_data/cleaned_odi.csv")
df = df.dropna(subset=["bowler", "venue", "bat_team", "wickets", "recent_form_avg_wickets", "date"])
df = df.rename(columns={"bowler": "player", "bat_team": "opposition"})
df["date"] = pd.to_datetime(df["date"])

df["venue_enc"] = LabelEncoder().fit_transform(df["venue"])
df["oppo_enc"] = LabelEncoder().fit_transform(df["opposition"])
df = df.sort_values(by=["player", "date"])

# Prepare sequences
X, y = [], []
for player in df["player"].unique():
    group = df[df["player"] == player]
    if len(group) <= 5:
        continue
    for i in range(len(group) - 5):
        seq = group.iloc[i:i+5][["recent_form_avg_wickets", "venue_enc", "oppo_enc"]].values
        target = group.iloc[i+5]["wickets"]
        X.append(seq)
        y.append(target)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(f" Prepared {len(X)} sequences.")

dataset = CricketDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMPredictor(input_dim=3)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(" Training model...")
for epoch in range(10):
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f" Epoch {epoch+1}/10 - Loss: {round(total_loss / len(loader), 4)}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm_bowling_odi.pt")
print(" Model saved to models/lstm_bowling_odi.pt")
