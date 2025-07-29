
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
df = pd.read_csv("data/cleaned_data/cleaned_t20.csv")
print(" CSV loaded. Rows:", len(df))

df = df.dropna(subset=["batsman", "venue", "bowl_team", "runs", "recent_form_avg_runs", "date"])
df = df.rename(columns={"batsman": "player", "bowl_team": "opposition"})
df["date"] = pd.to_datetime(df["date"])
print(" Data cleaned. Rows after dropna:", len(df))

# Encode categorical
df["venue_enc"] = LabelEncoder().fit_transform(df["venue"])
df["oppo_enc"] = LabelEncoder().fit_transform(df["opposition"])
df = df.sort_values(by=["player", "date"])

# Sequence preparation
sequence_length = 5
X, y = [], []

for player in df["player"].unique():
    group = df[df["player"] == player]
    if len(group) <= sequence_length:
        continue
    for i in range(len(group) - sequence_length):
        seq = group.iloc[i:i + sequence_length][["recent_form_avg_runs", "venue_enc", "oppo_enc"]].values
        target = group.iloc[i + sequence_length]["runs"]
        X.append(seq)
        y.append(target)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
print(" Sequence data prepared. Total sequences:", len(X))

# Train model
dataset = CricketDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = LSTMPredictor(input_dim=3)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(" Starting training loop...")
for epoch in range(10):
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f" Epoch {epoch+1}/10 - Loss: {round(total_loss / len(loader), 2)}")

os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), "models/lstm_batting_t20.pt")
print(" Training complete. Model saved to ../models/lstm_batting_t20.pt")
