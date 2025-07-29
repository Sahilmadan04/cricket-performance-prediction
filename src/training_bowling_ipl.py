import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib

# Load full IPL dataset
df = pd.read_csv("data/cleaned_data/cleaned_ipl.csv")

# Optional: sample for faster training
df = df.sample(n=20000, random_state=42)

# Drop missing values
df = df.dropna(subset=["bowler", "venue", "bat_team", "wickets", "recent_form_avg_wkts"])

# Rename for consistency
df = df.rename(columns={"bowler": "player", "bat_team": "opposition"})

# Select features and target
X = df[["player", "venue", "opposition", "recent_form_avg_wkts"]]
y = df["wickets"]

# One-hot encode categorical features
categorical_features = ["player", "venue", "opposition"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

# LightGBM model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save
model.fit(X_train, y_train)
joblib.dump(model, "models/ipl_bowling_model.joblib")
print(" Fast model trained and saved as models/ipl_bowling_model.joblib")
