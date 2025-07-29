import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib

# Load full ODI dataset
df = pd.read_csv("data/cleaned_data/cleaned_ipl.csv")

# Optional: Reduce size for faster training (e.g., 20,000 rows)
df = df.sample(n=20000, random_state=42)

# Drop rows with missing key values
df = df.dropna(subset=["batsman", "venue", "bowl_team", "runs", "recent_form_avg_runs"])

# Rename for prediction consistency
df = df.rename(columns={"batsman": "player", "bowl_team": "opposition"})

# Features and labels
X = df[["player", "venue", "opposition", "recent_form_avg_runs"]]
y = df["runs"]

# Preprocessing pipeline
categorical_features = ["player", "venue", "opposition"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

# Fast & accurate regressor (LightGBM)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(n_estimators=100, random_state=42))
])

# Train-test split (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/ipl_batting_model.joblib")
print(" Fast model trained and saved as models/ipl_batting_model.joblib")
