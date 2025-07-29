
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import joblib

df = pd.read_csv("data/cleaned_data/cleaned_t20.csv")
df = df.dropna(subset=["bowler", "venue", "bat_team", "wickets", "recent_form_avg_wkts"])
df = df.rename(columns={"bowler": "player", "bat_team": "opposition"})

X = df[["player", "venue", "opposition", "recent_form_avg_wkts"]]
y = df["wickets"]

categorical_features = ["player", "venue", "opposition"]
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LGBMRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "models/t20_bowling_model.joblib")
print(" Fast model trained and saved as models/t20_bowling_model.joblib")
