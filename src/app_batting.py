import streamlit as st
import pandas as pd
import joblib
import os

# Supported formats
formats = {
    "odi": {
        "model": "../models/train_model_batting_odi.joblib",
        "csv": "../data/cleaned_data/cleaned_odi.csv"
    },
    "t20": {
        "model": "../models/train_model_batting_t20.joblib",
        "csv": "../data/cleaned_data/cleaned_t20.csv"
    },
    "ipl": {
        "model": "../models/train_model_batting_ipl.joblib",
        "csv": "../data/cleaned_data/cleaned_ipl.csv"
    }
}

# UI
st.title("Cricket Batting Performance Predictor")
match_format = st.selectbox("Select Match Format", list(formats.keys()))

# Load model and data
model_path = formats[match_format]["model"]
csv_path = formats[match_format]["csv"]

if not os.path.exists(model_path) or not os.path.exists(csv_path):
    st.error("Model or dataset not found. Please check your paths.")
    st.stop()

model = joblib.load(model_path)
df = pd.read_csv(csv_path)
df = df.dropna(subset=["batsman", "venue", "bowl_team", "runs", "recent_form_avg_runs"])
df = df.rename(columns={"batsman": "player", "bowl_team": "opposition"})

# Dropdowns
player = st.selectbox("Select Player", sorted(df["player"].unique()))
venue = st.selectbox("Select Venue", sorted(df["venue"].unique()))
opposition = st.selectbox("Select Opposition", sorted(df["opposition"].unique()))

# Use CLI logic — use CSV's precomputed recent form
def get_recent_form(player_name):
    rows = df[df["player"].str.lower() == player_name.lower()]
    return rows["recent_form_avg_runs"].mean()

if st.button("Predict Runs"):
    recent_form = get_recent_form(player)
    input_df = pd.DataFrame([{
        "player": player,
        "venue": venue,
        "opposition": opposition,
        "recent_form_avg_runs": recent_form
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Runs: {round(prediction, 2)}")
