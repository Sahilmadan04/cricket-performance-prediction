import streamlit as st
import pandas as pd
import joblib
import os

# Supported formats and their paths
formats = {
    "odi": {
        "model": "../models/train_model_bowling_odi.joblib",
        "csv": "../data/cleaned_data/cleaned_odi.csv"
    },
    "t20": {
        "model": "../models/train_model_bowling_t20.joblib",
        "csv": "../data/cleaned_data/cleaned_t20.csv"
    },
    "ipl": {
        "model": "../models/train_model_bowling_ipl.joblib",
        "csv": "../data/cleaned_data/cleaned_ipl.csv"
    }
}

# Page title
st.title("Cricket Bowling Performance Predictor")

# Match format selector
match_format = st.selectbox("Select Match Format", list(formats.keys()))

# Get model and CSV paths
model_path = formats[match_format]["model"]
csv_path = formats[match_format]["csv"]

# Check existence
if not os.path.exists(model_path) or not os.path.exists(csv_path):
    st.error("Model or dataset not found. Please check your files.")
    st.stop()

# Load model and dataset
model = joblib.load(model_path)
df = pd.read_csv(csv_path)

# Standardize and clean up data
df = df.rename(columns={"bowler": "player", "bowl_team": "opposition"})
df = df.dropna(subset=["player", "venue", "opposition", "wickets", "recent_form_avg_wkts"])

# UI Dropdowns
player = st.selectbox("Select Bowler", sorted(df["player"].unique()))
venue = st.selectbox("Select Venue", sorted(df["venue"].unique()))
opposition = st.selectbox("Select Opposition", sorted(df["opposition"].unique()))

# Use CLI logic: fetch from CSV directly
def get_recent_form(player_name):
    rows = df[df["player"].str.lower() == player_name.lower()]
    return rows["recent_form_avg_wkts"].mean()

# Predict on button click
if st.button("Predict Wickets"):
    recent_form = get_recent_form(player)
    input_df = pd.DataFrame([{
        "player": player,
        "venue": venue,
        "opposition": opposition,
        "recent_form_avg_wkts": recent_form
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Wickets: {round(prediction, 2)}")
