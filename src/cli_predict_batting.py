import sys
import os
import joblib
import pandas as pd
import argparse

# Ensure the utils folder is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.feature_engineering import get_recent_runs_avg

def predict_runs(player, venue, opposition, match_format):
    model_path = f"models/{match_format}_batting_model.joblib"
    model = joblib.load(model_path)

    recent_avg = get_recent_runs_avg(player, venue, opposition, match_format)

    features = pd.DataFrame([{
        "player": player,
        "venue": venue,
        "opposition": opposition,
        "recent_form_avg_runs": recent_avg
    }])

    prediction = model.predict(features)[0]
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--venue", required=True)
    parser.add_argument("--opposition", required=True)
    parser.add_argument("--format", choices=["odi", "t20", "ipl"], required=True)
    args = parser.parse_args()

    try:
        predicted_runs = predict_runs(args.player, args.venue, args.opposition, args.format)
        print(f"Predicted Runs for {args.player} in next {args.format.upper()} match: {predicted_runs:.1f}")
    except Exception as e:
        print(str(e))
