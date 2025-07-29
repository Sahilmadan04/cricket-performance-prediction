import sys
import os
import joblib
import pandas as pd
import argparse

# Ensure the utils folder is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_recent_wickets_avg(player, venue, opposition, match_format, venue_col, opposition_col):
    """Return recent bowling average only if player is a genuine bowler (has taken wickets)."""
    try:
        path = f"data/cleaned_data/cleaned_{match_format}.csv"
        df = pd.read_csv(path)

        # Filter for this player's bowling records
        player_matches = df[df['bowler'].str.lower() == player.lower()]

        # Only consider matches where player actually took wickets
        valid_matches = player_matches[player_matches["wickets"] > 0]

        if len(valid_matches) == 0:
            # Player has never taken a wicket — likely not a bowler
            return 0.0

        # Otherwise, take the most recent match stat
        most_recent = valid_matches.iloc[-1]
        return float(most_recent['recent_form_avg_wkts'])

    except Exception as e:
        print(f" Warning in get_recent_wickets_avg: {str(e)}")
        return 0.0

def predict_wickets(player, venue, opposition, match_format):
    """Predict wickets for a bowler in an upcoming match"""
    try:
        model_path = f"models/{match_format}_bowling_model.joblib"
        data_path = f"data/cleaned_data/cleaned_{match_format}.csv"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        df = pd.read_csv(data_path)

        #  STEP 1: Filter to include only players with enough bowling records
        bowling_counts = df["bowler"].value_counts()
        eligible_bowlers = bowling_counts[bowling_counts >= 5].index  # 5+ appearances
        
        if player not in eligible_bowlers:
            raise ValueError(
                f" {player} is not a regular bowler in the {match_format.upper()} dataset."
            )

        model = joblib.load(model_path)

        recent_avg = get_recent_wickets_avg(
            player, venue, opposition, match_format, "venue", "opposition"
        )

        # Create DataFrame with proper feature names and types
        features = pd.DataFrame([{
            "player": str(player),
            "venue": str(venue),
            "opposition": str(opposition),
            "recent_form_avg_wkts": float(recent_avg)
        }])

        prediction = model.predict(features)[0]
        return max(0, prediction)  # Ensure non-negative prediction
    
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict wickets for a bowler in the next match."
    )
    parser.add_argument("--player", required=True, help="Name of the bowler")
    parser.add_argument("--venue", required=True, help="Venue of the match")
    parser.add_argument("--opposition", required=True, help="Opposition team")
    parser.add_argument(
        "--format", 
        required=True, 
        choices=["odi", "t20i", "ipl"], 
        help="Match format"
    )
    
    args = parser.parse_args()

    try:
        predicted_wkts = predict_wickets(
            args.player, args.venue, args.opposition, args.format
        )
        print(
            f" Predicted Wickets for {args.player} at {args.venue} "
            f"vs {args.opposition} ({args.format.upper()}): {predicted_wkts:.2f}"
        )
    except Exception as e:
        print(f" Could not generate prediction: {str(e)}")
