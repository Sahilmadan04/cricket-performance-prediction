import pandas as pd
import os

# Load the cleaned ODI stats dataset
# Make sure the path points to your cleaned ODI CSV
input_path = os.path.join('data', 'processed', 'odi_stats_cleaned.csv')
df = pd.read_csv(input_path)

# -------- Feature Engineering --------

# Batting Impact: Combines runs and strike rate into one measure
df["Batting_Impact"] = (df["Runs"] * df["SR"]) / 100

# Bowling Efficiency: More wickets per over with lower economy is better
df["Bowling_Efficiency"] = (
    df["Wkts"] / df["Overs"].replace(0, 0.1)  # avoid divide-by-zero
) * (1 / df["Econ"].replace(0, 0.1))  # reward low economy

# Format Experience: Simply use number of matches played
df["Format_Experience"] = df["Mat"]

# Career Stage: Relative to the most experienced player in the dataset
df["Career_Stage"] = df["Mat"] / df["Mat"].max()

# Add other optional features as needed:
# df["Boundary_Rate"] = df["4s"] + df["6s"]
# df["Bowling_Strike_Rate"] = df["Balls"] / df["Wkts"]

# -------- Save Final Feature Dataset --------
# This will be used as input for model training
output_path = os.path.join('data', 'processed', 'odi_features_ready.csv')
df.to_csv(output_path, index=False)

print(f"Feature-engineered dataset saved to {output_path}")
