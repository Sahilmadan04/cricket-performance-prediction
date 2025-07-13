import pandas as pd

# Load cleaned T20I dataset
df = pd.read_csv("data/processed/t20i_stats_cleaned.csv")

# Convert Overs to float if it's not already
df['Overs'] = pd.to_numeric(df['Overs'], errors='coerce')

# Feature 1: Batting consistency = Runs per match
df['Runs_Per_Match'] = df['Runs'] / df['Mat']

# Feature 2: Boundary Contribution Estimate (based on 100s and 50s as a proxy)
df['Boundary_Contribution_Score'] = df['100s'] * 100 + df['50s'] * 50

# Feature 3: Batting Impact Score = Runs × SR (weighted run rate)
df['Batting_Impact'] = df['Runs'] * df['SR']

# Feature 4: Bowling Strike Rate = Balls per wicket = (Overs × 6) / Wkts
df['Bowl_StrikeRate'] = (df['Overs'] * 6) / df['Wkts'].replace(0, pd.NA)

# Feature 5: Bowling Impact Score = Wickets × Economy
df['Bowling_Impact'] = df['Wkts'] * df['Econ']

# Feature 6: Match Participation Ratio = (Bat Innings + Bowl Innings) / Matches
df['Match_Contribution'] = (df['Inns'] + df['Bowl_Inns']) / df['Mat']

# Save the updated file
df.to_csv("data/processed/t20i_features_ready.csv", index=False)

print("✅ T20I features engineered and saved to t20i_features_ready.csv")
