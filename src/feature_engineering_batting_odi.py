import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/cleaned_data/cleaned_odi.csv")
df['date'] = pd.to_datetime(df['date'])

# Sort and compute rolling form
df = df.sort_values(by=['batsman', 'date'])
df['rolling_avg_runs'] = (
    df.groupby('batsman')['runs']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)

# Save output
os.makedirs("data/features", exist_ok=True)
df[['batsman', 'date', 'runs', 'rolling_avg_runs']].to_csv("data/features/odi_batting_form_scores.csv", index=False)
print("Saved to data/features/odi_batting_form_scores.csv")
