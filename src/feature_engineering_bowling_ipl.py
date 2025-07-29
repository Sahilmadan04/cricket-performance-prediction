
import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/cleaned_data/cleaned_ipl.csv")
df['date'] = pd.to_datetime(df['date'])

# Sort and compute rolling form
df = df.sort_values(by=['bowler', 'date'])
df['rolling_avg_wickets'] = (
    df.groupby('bowler')['wickets']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)

# Save output
os.makedirs("data/features", exist_ok=True)
df[['bowler', 'date', 'wickets', 'rolling_avg_wickets']].to_csv("data/features/ipl_bowling_form_scores.csv", index=False)
print("Saved to data/features/ipl_bowling_form_scores.csv")
