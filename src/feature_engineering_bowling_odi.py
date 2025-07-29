import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/cleaned_data/cleaned_odi.csv")
df['date'] = pd.to_datetime(df['date'])

# Sort and compute rolling average wickets for each bowler
df = df.sort_values(by=['bowler', 'date'])
df['recent_form_avg_wkts'] = (
    df.groupby('bowler')['wickets']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
)

# Save to new feature file
os.makedirs("data/features", exist_ok=True)
df[['bowler', 'date', 'wickets', 'recent_form_avg_wkts']].to_csv("data/features/odi_bowling_form_scores.csv", index=False)
print(" Saved to data/features/odi_bowling_form_scores.csv")
