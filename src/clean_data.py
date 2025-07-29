
import pandas as pd
import os

# Paths to your original CSV files
odi_path = r"C:\University Work\cricket-performance-prediction\cricket-performance-prediction\data\odi.csv"
t20_path = r"C:\University Work\cricket-performance-prediction\cricket-performance-prediction\data\t20.csv"
ipl_path = r"C:\University Work\cricket-performance-prediction\cricket-performance-prediction\data\ipl.csv"

def clean_cricket_data(filepath):
    df = pd.read_csv(filepath)

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows with missing critical fields
    df = df.dropna(subset=['batsman', 'bowler', 'runs', 'wickets'])

    # Fill missing values for venue and teams
    df['venue'] = df['venue'].fillna('Unknown')
    df['bat_team'] = df['bat_team'].fillna('Unknown')
    df['bowl_team'] = df['bowl_team'].fillna('Unknown')

    # Standardize text columns
    text_columns = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip().str.title()

    # Convert numeric columns
    numeric_columns = ['runs', 'wickets', 'overs', 'runs_last_5', 'wickets_last_5', 'total']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

# Clean each dataset
cleaned_odi = clean_cricket_data(odi_path)
cleaned_t20 = clean_cricket_data(t20_path)
cleaned_ipl = clean_cricket_data(ipl_path)

# Save cleaned versions
output_dir = r"C:\University Work\cricket-performance-prediction\cricket-performance-prediction\data\cleaned_data"
os.makedirs(output_dir, exist_ok=True)
cleaned_odi.to_csv(os.path.join(output_dir, "cleaned_odi.csv"), index=False)
cleaned_t20.to_csv(os.path.join(output_dir, "cleaned_t20.csv"), index=False)
cleaned_ipl.to_csv(os.path.join(output_dir, "cleaned_ipl.csv"), index=False)

print(" Cleaning complete. Cleaned files saved to 'cleaned_data/' folder.")
