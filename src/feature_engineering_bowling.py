import pandas as pd
import os

def generate_recent_form_avg_wkts(format_):
    try:
        path = f"data/cleaned_data/cleaned_{format_}.csv"
        
        # Read data with explicit type handling
        df = pd.read_csv(path, dtype={
            'bowler': str,
            'wickets': float,
            'venue': str,
            'opposition': str
        })
        
        # Convert date column safely
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Remove rows with invalid dates
        
        # Ensure wickets column is numeric
        df['wickets'] = pd.to_numeric(df['wickets'], errors='coerce')
        df = df.dropna(subset=['wickets'])  # Remove rows with invalid wickets
        
        # Sort by bowler and date
        df = df.sort_values(by=['bowler', 'date'])
        
        # Calculate rolling average safely
        df['recent_form_avg_wkts'] = (
            df.groupby('bowler')['wickets']
            .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )
        
        # Save with all numeric columns properly formatted
        df.to_csv(path, index=False)
        print(f" recent_form_avg_wkts added and saved to {path}")
        return True
    
    except Exception as e:
        print(f" Error processing {format_}: {str(e)}")
        return False

# Run for all formats with error handling
for fmt in ['ipl', 'odi', 't20']:
    success = generate_recent_form_avg_wkts(fmt)
    if not success:
        print(f" Failed to process {fmt} format")