import pandas as pd
import os

def load_raw_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path)

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df

def convert_data_types(df):
    numeric_columns = ['Mat', 'Inns', 'NO', 'Runs', 'Avg', 'SR', '100s', '50s', 'Overs', 
                       'Wkts', 'Bowl_Avg', 'Econ']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def handle_missing_values(df, strategy="fill_zero"):
    if strategy == "fill_zero":
        return df.fillna(0)
    elif strategy == "drop":
        return df.dropna()
    elif strategy == "fill_median":
        return df.fillna(df.median(numeric_only=True))
    else:
        raise ValueError("Unsupported missing value strategy")

def drop_irrelevant_columns(df):
    columns_to_drop = ['Format', 'HS']
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

def preprocess_and_save(input_path, output_path):
    df = load_raw_data(input_path)
    df = clean_column_names(df)
    df = convert_data_types(df)
    df = handle_missing_values(df, strategy="fill_zero")
    df = drop_irrelevant_columns(df)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")


# Run the preprocessing directly when script is executed
if __name__ == "__main__":
    raw_path = "data/raw/odi_stats.csv"
    cleaned_path = "data/processed/odi_stats_cleaned.csv"
    preprocess_and_save(raw_path, cleaned_path)
