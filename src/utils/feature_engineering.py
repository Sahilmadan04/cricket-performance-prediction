import pandas as pd

def get_recent_runs_avg(player_name, venue, opposition, format_):
    path = f"data/cleaned_data/cleaned_{format_}.csv"
    df = pd.read_csv(path)

    required_cols = {"batsman", "venue", "bowl_team", "runs", "date"}
    missing_cols = required_cols - set(df.columns.str.lower())
    if missing_cols:
        raise ValueError(f"Required columns are missing: {missing_cols}")

    df = df[df["batsman"].str.lower() == player_name.lower()]
    df = df[df["venue"].str.lower() == venue.lower()]
    df = df[df["bowl_team"].str.lower() == opposition.lower()]

    df = df.sort_values(by="date", ascending=False)
    recent = df.head(5)

    if recent.empty:
        raise ValueError(f"Not enough batting data for player: {player_name} in {format_.upper()} format.")

    return recent["runs"].mean()
    path = f"data/cleaned_data/cleaned_{format_}.csv"
    df = pd.read_csv(path)

    # FIX missing 'bowl_team'
    if 'opposition' in df.columns and 'bowl_team' not in df.columns:
        df.rename(columns={'opposition': 'bowl_team'}, inplace=True)

    df = df[df['batsman'].str.lower() == player_name.lower()]
    df = df[df['venue'].str.lower() == venue.lower()]
    df = df[df['bowl_team'].str.lower() == opposition.lower()]
    df = df.sort_values(by='date', ascending=False)
    recent = df.head(5)

    if recent.empty:
        raise ValueError(f"Not enough batting data for player: {player_name} in {format_.upper()} format.")
    return recent['runs'].mean()

    path = f"data/cleaned_data/cleaned_{format_}.csv"
    df = pd.read_csv(path)

    print("Initial rows:", len(df))
    df['batsman_clean'] = df['batsman'].str.strip().str.lower()
    df['venue_clean'] = df['venue'].str.strip().str.lower()
    df['bowl_team_clean'] = df['bowl_team'].str.strip().str.lower()

    print("Unique batsmen:", df['batsman_clean'].unique()[:10])

    player = player_name.strip().lower()
    venue = venue.strip().lower()
    opposition = opposition.strip().lower()

    df = df[df['batsman_clean'] == player]
    print("After batsman filter:", len(df))

    df = df[df['venue_clean'] == venue]
    print("After venue filter:", len(df))

    df = df[df['bowl_team_clean'] == opposition]
    print("After opposition filter:", len(df))

    df = df.sort_values(by='date', ascending=False)
    recent = df.head(5)

    if recent.empty:
        raise ValueError(f"Not enough batting data for player: {player_name} in {format_.upper()} format.")

    return recent['runs'].mean()
    
def get_recent_wickets_avg(df, player_col, wickets_col, date_col, venue_col, opposition_col, new_col_name="recent_form_avg_wkts"):
    """
    Calculates the average wickets taken in last 5 matches for a bowler in same venue and against same opposition.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[new_col_name] = 0.0

    for idx in range(len(df)):
        player = df.loc[idx, player_col]
        venue = df.loc[idx, venue_col]
        opposition = df.loc[idx, opposition_col]
        current_date = df.loc[idx, date_col]

        past_matches = df[
            (df[player_col] == player) &
            (df[venue_col] == venue) &
            (df[opposition_col] == opposition) &
            (df[date_col] < current_date)
        ].sort_values(by=date_col, ascending=False).head(5)

        if not past_matches.empty:
            avg_wkts = past_matches[wickets_col].mean()
        else:
            avg_wkts = 0.0

        df.at[idx, new_col_name] = avg_wkts

    return df

