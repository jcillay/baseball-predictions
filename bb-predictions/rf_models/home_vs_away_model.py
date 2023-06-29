from typing import Any, Dict, List

from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from client import SportRadarClient
import pandas as pd
from sql_client.sql_client import PSQLClient


def get_on_avg_stats(team_name: str, year: int, is_home_team: bool) -> Dict[str, Any]:
    """ """
    team_name_to_id_mapping = SportRadarClient.get_team_name_to_id_mapping()
    team_id = team_name_to_id_mapping[team_name]
    res = PSQLClient.select_by_columns("individual_stats", [f"team_id='{team_id}'"])
    total_games = len(res)
    slg_total = sum([s[team_id + f"_{year}"]["slg"] for (_, _, _, s) in res]) / total_games
    batting_avg = sum([s[team_id + f"_{year}"]["ba"] for (_, _, _, s) in res]) / total_games
    at_bats = sum([s[team_id + f"_{year}"]["ab"] for (_, _, _, s) in res]) / total_games
    runs = sum([s[team_id + f"_{year}"]["r"] for (_, _, _, s) in res]) / total_games
    on_base_p = sum([s[team_id + f"_{year}"]["obp"] for (_, _, _, s) in res]) / total_games
    singles = sum([s[team_id + f"_{year}"]["s"] for (_, _, _, s) in res]) / total_games
    doubles = sum([s[team_id + f"_{year}"]["d"] for (_, _, _, s) in res]) / total_games
    triples = sum([s[team_id + f"_{year}"]["t"] for (_, _, _, s) in res]) / total_games
    hrs = sum([s[team_id + f"_{year}"]["hr"] for (_, _, _, s) in res]) / total_games
    lob = sum([s[team_id + f"_{year}"]["lob"] for (_, _, _, s) in res]) / total_games
    w = sum([s[team_id + f"_{year}"]["w"] for (_, _, _, s) in res]) / total_games
    roe = sum([s[team_id + f"_{year}"]["roe"] for (_, _, _, s) in res]) / total_games
    rbis = sum([s[team_id + f"_{year}"]["rbi"] for (_, _, _, s) in res]) / total_games
    iso = sum([s[team_id + f"_{year}"]["iso"] for (_, _, _, s) in res]) / total_games
    # Pitching #
    era = sum([s[team_id + f"_{year}"]["era"] for (_, _, _, s) in res]) / total_games
    kbb = sum([s[team_id + f"_{year}"]["kbb"] for (_, _, _, s) in res]) / total_games
    k9s = sum([s[team_id + f"_{year}"]["k9"] for (_, _, _, s) in res]) / total_games
    double_plays = sum([s[team_id + f"_{year}"]["dp"] for (_, _, _, s) in res]) / total_games
    err = sum([s[team_id + f"_{year}"]["err"] for (_, _, _, s) in res]) / total_games
    runs_allowed = sum([s["runs_allowed"] for (_, _, _, s) in res]) / total_games
    # day_or_night_encoded = 1
    # double_header_encoded = 0
    return {
        # "team_id": [team_to_id[team_name]],
        f'{"home" if is_home_team else "away"}_team': [ team_to_id[team_name]],
        f'{"home" if is_home_team else "away"}_slg': [slg_total],  # Example value for slugging percentage statistic
        f'{"home" if is_home_team else "away"}_ba': [batting_avg],  # Example value for batting average statistic
        f'{"home" if is_home_team else "away"}_ab': [at_bats],  # Example value for at-bats statistic
        f'{"home" if is_home_team else "away"}_r': [runs],  # Example value for runs scored statistic
        f'{"home" if is_home_team else "away"}_obp': [on_base_p],  # Example value for on-base percentage statistic
        f'{"home" if is_home_team else "away"}_s': [singles],  # Example value for singles statistic
        f'{"home" if is_home_team else "away"}_d': [doubles],  # Example value for doubles statistic
        f'{"home" if is_home_team else "away"}_t': [triples],  # Example value for triples statistic
        f'{"home" if is_home_team else "away"}_hr': [hrs],  # Example value for home runs statistic
        f'{"home" if is_home_team else "away"}_lob': [lob],  # Example value for left-on-base statistic
        f'{"home" if is_home_team else "away"}_roe': [roe],  # Example value for reached on error statistic
        f'{"home" if is_home_team else "away"}_rbi': [rbis],  # Example value for runs batted in statistic
        f'{"home" if is_home_team else "away"}_iso': [iso],  # Example value for isolated power statistic
        f'{"home" if is_home_team else "away"}_era': [era],  # Example value for earned run average statistic
        f'{"home" if is_home_team else "away"}_kbb': [kbb],  # Example value for strikeout-to-walk ratio statistic
        f'{"home" if is_home_team else "away"}_k9': [k9s],  # Example value for strikeouts per 9 innings statistic
        f'{"home" if is_home_team else "away"}_dp': [double_plays],  # Example value for double plays turned statistic
        f'{"home" if is_home_team else "away"}_err': [err],  # Example value for errors statistic
        f'{"home" if is_home_team else "away"}_runs_allowed': [runs_allowed],  # Example value for errors statistic
    }

def get_all_game_headers() -> List[str]:
    headers_for_both_teams = [
        "_team", "_slg", "_ba",
        "_ab", "_r", "_obp", "_s", "_d", "_t", "_hr", "_lob", "_roe", "_rbi", "_iso",
        "_era", "_kbb", "_k9", "_dp", "_err",  "_runs_allowed",
    ]
    home_team_headers = list(map(lambda h: "home" + h, headers_for_both_teams))
    away_team_headers = list(map(lambda h: "away" + h, headers_for_both_teams))
    generic_headers = ["winning_team"] # "d_or_n", "2_h"]
    return home_team_headers + away_team_headers + generic_headers


def train_model():
    dataframes = []
    # for _, team_name in SportRadarClient.get_team_id_to_name_mapping().items():
    #     file_path = f'in_game_stats_{team_name.strip()}_2023.csv'
    #     try:
    #         df = pd.read_csv(file_path)
    #     except FileNotFoundError:
    #         continue
    #     dataframes.append(df)
    combined_df = pd.read_csv("game_by_game_stats.csv")

    # Concatenate all DataFrames into a single DataFrame
    # combined_df = pd.concat(dataframes, ignore_index=True)

    # Select relevant columns for training
    # selected_columns = ['home_team', 'away_team', 'winner', 'lob', 'era', 'betting_odds']
    features = get_all_game_headers()
    df = combined_df[features]

    # Perform label encoding for categorical columns
    categorical_columns = ['d_or_n', '2_h']
    categorical_columns = []
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])

    # Split data into features (X) and target variable (y)
    X = df.drop('winning_team', axis=1)
    y = df['winning_team']
    train_data, test_data = train_test_split(df, test_size=0.2)
    # Split data into training and testing sets
    X_train = train_data.drop("winning_team", axis=1)
    y_train = train_data["winning_team"]

    # Prepare the testing data
    X_test = test_data.drop("winning_team", axis=1)
    y_test = test_data["winning_team"]

    # Create and train the model
    model = RandomForestRegressor(n_estimators=200)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    # r2 = r2_score(y_test, predictions)
    print("MEAN ABSOLUTE ERROR", mae)
    print("MEAN SQUARED ERROR", mse)
    # print("R2", r2)

    # Create a Random Forest Classifier model
    # model = RandomForestClassifier()
    # model = RandomForestRegressor(n_estimators=200)
    # # Train the model
    # model.fit(X_train, y_train)


    # Calculate the accuracy of the model
    # accuracy = accuracy_score(y_test, predictions)
    # print("Models Training Accuracy!", accuracy)
    return label_encoder, model