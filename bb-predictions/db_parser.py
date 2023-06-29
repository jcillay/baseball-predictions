""" EXCEL WRITING: Functions that write to excel files. """

import csv
import time
from typing import Any, Dict, List

import pandas as pd
# from sklearn.base import r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from client import SportRadarClient
from sql_client import PSQLClient

team_to_id = {
    "Arizona Diamondbacks": 1,
    "Atlanta Braves": 2,
    "Baltimore Orioles": 3,
    "Boston Red Sox": 4,
    "Chicago Cubs": 5,
    "Chicago White Sox": 6,
    "Cincinnati Reds": 7,
    "Cleveland Guardians": 8,
    "Colorado Rockies": 9,
    "Detroit Tigers": 10,
    "Houston Astros": 11,
    "Kansas City Royals": 12,
    "Los Angeles Angels": 13,
    "Los Angeles Dodgers": 14,
    "Miami Marlins": 15,
    "Milwaukee Brewers": 16,
    "Minnesota Twins": 17,
    "New York Mets": 18,
    "New York Yankees": 19,
    "Oakland Athletics": 20,
    "Philadelphia Phillies": 21,
    "Pittsburgh Pirates": 22,
    "San Diego Padres": 23,
    "Seattle Mariners": 24,
    "San Francisco Giants": 25,
    "St. Louis Cardinals": 26,
    "Tampa Bay Rays": 27,
    "Texas Rangers": 28,
    "Toronto Blue Jays": 29,
    "Washington Nationals": 30
}
id_to_team = {v: k for k, v in team_to_id.items()}

def write_game_by_game_statistics():
    game_ids = set(PSQLClient.select_column("individual_stats", ["game_id"]))
    id_to_name_mapping = SportRadarClient.get_team_id_to_name_mapping()

    headers = get_all_game_headers()
    print("total headers", len(headers))
    with open("game_by_game_stats.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for game_id, in game_ids:
            teams = PSQLClient.select_by_columns("individual_stats", [f"game_id='{game_id}'"])
            home_team_stats = []
            away_team_stats = []
            away_win, home_win = None, None
            for (id, _, team_id, stats) in teams:
                is_home = int(stats["is_home"])
                in_order_stats = [v for k, v in stats[team_id + f"_{2023}"].items() if k != "w" and k != "win"]
                day_or_night = str(stats["d_or_n"])
                double_header = bool(stats["2_h"])
                if is_home:
                    home_team_stats.append(team_to_id[id_to_name_mapping[team_id]])
                    # TODO Replace this with the year of the stats
                    home_team_stats += in_order_stats + [stats["runs_allowed"]]
                    home_win = [stats["win"]]
                else:
                    away_team_stats.append(team_to_id[id_to_name_mapping[team_id]])
                    # TODO Replace this with the year of the stats
                    away_team_stats += in_order_stats + [stats["runs_allowed"]]
                    away_win = [stats["win"]]

                # print("away length", len(away_team_stats))
                # print("home length", len(home_team_stats))
            if home_win is None or away_win is None:
                print("game_id", game_id)
                raise ValueError("Both teams can not be home or away!")
            elif home_win == away_win:
                print("game_id", game_id)
                raise ValueError("Both teams can not win the game")
            winning_team = home_team_stats[0] if home_win == 1 else away_team_stats[0]
            total_stats =  home_team_stats + away_team_stats + [winning_team] # day_or_night, double_header]
            # print("total_stats", len(total_stats))
            writer.writerow( total_stats)

def write_team_game_stats(year: int, team: str):
    data = SportRadarClient.gather_individual_game_statistics(year, team)
    mapping = SportRadarClient.get_schedule_mapping(year)
    id_to_name_mapping = SportRadarClient.get_team_id_to_name_mapping()
    in_game_stats = ["slg", "ba", "ab", "r", "obp", "s", "d", "t", "hr",
                     "lob", "w", "roe", "rbi", "iso", "era", "kbb", "k9", "dp", "err",]
    game_info = ["d_or_n", "2_h", "runs", "win", "runs_allowed", "is_home", "is_away"]
    # Then when we predict we will be able to access betting odds for the prediction
    with open(f'in_game_stats_{team.strip()}_{year}.csv', 'w+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        seen_first = False
        for d in data:
            _, game_id, team_id, new_d = d
            game_data = mapping.get(game_id)
            if not seen_first:
                headers = ["team_id", "home_team", "away_team", "winning_team"] + in_game_stats + game_info
                writer.writerow(headers)
                seen_first = True
            if int(new_d["win"]):
                winning_team = game_data["home_team"] if new_d["is_home"] else game_data["away_team"]
            else:
                winning_team = game_data["away_team"] if new_d["is_home"] else game_data["home_team"]
            winning_team_train_id = team_to_id[id_to_name_mapping[winning_team]]
            home_team_train_id = team_to_id[id_to_name_mapping[game_data["home_team"]]]
            away_team_train_id = team_to_id[id_to_name_mapping[game_data["away_team"]]]
            team_train_id = team_to_id[id_to_name_mapping[team_id]]
            valid_data = [team_train_id, home_team_train_id, away_team_train_id, winning_team_train_id]
            data_game = [new_d[team_id + f"_{year}"][key] for key in in_game_stats]
            info_game = [new_d[key] for key in game_info]

            # write the data
            writer.writerow(valid_data + data_game + info_game)

def get_on_avg_stats_and_predict(
        team_name: str, year: int, label_encoder: LabelEncoder, model: RandomForestClassifier,
        home_team: str, away_team: str, day_or_night: str, double_header: bool
    ) -> pd.DataFrame:
    """ """
    team_name_to_id_mapping = SportRadarClient.get_team_name_to_id_mapping()
    team_id = team_name_to_id_mapping[team_name]
    home_team_id = team_name_to_id_mapping[home_team]
    away_team_id = team_name_to_id_mapping[away_team]
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
    is_home = 1 if team_id == home_team_id else 0
    is_away = 1 if team_id == away_team_id else 0
    # home_team_encoded = label_encoder.transform([home_team_id])
    # away_team_encoded = label_encoder.transform([away_team_id])
    # team_id_encoded = label_encoder.transform([team_id])

    # day_or_night_encoded = label_encoder.transform([day_or_night])
    # double_header_encoded = label_encoder.transform([double_header])
    input_data = pd.DataFrame({
        # "team_id": [team_to_id[team_name]],
        'home_team': [team_to_id[home_team]],
        'away_team': [team_to_id[away_team]],
        'slg': [slg_total],  # Example value for slugging percentage statistic
        'ba': [batting_avg],  # Example value for batting average statistic
        'ab': [at_bats],  # Example value for at-bats statistic
        'r': [runs],  # Example value for runs scored statistic
        'obp': [on_base_p],  # Example value for on-base percentage statistic
        's': [singles],  # Example value for singles statistic
        'd': [doubles],  # Example value for doubles statistic
        't': [triples],  # Example value for triples statistic
        'hr': [hrs],  # Example value for home runs statistic
        'lob': [lob],  # Example value for left-on-base statistic
        'roe': [roe],  # Example value for reached on error statistic
        'rbi': [rbis],  # Example value for runs batted in statistic
        'iso': [iso],  # Example value for isolated power statistic
        'era': [era],  # Example value for earned run average statistic
        'kbb': [kbb],  # Example value for strikeout-to-walk ratio statistic
        'k9': [k9s],  # Example value for strikeouts per 9 innings statistic
        'dp': [double_plays],  # Example value for double plays turned statistic
        'err': [err],  # Example value for errors statistic
        # 'd_or_n': [day_or_night_encoded],  # Example value for day or night game indicator statistic
        # '2_h': [double_header_encoded],  # Example value for two-hit games statistic
        'runs_allowed': [runs_allowed],  # Example value for runs allowed statistic
        'is_home': [is_home],  # Example value for home game indicator statistic
        'is_away': [is_away],  # Example value for away game indicator statistic
    })
    prediction = model.predict(input_data)
    return prediction

def get_list_of_features() -> List[str]:
    return [
        # 'team_id',
        'home_team', 'away_team', 'slg', 'ba',
        'ab', 'r', 'obp', 's', 'd', 't', 'hr', 'lob', 'roe', 'rbi', 'iso',
        'era', 'kbb', 'k9', 'dp', 'err', 'd_or_n', '2_h', 'runs_allowed',
        'is_home', 'is_away', 'winning_team'
    ]

# # List of CSV files
# csv_files = ['file1.csv', 'file2.csv', 'file3.csv']

# # Read CSV files and concatenate into a single DataFrame
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
    model = RandomForestRegressor(n_estimators=100)
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

write_game_by_game_statistics()
encoder, model = train_model()
# day_or_night_encoded = encoder.transform(["D"])
# double_header_encoded = encoder.transform([False])

kc_stats = get_on_avg_stats("Seattle Mariners" , 2023, False, encoder, "D", False)
tb_stats = get_on_avg_stats("New York Yankees", 2023, True, encoder, "D", False)
tb_stats.update(kc_stats)
# tb_stats.update({"d_or_n": [day_or_night_encoded], "2_h": [double_header_encoded] })

kc_vs_tb = pd.DataFrame(tb_stats)
# combined_df = pd.concat(new_dfs, ignore_index=True)
# print(combined_df)
print(kc_vs_tb)
prediction = model.predict(kc_vs_tb)
print("PREDICTION = ", prediction)
print("PREDICTION = ",  id_to_team[prediction[0]])

bos_stats = get_on_avg_stats("Boston Red Sox", 2023, False, encoder, "D", False)
min_stats = get_on_avg_stats("Minnesota Twins", 2023, True, encoder, "D", False)
min_stats.update(bos_stats)
# min_stats.update({"d_or_n": [day_or_night_encoded], "2_h": [double_header_encoded] })

new_dfs = pd.DataFrame(min_stats)
# combined_df = pd.concat(new_dfs, ignore_index=True)
# print(combined_df)
print(new_dfs)
prediction = model.predict(new_dfs)[0]
print("PREDICTION = ", prediction)
print("PREDICTION = ",  id_to_team[prediction])


# boston_prediction = get_on_avg_stats_and_predict(
#     "Boston Red Sox", 2023, encoder, model, "Minnesota Twins", "Boston Red Sox",
#     "D", False
# )
# min_prediction = get_on_avg_stats_and_predict(
#     "Minnesota Twins", 2023, encoder, model, "Minnesota Twins", "Boston Red Sox",
#     "D", False
# )
# print("BOSTON PREDICTION", boston_prediction)
# print("MINNESOTA PREDICTION", min_prediction)
# kc_prediction = get_on_avg_stats_and_predict(
#     "Kansas City Royals", 2023, encoder, model, "Tampa Bay Rays", "Kansas City Royals",
#     "D", False
# )
# tp_prediction = get_on_avg_stats_and_predict(
#     "Tampa Bay Rays", 2023, encoder, model, "Tampa Bay Rays", "Kansas City Royals",
#     "D", False
# )
# print("kc PREDICTION", kc_prediction)
# print("tp PREDICTION", tp_prediction)

# TO train data
# Split the data into two segments:
#   home team and Away team
#       For each teams data
#           Get the average overall stats after each game
#           Then we can predict on the average overall stats
#

# Then when predicting We will plug in the current average stats for each team.
#

# Two models
#   One predicts the winning team
#   And one predicts the runs scored for each

# for t, i in SportRadarClient.get_team_name_to_id_mapping().items():
#     write_team_to_excel(2023, t)
# Predict the winner for two teams
# home_team = SportRadarClient.get_team_name_to_id_mapping()["Tampa Bay Rays"]
# away_team = SportRadarClient.get_team_name_to_id_mapping()["Baltimore Orioles"]

# lob = 8  # Example value for left-on-base statistic
# era = 3.5  # Example value for earned run average statistic
# betting_odds = 2.5  # Example value for betting odds

    # Perform label encoding for categorical features

# # Create a DataFrame for the input data
# # PRedicting if the home team will win
# input_data = pd.DataFrame({
#     "team_id": away_team_encoded,
#     'home_team': home_team_encoded,
#     'away_team': away_team_encoded,
#     'slg': [0],  # Example value for slugging percentage statistic
#     'ba': [0],  # Example value for batting average statistic
#     'ab': [0],  # Example value for at-bats statistic
#     'r': [0],  # Example value for runs scored statistic
#     'obp': [0],  # Example value for on-base percentage statistic
#     's': [0],  # Example value for singles statistic
#     'd': [0],  # Example value for doubles statistic
#     't': [0],  # Example value for triples statistic
#     'hr': [0],  # Example value for home runs statistic
#     'lob': [0],  # Example value for left-on-base statistic
#     'w': [0],  # Example value for wins statistic
#     'roe': [0],  # Example value for reached on error statistic
#     'rbi': [0],  # Example value for runs batted in statistic
#     'iso': [0],  # Example value for isolated power statistic
#     'era': [0],  # Example value for earned run average statistic
#     'kbb': [0],  # Example value for strikeout-to-walk ratio statistic
#     'k9': [0],  # Example value for strikeouts per 9 innings statistic
#     'dp': [0],  # Example value for double plays turned statistic
#     'err': [0],  # Example value for errors statistic
#     'd_or_n': [0],  # Example value for day or night game indicator statistic
#     '2_h': [0],  # Example value for two-hit games statistic
#     'runs': [0],  # Example value for runs scored statistic
#     # 'win': [0],  # Example value for win statistic
#     'runs_allowed': [0],  # Example value for runs allowed statistic
#     'is_home': [0],  # Example value for home game indicator statistic
#     'is_away': [0],  # Example value for away game indicator statistic
# })
# # CAN GET:
# # For betting:
# #   Runs allows
# #   Hits
# #   errors
# #   is_home
# #   is_away
# #   day or night
# #   strikeouts
# #
# #

# # Make predictions
# prediction = model.predict(input_data)

# # Perform label decoding for the predicted winner
# predicted_winner = label_encoder.inverse_transform(prediction)[0]

# # Print the predicted winner
# print(f"The predicted winner between {home_team} and {away_team} is: {predicted_winner}")

# raise ValueError

# ### Problems !!!! ###


# # # List of team names
# team_names = ['team1', 'team2', 'team3', ..., 'team30']


# # # # Load data for each team and create a combined dataset
# # combined_data = pd.DataFrame()

# # for team_id, team_name in SportRadarClient.get_team_id_to_name_mapping().items():
# #     file_path = f'in_game_stats_{team_name.strip()}_2023.csv'  # Assuming each team's CSV file is named as 'team_name.csv'
# #     team_data = pd.read_csv(file_path)
# #     combined_data = combined_data

# # # Select the relevant features from the combined dataset
# combined_data = pd.concat(dataframes, ignore_index=True)

# # Select relevant columns for training
# # selected_columns = ['home_team', 'away_team', 'winner', 'lob', 'era', 'betting_odds']
# features = combined_data[get_list_of_features()]

# # # Select the target variable (game outcome)
# target = combined_data['win']

# # # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # # Create and train the RandomForestClassifier model
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train, y_train)

# # # Make predictions on the test set
# predictions = rf_model.predict(X_test)

# # # Evaluate the model's performance
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy:", accuracy)

# # import pandas as pd
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_absolute_error

# # Load the data from your dataset
data = pd.read_csv('mlb_game_data.csv')  # Replace 'mlb_game_data.csv' with the actual file name

# # Select the relevant features
features = data[['home_team_runs', 'away_team_runs',
                 'home_team_ERA', 'away_team_ERA',
                 'home_team_batting_average', 'away_team_batting_average',
                 'home_team_strikeouts', 'away_team_strikeouts',
                 'home_team_stolen_bases', 'away_team_stolen_bases']]

# # Calculate the run differential
data['run_differential'] = data['home_team_runs'] - data['away_team_runs']

# # Select the target variable (winning team and run differential)
target = data[['home_team_win', 'run_differential']]

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# # Create and train the RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# # Make predictions on the test set
predictions = rf_model.predict(X_test)

# # Evaluate the model's performance (mean absolute error for run differential)
mae = mean_absolute_error(y_test['run_differential'], predictions[:, 1])
print("Mean Absolute Error for Run Differential:", mae)

# # Get the predicted winning team
predicted_winner = predictions[:, 0].round().astype(int)
print("Predicted Winning Team:")
print(predicted_winner)


# write_team_to_excel(2023, "Seattle Mariners")
for team in SportRadarClient.get_team_name_to_id_mapping().keys():
    print("Gathering Team", team)
