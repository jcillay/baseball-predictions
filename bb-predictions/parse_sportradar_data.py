""" """

import csv
from typing import Optional

import pandas as pd
# from sklearn.base import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

import client as client

year = 2023
def get_sports_info(year: int, team_name: Optional[str] = None):
    client = client.SportRadarClient()
    get_team_p_h_f_stats = client.find_team_in_game_stats_by_year(year, team_name)
    finding_stats_for_standings = client.find_team_standings_stats_by_year(year)
    total_dict_mapping = {}
    for k, v in get_team_p_h_f_stats.items():
        total_dict_mapping[k] = v[k]
        total_dict_mapping[k].update(finding_stats_for_standings[k])
        total_dict_mapping[k]["team_name"] = client.get_team_id_to_name_mapping()[k[:-5]]
        # print("Mapped Team stats", client.get_team_id_to_name_mapping()[k[:-5]])
        # print(total_dict_mapping[k])
        # print()
        # print()
        # print()
    return total_dict_mapping

def write_to_excel_file(year: int) -> None:
    sports_data_mapping = get_sports_info(year)
    updated_headers = False
    with open('output.csv', 'w') as f_output:
        for id, v in sports_data_mapping.items():
            print("id", id)
            print(v)
            if not updated_headers:
                csv_output = csv.DictWriter(f_output, fieldnames=["id"] + sorted(v.keys()))
                csv_output.writeheader()
                updated_headers = True
            v.update({"id": id})
            csv_output.writerow(v)


def read_output_and_train(year: int) -> None:
    sports_data_mapping = get_sports_info(year)
    keys = []
    for id, v in sports_data_mapping.items():
        del v["team_name"]
        keys = sorted(v.keys())
    print(keys)
    # Gather and preprocess the data
    # Assuming you have the data in a CSV file named 'baseball_data.csv'
    data = pd.read_csv('output.csv')

    # Select the relevant features
    features = data[keys]
    train_data, test_data = train_test_split(features, test_size=0.2)


    # Select the target variable

    # Prepare the training data
    X_train = train_data
    y_train = train_data

    # Prepare the testing data
    X_test = test_data
    y_test = test_data

    # Create and train the model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    print("MS", mse)
    # print("r2", r2)
    print("mae", mae)

    print("Mean Squared Error:", mse)

read_output_and_train(2023)
