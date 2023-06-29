
from functools import partial
import json
import time
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union
import http.client

from team_stat_calcs import calculate_iso

from sql_client import PSQLClient

SEASONS_ENDPOINT = "/mlb/trial/v7/en/league/seasons.json?api_key="
TEAM_HIERARCHY_ENDPOINT = "/mlb/trial/v7/en/league/hierarchy.json?api_key="
SEASONAL_STATS = "/mlb/trial/v7/en/seasons/{team}/REG/teams/{team_id}/statistics.json?api_key="


# Might need to use this season id but also might not
SEASONAL_STATS_URL = "league/seasons/{season_id}/teams/statistics.json?api_key="
SEASONS_URL = "seasons.json?api_key="


API_KEY = "5pnyxrn5ezxedwetq659sadx"


class SportRadarClient:
    """ SportRadarClient class

    """
    _db_client = None
    _client = None
    _seasonal_id_mapping: ClassVar[Optional[Dict[int, str]]] = None
    _team_name_to_id_mapping: ClassVar[Optional[Dict[str, str]]] = None
    _team_id_to_name_mapping: ClassVar[Optional[Dict[str, str]]] = None
    _team_in_game_stats: ClassVar[Optional[Dict[str, Dict[str, Any]]]] = None
    _team_standing_stats: ClassVar[Optional[Dict[str, Dict[str, Any]]]] = None

    # Dictionary Mapping game id to home team and away team
    _schedule_mapping: ClassVar[Optional[Dict[str, Dict[str, str]]]] = None

    @classmethod
    def get_db_client(cls) -> PSQLClient:
        if cls._db_client is None:
            cls._db_client = PSQLClient()
        return cls._db_client

    @classmethod
    def get_client(cls) -> http.client.HTTPSConnection:
        """ Connects to an HTTPSConnection client @api.sportradar.us. """
        if cls._client is None:
            cls._client = http.client.HTTPSConnection("api.sportradar.us")
        return cls._client

    ## Querying and Writing ##
    @classmethod
    def _query_for_seasonal_data(cls) -> None:
        """ Queries the API for season data. It also writes to a json file
            with the data it finds.
            NOTE: Filename: mlb_season_mapping.json"""
        print("No file found, querying the API for Seasonal Data. Hold Tight!")
        # Gets an endpoint for seasons stuff
        json_val = cls._query_api_read_data_and_raise(SEASONS_ENDPOINT + API_KEY)
        # Get Response object and ensure it has a 200 response
        with open("json_data/seasonal_info/mlb_season_mapping.json", "w+") as outfile:
            outfile.write(json_val)

    @classmethod
    def _query_for_team_ids(cls) -> None:
        """ Queries the API for a teams hierarchy and all the data that
            the team is involved with. It also writes to a json file  with the
            data it finds.
            NOTE: Filename: mlb_team_mapping.json """
        print("No file found, querying the API for Team Hierarchal Data. Hold Tight!")
        # Gets an endpoint for seasons stuff
        json_val = cls._query_api_read_data_and_raise(TEAM_HIERARCHY_ENDPOINT + API_KEY)
        # Get Response object and ensure it has a 200 response
        with open("json_data/team_info/mlb_team_mapping.json", "w+") as outfile:
            outfile.write(json_val)

    @classmethod
    def _query_for_in_game_team_stats(cls, year: int, team_id: str, filename: str) -> None:
        """ Queries the API for In Game Team Statistics like hits, runs, lob,
            era, etc.. It then writes to a file that can be read and returned.

            Args:
                year: Year to query for
                team_id: Id of the team querying for
                filename: Name of File to write to """
        print("No file found, querying the API for Team In Game Statistics Data. Hold Tight!")
        IN_GAME_STATS_URL = f"/mlb/trial/v7/en/seasons/{year}/REG/teams/{team_id}/statistics.json?api_key="
        # Gets an endpoint for seasons stuff
        stats = cls._query_api_read_data_and_raise(IN_GAME_STATS_URL + API_KEY)
        statistical_dict = cls._parse_overall_in_game_stats(stats, team_id, year)
        # Get Response object and ensure it has a 200 response
        with open(filename, "w+") as outfile:
            outfile.write(json.dumps(statistical_dict, indent=4))

    @classmethod
    def _query_schedule(cls, year: int, filename: str) -> None:
        """ Queries the API for In Game Team Statistics like hits, runs, lob,
            era, etc.. It then writes to a file that can be read and returned.

            Args:
                year: Year to query for
                team_id: Id of the team querying for
                filename: Name of File to write to """
        print(f"No file found, querying the API for the schedule in {year}. Hold Tight!")
        SCHEDULE_URL = f"/mlb/trial/v7/en/games/{year}/REG/schedule.json?api_key="
        # Gets an endpoint for seasons stuff
        stats = cls._query_api_read_data_and_raise(SCHEDULE_URL + API_KEY)
        json_stats = json.loads(stats)
        schedule_mapping = {}
        id_name_map = cls.get_team_id_to_name_mapping()
        for game in json_stats["games"]:
            id_ = game["id"]
            status = game["status"]
            home_team_id = game["home_team"]
            away_team_id = game["away_team"]

            if id_ in schedule_mapping:
                print("THERE WAS A DUPLICATE GAME.")
                print("Home Team", id_name_map[home_team_id])
                print("Away Team", id_name_map[away_team_id])

            schedule_mapping[id_] = {"home_team": home_team_id, "away_team": away_team_id,
                                     "status": status}

        # Get Response object and ensure it has a 200 response
        with open(filename, "w+") as outfile:
            outfile.write(json.dumps(schedule_mapping, indent=4))

    @classmethod
    def _query_for_standing_team_stats(cls, year: int, filename: str) -> None:
        """ Queries the API for Standings Team Statistics like wins, losses,
            streaks, etc. It then writes to a file that can be read and returned.

            Args:
                year: Year to query for
                team_id: Id of the team querying for
                filename: Name of File to write to """
        print("No file found, querying the API for Team Standings Data. Hold Tight!")
        SEASONAL_STATS_URL = f"/mlb/trial/v7/en/seasons/{year}/REG/standings.json?api_key="
        # Gets an endpoint for seasons stuff
        stats = cls._query_api_read_data_and_raise(SEASONAL_STATS_URL + API_KEY)
        statistical_dict = cls._parse_overall_season_stats(stats)
        # Get Response object and ensure it has a 200 response
        with open(filename, "w+") as outfile:
            outfile.write(json.dumps(statistical_dict, indent=4))

    @classmethod
    def _query_for_individual_game(cls, game_id: str, filename: str) -> Dict[str, Any]:
        """ Queries the API for Standings Team Statistics like wins, losses,
            streaks, etc. It then writes to a file that can be read and returned.

            Args:
                year: Year to query for
                team_id: Id of the team querying for
                filename: Name of File to write to """
        print("No file found, querying the API for Team Standings Data. Hold Tight!")
        GAME_STATS_URL = f"/mlb/trial/v7/en/games/{game_id}/summary.json?api_key="
        # Gets an endpoint for seasons stuff
        stats = cls._query_api_read_data_and_raise(GAME_STATS_URL + API_KEY)
        return json.loads(stats)
        # statistical_dict = cls._parse_overall_season_stats(stats)
        # Get Response object and ensure it has a 200 response
        with open(filename, "w+") as outfile:
            outfile.write(json.dumps(stats, indent=4))

    ## End of Querying and Writing ##

    ## Gather Class Variables storing Data ##
    @classmethod
    def get_schedule_mapping(cls, year: int) -> Dict[str, Dict[str, str]]:
        """ Maps a schedule id to a home team's id and an away teams id """
        print("Checking for Schedule Information ....")
        fname = f"json_data/seasonal_info/mlb_schedule_mapping_{year}.json"
        schedule_q = partial(cls._query_schedule, year=year, filename=fname)
        if cls._schedule_mapping is None:
            cls._schedule_mapping = cls._query_for_file(
                filename=fname, query_fn=schedule_q
            )
        return cls._schedule_mapping

    @classmethod
    def get_unclosed_games(cls, year: int) -> List[str]:
        """ Maps a schedule id to a home team's id and an away teams id """
        sched_mapping = cls.get_schedule_mapping(year) if cls._schedule_mapping is None else cls._schedule_mapping
        return [game for game, info in sched_mapping.items() if info["status"] != "closed"]

    @classmethod
    def gather_individual_game_statistics(cls, year: int, team: Optional[str] = None) -> None:
        schedule_mapping = cls.get_schedule_mapping(year)
        id_name_map = cls.get_team_id_to_name_mapping()
        team_name_to_id_map = cls.get_team_name_to_id_mapping()
        db_client = cls.get_db_client()
        db_client.create_table("individual_stats", "id", [
            "game_id VARCHAR(40) NOT NULL",
            "team_id VARCHAR(40) NOT NULL",
            "in_game_stats json NOT NULL",
            "UNIQUE (team_id, game_id)"
        ])
        if team is not None:
            team_id = team_name_to_id_map[team]
            tries = 0
            for game_id, team_dict in schedule_mapping.items():
                home_team_id, away_team_id = team_dict["home_team"], team_dict["away_team"]
                if team_id not in [home_team_id, away_team_id] or team_dict["status"] != "closed":
                    print("Skipping Upcoming Game", game_id)
                    continue
                res = db_client.select_by_columns(
                    "individual_stats", column_conditions=[
                        f"team_id='{team_id}'",
                        f"game_id='{game_id}'",
                    ]
                )
                if not len(res):
                    stats = cls._query_for_individual_game(game_id, "practice_time.json")
                    home_team_id = stats["game"]["home"]["id"]
                    away_team_id = stats["game"]["away"]["id"]
                    day_or_night = stats["game"]["day_night"]
                    double_header = stats["game"]["double_header"]
                    home_runs = stats["game"]["home"]["runs"]
                    away_runs = stats["game"]["away"]["runs"]
                    away_win = 1 if home_runs < away_runs else 0
                    home_win = 1 if home_runs > away_runs else 0
                    addnl_stats_home = {
                        "d_or_n": day_or_night, "2_h": double_header, "runs": home_runs,
                        "win": home_win, "runs_allowed": away_runs, "is_home": 1,
                        "is_away": 0
                    }
                    addnl_stats_away = {
                        "d_or_n": day_or_night, "2_h": double_header, "runs": away_runs,
                        "win": away_win, "runs_allowed": home_runs,
                        "is_home": 0, "is_away": 1
                    }
                    try:
                        home_team_d = cls._parse_overall_in_game_stats(stats["game"]["home"], home_team_id, year)
                        away_team_d = cls._parse_overall_in_game_stats(stats["game"]["away"], away_team_id, year)
                        home_team_d.update(addnl_stats_home)
                        away_team_d.update(addnl_stats_away)
                    except KeyError as k:
                        print(f"!!!!! FAILED FINDING KEY {k}!!!")
                        print()
                        home_team_d = {}
                        away_team_d = {}
                        raise ValueError("Failed to find stats for game", game_id)
                    new_d = [(game_id, home_team_id, json.dumps(home_team_d),),
                             (game_id, away_team_id, json.dumps(away_team_d),)]
                    db_client.insert_all("individual_stats", ("game_id", "team_id", "in_game_stats"),
                                         new_d)
                    db_client.commit()
                    tries += 1
                    if tries >= 75:
                        print("breaking")
                        break
                res = db_client.select_by_columns(
                    "individual_stats", column_conditions=[
                        f"team_id='{team_name_to_id_map[team]}'",
                    ]
                )
        else:
            res = db_client.select_all("individual_stats")
        return res

    @classmethod
    def get_seasonal_id_mapping(cls) -> Dict[int, str]:
        """ Maps a year to a season id in ascending order. For example:

            {
                2021: 03556285.... ,
                2022: 2923592b.... ,
            }

        """
        print("Checking for seasonal info....")
        if cls._seasonal_id_mapping is None:
            season_json = cls._query_for_file(
                filename="json_data/seasonal_info/mlb_season_mapping.json",
                query_fn=cls._query_for_seasonal_data
            )
            cls._seasonal_id_mapping = {season["year"]: season["id"]
                                        for season in season_json["seasons"]}
        return cls._seasonal_id_mapping

    @classmethod
    def get_team_id_to_name_mapping(cls) -> Dict[str, str]:
        """ Maps a year to a season id in ascending order. For example:

            {
                2021: 03556285.... ,
                2022: 2923592b.... ,
            }

        """
        print("Checking for seasonal info....")
        if cls._team_id_to_name_mapping is None:
            cls.get_team_name_to_id_mapping()
        assert cls._team_id_to_name_mapping is not None
        return cls._team_id_to_name_mapping

    @classmethod
    def get_team_name_to_id_mapping(cls) -> Dict[str, str]:
        """ Maps a team name to a teams id for the API with the capital letters
            and spaces between names. For example:

            {
                Seattle Mariners: 03556285.... ,
                Los Angeles Angels: 2923592b.... ,
            }

            Returns: Mapping of team name to team's API id
        """
        print("Checking for teams info....")
        if cls._team_name_to_id_mapping is None:
            cls._team_name_to_id_mapping = {}
            team_json = cls._query_for_file("json_data/team_info/mlb_team_mapping.json", cls._query_for_team_ids)
            # TODO A lot of optimization can be done on how the teams are stored
            #   And how we query for team ids/data
            for league in team_json["leagues"]:
                for division in league["divisions"]:
                    for team in division["teams"]:
                        team_name = team["market"] + " " + team["name"]
                        cls._team_name_to_id_mapping[team_name] = team["id"]
        cls._team_id_to_name_mapping = {
            v: k for k, v in cls._team_name_to_id_mapping.items()
        }
        return cls._team_name_to_id_mapping

    @classmethod
    def _gather_in_game_team_stats(
        cls, year: int, team_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """ TODO """
        print("Checking for team stats info....")
        fname = f"json_data/team_stats/in_game/mlb_team_statistics_{team_id}_{year}.json"
        q_t_s = partial(cls._query_for_in_game_team_stats, year=year, team_id=team_id,
                        filename=fname)
        year_team_mapping = f"{team_id}_{year}"
        if cls._team_in_game_stats is None:
            cls._team_in_game_stats = {}
            team_stats_json = cls._query_for_file(fname, q_t_s)
            cls._team_in_game_stats = {f"{team_id}_{year}": team_stats_json}
            return cls._team_in_game_stats

        available_data = cls._team_in_game_stats.get(year_team_mapping, None)
        # If we don't have the data for this team update it
        if available_data is None:
            team_stats_json = cls._query_for_file(fname, q_t_s)
            cls._team_in_game_stats.update({year_team_mapping: team_stats_json})
        return cls._team_in_game_stats

    @classmethod
    def _gather_standings_team_stats(cls, year: int) -> Dict[str, Any]:
        """ """
        print("Checking for team standings stats info....")
        fname = f"json_data/team_stats/standings/mlb_team_standing_statistics_{year}.json"
        q_s_t_s = partial(cls._query_for_standing_team_stats, year=year,
                          filename=fname)
        if cls._team_standing_stats is None:
            team_stats_json = cls._query_for_file(fname, q_s_t_s)
            cls._team_standing_stats = team_stats_json
        return cls._team_standing_stats
    ## End of Gather Class Variables storing Data ##

    ## User Accessible Endpoints ##
    @classmethod
    def find_team_in_game_stats_by_year(
        cls, year: int, team_name: Optional[str] = None
    ) -> Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
        team_mappings = cls.get_team_name_to_id_mapping()
        if team_name is None:
            for team_id in team_mappings.values():
                cls._gather_in_game_team_stats(year, team_id)
            return cls._team_in_game_stats
        else:
            team_id = team_mappings.get(team_name, None)
            if team_id is None:
                match_str = f"Invalid team name given: {team_name}. See get_teams_mapping() for details"
                raise ValueError(match_str)
            stats = cls._gather_in_game_team_stats(year, team_id)
            if stats is None:
                raise ValueError(f"Could not gather stats stats of team: {team_name} with id {team_id}.")
            return stats[f"{team_id}_{year}"]

    @classmethod
    def find_team_standings_stats_by_year(
        cls, year: int, team_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """ TODO """
        stats = cls._gather_standings_team_stats(year)
        if stats is None:
            raise ValueError(f"Could not gather standings for year: {year}.")
        if team_name is None:
            return stats
        team_mappings = cls.get_team_name_to_id_mapping()
        team_id = team_mappings.get(team_name, None)
        if team_id is None:
            match_str = f"Invalid team name given: {team_name}. See get_teams_mapping() for details"
            raise ValueError(match_str)
        return stats[f"{team_id}_{year}"]

    @classmethod
    def _query_for_file(cls, filename: str, query_fn: Callable[..., None]) -> Dict[str, Any]:
        """ Queries for a json file before accessing an API. If the file is not found
            then the API is queried for and the file should be written. If
            again the file is not found, raise a ValueError. """
        try:
            with open(filename, "r") as open_map_first:
                print("Found file:", filename, ". Returning without accessing API!")
                return json.load(open_map_first)
        except OSError as e:
            query_fn()
            try:
                with open(filename, "r") as open_map_second:
                    print("Found file:", filename, " after accessing API.")
                    return json.load(open_map_second)
            except OSError as e:
                raise ValueError("Couldn't open up json file for team ids") from e

    @classmethod
    def _parse_overall_in_game_stats(
            cls, stats_str: str, team_id: str, year: int
    ) -> Dict[str, Dict[str, Any]]:
        stats = json.loads(stats_str) if isinstance(stats_str, str) else stats_str
        # Has pitching, hitting, and fielding statistics
        bb_stats = stats["statistics"]

        ## HITTING ##
        overall_h = bb_stats["hitting"]["overall"]
        slg_pct = float(overall_h["slg"])
        batting_avg = float(overall_h["avg"])
        total_abs = float(overall_h["ab"])
        total_runs = float(overall_h["runs"]["total"])
        on_base_pct = float(overall_h["obp"])
        lob = float(overall_h["lob"])
        single = float(overall_h["onbase"]["s"])
        double = float(overall_h["onbase"]["d"])
        triple = float(overall_h["onbase"]["t"])
        hr = float(overall_h["onbase"]["hr"])
        walks = float(overall_h["onbase"]["bb"]) + \
            float(overall_h["onbase"]["ibb"]) + \
            float(overall_h["onbase"]["hbp"])
        reached_on_err = float(overall_h["onbase"]["roe"])
        rbis = float(overall_h["rbi"])
        iso_power = calculate_iso(slg_pct, batting_avg)
            # Want (maybe):
            #  wrc+

        ## End Hitting ##

        ## Pitching ##
        overall_p = bb_stats["pitching"]["overall"]
        # ERA
        era = float(overall_p["era"])
        # Strikeouts recorded for each walk
        kbb = float(overall_p["kbb"])
        # Strikeouts through 9 innings
        k9s = float(overall_p["k9"])
        # Want (maybe):
        # Fielding Independent Pitching
        #
        ## End Pitching ##

        ## Fielding ##
        overall_f = bb_stats["fielding"]["overall"]
        errors = float(overall_f["error"])
        double_plays = float(overall_f["dp"])
        ## End Fielding ##
        return {f"{team_id}_{year}": {
            "slg": slg_pct, "ba": batting_avg, "ab": total_abs, "r": total_runs,
            "obp": on_base_pct, "s": single, "d": double, "t": triple, "hr": hr,
            "lob": lob,
            "w": walks, "roe": reached_on_err, "rbi": rbis, "iso": iso_power,
            "era": era, "kbb": kbb, "k9": k9s,
            "dp": double_plays, "err": errors
        }}

    @classmethod
    def _parse_overall_season_stats(cls, stats_str: str
    ) -> Dict[str, Dict[str, Any]]:
        """ TODO """
        stats = json.loads(stats_str)

        season_data = stats["league"]["season"]
        year = season_data["year"]
        team_data = {}
        for season in season_data["leagues"]:
            for division in season["divisions"]:
                for team in division["teams"]:
                    team_id = team["id"]
                    away_win = team["away_win"]
                    away_loss = team["away_loss"]

                    home_win = team["home_win"]
                    home_loss = team["home_loss"]

                    rank = team["rank"]["league"]

                    away_loss = team["away_loss"]
                    streak_var = team["streak"]
                    is_win_streak = "w" in streak_var[0].lower()
                    win_streak, losing_streak = (float(streak_var[1:]), 0) \
                        if is_win_streak else (0, float(streak_var[1:]))
                    team_data.update(
                        {f"{team_id}_{year}": {
                            "away_w": away_win, "away_l": away_loss, "home_w": home_win,
                            "h_loss": home_loss, "h_win": home_win, "rank": rank,
                            "w_streak": win_streak, "l_streak": losing_streak
                        }}
                    )
        return team_data

    @classmethod
    def _query_api_read_data_and_raise(cls, url: str) -> Any:
        """ Reads or raises an error based on HTTPS request. """
        conn = cls.get_client()
        # Make request
        conn.request("GET", url)
        res = conn.getresponse()
        # Get Response object and ensure it has a 200 response
        if res.status == 200:
            time.sleep(2)
            data = res.read()
            return data.decode("utf-8")
        print("reason", res.reason)
        print("status", res.status)
        print("headers", res.getheaders())
        print("msg", res.msg)
        raise ValueError("Theres some problems getting the data from the sport radar")

# print("Getting the 2023 schedule")
# SportRadarClient().get_schedule_mapping(2023)
# closed = SportRadarClient().get_unclosed_games(2023)
# stats = SportRadarClient().gather_individual_game_statistics(2023, "Seattle Mariners")


