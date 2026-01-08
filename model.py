import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

WINDOW = 5


def calculate_elo(df, k=20, home_advantage=100):
    elo = {}

    elo_home = []
    elo_away = []

    for _, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Initialise teams if missing
        if home not in elo:
            elo[home] = 1500
        if away not in elo:
            elo[away] = 1500

        r_home = elo[home] + home_advantage
        r_away = elo[away]

        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        expected_away = 1 - expected_home

        if row["FTR"] == "H":
            score_home, score_away = 1, 0
        elif row["FTR"] == "A":
            score_home, score_away = 0, 1
        else:
            score_home, score_away = 0.5, 0.5

        elo_home.append(elo[home])
        elo_away.append(elo[away])

        elo[home] += k * (score_home - expected_home)
        elo[away] += k * (score_away - expected_away)

    df["Elo_home"] = elo_home
    df["Elo_away"] = elo_away

    return df, elo



def load_and_train_model(csv_path):
    df = pd.read_csv(csv_path)

    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)

    df, final_elo = calculate_elo(df)

    home = df[["Date", "HomeTeam", "FTHG", "FTAG", "FTR", "Elo_home"]].copy()
    home.columns = ["Date", "Team", "GoalsFor", "GoalsAgainst", "Result", "Elo"]
    home["Home"] = 1
    home["Result"] = home["Result"].map({"H": "W", "A": "L", "D": "D"})

    away = df[["Date", "AwayTeam", "FTAG", "FTHG", "FTR", "Elo_away"]].copy()
    away.columns = ["Date", "Team", "GoalsFor", "GoalsAgainst", "Result", "Elo"]
    away["Home"] = 0
    away["Result"] = away["Result"].map({"H": "L", "A": "W", "D": "D"})

    teams = pd.concat([home, away]).sort_values("Date")
    teams["Points"] = teams["Result"].map({"W": 3, "D": 1, "L": 0})

    teams["GF_avg"] = (
        teams.groupby("Team")["GoalsFor"]
        .rolling(WINDOW)
        .mean()
        .reset_index(level=0, drop=True)
    )

    teams["GA_avg"] = (
        teams.groupby("Team")["GoalsAgainst"]
        .rolling(WINDOW)
        .mean()
        .reset_index(level=0, drop=True)
    )

    teams["Pts_avg"] = (
        teams.groupby("Team")["Points"]
        .rolling(WINDOW)
        .mean()
        .reset_index(level=0, drop=True)
    )

    home_feats = teams[teams["Home"] == 1]
    away_feats = teams[teams["Home"] == 0]

    matches = df.merge(
        home_feats,
        left_on=["Date", "HomeTeam"],
        right_on=["Date", "Team"],
        how="left"
    )

    matches = matches.merge(
        away_feats,
        left_on=["Date", "AwayTeam"],
        right_on=["Date", "Team"],
        how="left",
        suffixes=("_home", "_away")
    )

    X = matches[[
        "GF_avg_home", "GA_avg_home", "Pts_avg_home",
        "GF_avg_away", "GA_avg_away", "Pts_avg_away",
        "Elo_home", "Elo_away"
    ]]

    y = matches["FTR"]

    mask = X.notnull().all(axis=1)
    X = X[mask]
    y = y[mask]

    model = LogisticRegression(
        multi_class="multinomial",
        max_iter=1000
    )
    model.fit(X, y)

    latest_stats = teams.groupby("Team").tail(1).set_index("Team")

    return model, latest_stats
