from pathlib import Path

import pandas as pd


TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"


def load_train_data(path: str | Path = "data/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_xy(df: pd.DataFrame):
    y = df[TARGET_COL].map({"Y": 1, "N": 0}).astype(int)
    x = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    return x, y
