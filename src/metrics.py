import pandas as pd

def summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame ({
    "mean": df.mean(),
    "median": df.median(),
    "min": df.min(),
    "max": df.max()
})

def smoker_count(df: pd.DataFrame) -> int:
    return df["smoker"].value_counts(normalize=True)*100

def with_disease(df: pd.DataFrame) -> int:
    return df[df["disease"] == 1].sum()
    