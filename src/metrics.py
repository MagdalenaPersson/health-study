import pandas as pd

def summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame ({
    "mean": df.mean(),
    "median": df.median(),
    "min": df.min(),
    "max": df.max()
})