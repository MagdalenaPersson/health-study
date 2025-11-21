import pandas as pd
import numpy as np

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
    return (df["disease"] == 1).sum()

def ci_mean_bootstrap(x, B=5000):
    rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)

    b_means = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(B)])

    boot_mean = np.mean(b_means)
    ci_low, ci_high = np.percentile(b_means, [2.5, 97.5])
    return b_means, boot_mean, ci_low, ci_high

def ci_mean_normal(x):
    mean_x = x.mean()
    sd = x.std(ddof=1)             
    n = x.size
    se = sd / np.sqrt(n)

    z = 1.96
    lo_norm = (mean_x - z*se) 
    hi_norm = (mean_x + z*se)
    return mean_x, lo_norm, hi_norm, sd, se

def bootstrap_mean_difference(x1, x2, B=5000):

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    obs_diff = x1.mean() - x2.mean()

    boot_diffs = np.empty(B) 
    for i in range(B):
        s1 = np.random.choice(x1, size=len(x1), replace=True)
        s2 = np.random.choice(x2, size=len(x2), replace=True)
        boot_diffs[i] = s1.mean() - s2.mean()

    p_boot = np.mean(np.abs(boot_diffs) >= abs(obs_diff))
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    return obs_diff, boot_diffs, p_boot, ci_low, ci_high