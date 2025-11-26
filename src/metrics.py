import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summerar medel, median, minsta värde och största värde 
    för att kunna applicera på önskad kategori och returnerar det som en DataFrame.
    """
    return pd.DataFrame ({
    "mean": df.mean(),
    "median": df.median(),
    "min": df.min(),
    "max": df.max()
})

def smoker_count(df: pd.DataFrame) -> int:
    """
    Räknar antalet rökare
    """
    return df["smoker"].value_counts(normalize=True)*100

def with_disease(df: pd.DataFrame) -> int:
    """
    Räknar antalet med sjukdomen
    """
    return (df["disease"] == 1).sum()

def ci_mean_bootstrap(x, B=5000):
    """
    Räknar ut 95% konfidensintervall av medelvärdet med bootstrap.

    Returnerar array med bootstrap-sampelmedelvärden, medelvärdet av bootstrap-sampelmedelvärdena,
    nedre och övre gräns för 95% konfidensintervall.
    """
    rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    n = len(x)

    b_means = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(B)])

    boot_mean = np.mean(b_means)
    ci_low, ci_high = np.percentile(b_means, [2.5, 97.5])
    return b_means, boot_mean, ci_low, ci_high

def ci_mean_normal(x):
    """
    Räknar ut 95% konfidensintervall av medelvärdet med normalapproximation
    
    Returnerar medelvärdet av observationerna,
    nedre och övre gräns för 95% konfidensintervallet, 
    standardavvikelse och standardfel (standardavvikelse / sqrt(n))
    """
    mean_x = x.mean()
    sd = x.std(ddof=1)             
    n = x.size
    se = sd / np.sqrt(n)

    z = 1.96
    lo_norm = (mean_x - z*se) 
    hi_norm = (mean_x + z*se)
    return mean_x, lo_norm, hi_norm, sd, se

def bootstrap_mean_difference(x1, x2, B=5000):
    """
    Utför bootstrap-test för skillnaden i medelvärde mellan två grupper.

    Returnerar observerad skillnad i medelvärde mellan 'x1' och 'x2', 
    array med bootstrap-skillnader i medelvärde, Bootstrap p-värde, 
    nedre och övre för 95% konfidensintervallet för skillnaden.
    """
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

def estimate_power_by_simulation(n_no, n_yes, 
                                 mean_no, mean_yes, 
                                 std_no, std_yes, 
                                 alpha=0.05, num_simulations=5000):
    """
    Skattar statistisk power för ett tvåsidigt t-test genom simulering.

    Returnerar uppskattad power.
    """
    rng = np.random.default_rng()
    rejections = 0

    for _ in range(num_simulations):
       
        x = rng.normal(loc=mean_no, scale=std_no, size=n_no)
        y = rng.normal(loc=mean_yes, scale=std_yes, size=n_yes)

        _, pval = stats.ttest_ind(x, y, equal_var=False)

        if pval < alpha:
            rejections += 1

    return rejections / num_simulations

def lr_systolicbp_weight(df:pd.DataFrame):
    """
    Utför en enkel linjär regression mellan vikt och systoliskt blodtryck.
    
    Returnerar skärningspunkt för regressionslinjen, 
    lutningen för regressionslinjen och R² (Determinationskoefficient)
    """
    X = df[["weight"]].values
    y = df["systolic_bp"].values

    linreg = LinearRegression()
    linreg.fit(X, y)

    intercept = float(linreg.intercept_)
    slope = float(linreg.coef_[0])
    r2 = float(linreg.score(X, y))

    return intercept, slope, r2

def disease_per_gender(df: pd.DataFrame):
    """
    Räknar ut andelen sjuka per kön (%)
    """
    stats = df.groupby("sex")["disease"].mean()
    
    return f"Kvinnor:{stats.loc['F']:.1%}, Män:{stats.loc['M']:.1%}"

def pca_systolicbp_weight(df: pd.DataFrame):
    """
    Utför PCA på blodtryck och vikt. 
    
    Returnerar en förklarad variansandel för varje huvudkomponent och 
    koefficienter för varje komponent (rader = komponenter, kolumner = variabler)
    """
    pca_bw = df[["systolic_bp", "weight"]].values

    scaler = StandardScaler()
    pca_bw_scaled = scaler.fit_transform(pca_bw)

    pca = PCA(n_components=2)
    pca.fit(pca_bw_scaled)

    explained = pca.explained_variance_ratio_
    components = pca.components_

    return explained, components

def print_pca_bw(explained, components, names=["systolic_bp", "weight"]):
    """
    Skriver ut Förklarad varians per komponent och komponentvikter (PCA-axlar)
    använder sig av data från funktionen pca_systolicbp_weight
    """
    print("Förklarad varians per komponent:")
    for i, v in enumerate(explained):
        print(f"  PC{i+1}: {v*100:.2f}%")

    print("\nKomponentvikter (PCA-axlar):")
    for i, comp in enumerate(components):
        print(f"\nPC{i+1}:")
        for var, w in zip(names, comp):
            print(f"  {var:12s} -> {w:.3f}")
