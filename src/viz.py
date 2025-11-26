import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
import numpy as np
import metrics as M
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

@dataclass
class HealthAnalyzer:
    df: pd.DataFrame

    def hist_mean(self, ax, values, bins=20, title="", xlabel="", ylabel="Antal deltagare", kde=True, grid=True):
        """
        Ritar ett histogram för en variabel och markerar dess medelvärde. 
        """
        sns.histplot(values, bins=bins, kde=kde, edgecolor="black", ax=ax)
        ax.axvline(values.mean(), color="red", linestyle="--", linewidth="1", label="Medelvärde")
        ax.set_title(title, fontsize=15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.grid(True, axis="y")
        plt.tight_layout()
        return ax

    def box_plot(self, ax, df, column, by, title="", xlabel="", ylabel="", showmeans=True):
        """
        Ritar en boxplot för vald variabel grupperad efter en kategorivariabel.
        """
        df.boxplot(column=column, by=by, ax=ax, showmeans=showmeans)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.suptitle("") 
        plt.tight_layout()
        return ax

    def bar_smoker(self, ax, counts, title="Andel rökare"):
        """
        Ritar ett stapeldiagram över andelen rökare i procent.
        """
        ax.bar(counts.index.astype(str), 
            counts.values, 
            alpha=0.7, 
            edgecolor="black", 
            color=["tab:green", "tab:purple"]
            )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Andel (%)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Icke rökare", "Rökare"])
        ax.grid(True, axis="y")
        plt.tight_layout()
        return ax
    
    def bar_disease_simulation(self, ax, n=1000, title="Verklig andel sjuka vs simulation"):
        """
        Ritar ett stapeldiagram över verklig andel sjuka vs simulation.
        """
        sick = M.with_disease(self.df) / len(self.df)
        sim = np.random.choice([0, 1], size=n, p=[1 - sick, sick])
        sim_mean = sim.mean()

        ax.bar(
        ["Verklig andel", "Simulerad andel"],
        [sick * 100, sim_mean * 100],
        alpha=0.7,
        edgecolor="black",
        color=["tab:blue", "tab:orange"]
        )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Andel sjuka (%)")
        ax.grid(True, axis="y")

        plt.tight_layout()
        return ax
    
    def plot_bootstrap_vs_normal(self,
            ax1, ax2, *,
            b_means, mean_x,
            ci_low, ci_high,
            lo_norm, hi_norm,
            title_one="Bootstrap och normalapproximering av medelvärdet",
            title_two="Jämförelse av 95% CI: Bootstrap vs Normalapproximation"):
        """
        Tar fram två olika grafer.
        Graf 1. är ett histogram som visar Bootstrap och normalapproximering av medelvärdet för blodtryck.
        Graf 2. är en errorbar som visar en jämförelse av 95% CI: Bootstrap vs Normalapproximation
        """
    
        ax1.hist(b_means, bins=30, alpha=0.7, edgecolor="black")
        ax1.axvline(mean_x, color="tab:green", linestyle="--", linewidth=2, label="Stickprovsmedel")
        ax1.axvline(ci_low, color="tab:red", linestyle="--", label="Bootstrap 2.5%")
        ax1.axvline(ci_high, color="tab:red", linestyle="--", label="Bootstrap 97.5%")
        ax1.axvline(lo_norm, color="tab:blue", linestyle="--", label="Norm low")
        ax1.axvline(hi_norm, color="tab:blue", linestyle="--", label="Norm high")
        ax1.set_title(title_one)
        ax1.set_xlabel("Blodtryck (medel av stickprov)")
        ax1.set_ylabel("Antal")
        ax1.grid(True, axis="y")
        ax1.legend()
        
        boot_lower = mean_x - ci_low
        boot_upper = ci_high - mean_x
        norm_lower = mean_x - lo_norm
        norm_upper = hi_norm - mean_x

        ax2.errorbar("Bootstrap", [mean_x],
                    yerr=[[boot_lower], [boot_upper]],
                    elinewidth=2, markersize=8, fmt="o", capsize=8, label="Bootstrap CI")

        ax2.errorbar("Normalapproximation", [mean_x],
                    yerr=[[norm_lower], [norm_upper]],
                    elinewidth=2, markersize=8, fmt="o", capsize=8, label="Normal CI")

        ax2.set_ylabel("Medelvärde Blodtryck (mmHg)")
        ax2.set_title(title_two)
        ax2.grid(True, axis="y", linestyle="--")
        ax2.legend()

        return ax1, ax2
    
    def plot_lr_systolicbp_weight(self, ax):
        """
        Ritar en scatter med en linjär regression: Blodtryck ~ vikt
        """
        X = self.df[["weight"]].values
        y = self.df["systolic_bp"].values

        linreg = LinearRegression()
        linreg.fit(X, y)
        r2 = float(linreg.score(X, y))
        grid_x = np.linspace(X.min(), X.max(), 200)
        grid_y = linreg.predict(grid_x.reshape(-1, 1))

        ax.scatter(X, y, alpha=0.6, edgecolor="black", label="Data")
        ax.plot(grid_x, grid_y, linewidth=2, color="tab:red", linestyle="--", label=f"Trendlinje(R²={r2:.2f})")
        ax.set_xlabel("Vikt (kg)")
        ax.set_ylabel("Blodtryck (mmHg)")
        ax.set_title("Linjär regression: Blodtryck ~ vikt")
        ax.grid(True, color="gray", alpha=0.6)
        ax.legend()

        return ax
    
    def disease_per_gender_bar(self, ax=None, title="Andel sjuka per kön"):
        """
        Ritar ett stapeldiagram över andelen sjuka per kön.
        """
        stats = self.df.groupby("sex")["disease"].mean() * 100

        if ax is None:
            fig, ax = plt.subplots()
        
        ax.bar(
        stats.index.astype(str),
        stats.values,
        alpha=0.7,
        edgecolor="black",
        color=["tab:red", "tab:blue"]
        )

        ax.set_xticks(range(len(stats.index)))
        ax.set_xticklabels(["Kvinnor" if x=="F" else "Män" for x in stats.index])

        ax.set_title(title)
        ax.set_xlabel("Kön")
        ax.set_ylabel("Andel sjuka (%)")
        ax.grid(True, axis="y")

        plt.tight_layout()
        return ax
    
    def pca_plot(self, ax):
        """
        Ritar en scatter över PCA: Blodtryck + vikt
        """
        pca_bw = self.df[["systolic_bp", "weight"]].values

        scaler = StandardScaler()
        pca_bw_scaled = scaler.fit_transform(pca_bw)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(pca_bw_scaled)

        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolor="black")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA av blodtryck + vikt")

        return ax
