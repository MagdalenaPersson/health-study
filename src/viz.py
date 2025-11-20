import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass

@dataclass
class HealthAnalyzer:
    df: pd.DataFrame

    def hist_mean(self, ax, values, bins=20, title="", xlabel="", ylabel="Antal deltagare", kde=True, grid=True):
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
        df.boxplot(column=column, by=by, ax=ax, showmeans=showmeans)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.suptitle("") 
        plt.tight_layout()
        return ax

    def bar_smoker(self, ax, counts, title="Andel rökare"):
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
    
    def bar_disease_simulation(self, ax, real, simulated, title="Verklig andel sjuka vs simulation"):
        ax.bar(
        ["Verklig andel", "Simulerad andel"],
        [real * 100, simulated * 100],
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