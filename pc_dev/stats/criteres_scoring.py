import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Chargement
df = pd.read_csv("scores_tctdp.csv").dropna(subset=["Total"])

CRITERES = ["Cn", "Cm", "Ne", "Cl", "Cth", "Bfd", "Bfi", "Pe", "Hu", "Uc_b", "Uc_c", "Uc_d"]
couleurs  = {"C0": "#438FD2", "C1": "#DE4503"}


# Statistiques par critère
print("Moyennes par critère (global)")
moyennes_globales = df[CRITERES].mean().round(2).sort_values(ascending=False)
print(moyennes_globales)
print()

# Moyennes par critère et par condition
print("Moyennes par critère selon la condition")
print(df.groupby("Condition")[CRITERES].mean().round(2).T)
print()

# Tableau des différences de moyennes entre C0 et C1
C0_df = df[df["Condition"] == "C0"]
C1_df = df[df["Condition"] == "C1"]

print("Différences de moyennes C1 − C0 par critère")
diff_moyennes = (C1_df[CRITERES].mean() - C0_df[CRITERES].mean()).round(2).sort_values(ascending=False)
print(diff_moyennes)
print()

# Cohen's d par critère : quel critère différencie le plus C0 et C1 ?
print("Cohen's d par critère (différenciation C0 vs C1)")
def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (b.mean() - a.mean()) / pooled if pooled != 0 else 0

ds = {}
for c in CRITERES:
    d = cohens_d(C0_df[c].dropna(), C1_df[c].dropna())
    ds[c] = round(d, 3)
ds_series = pd.Series(ds).sort_values(ascending=False)
print(ds_series)


# Graphique (4 panels)
fig, axes = plt.subplots(1, 3, figsize=(22, 5))
fig.suptitle("Analyse des critères de scoring", fontsize=13, fontweight="bold")


# Panel 1 : score moyen par critère en points (plus en %)
ax = axes[0]
moy_triees      = moyennes_globales.sort_values()
couleurs_barres = ["#B0C4DE" if v < 2 else "#438FD2" for v in moy_triees]
ax.barh(moy_triees.index, moy_triees.values, color=couleurs_barres, alpha=0.85)
# Score affiché à droite de chaque barre
for i, v in enumerate(moy_triees.values):
    ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
ax.set_xlabel("Score moyen (points)", fontsize=10)
ax.set_title("Score moyen\npar critère", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)


# Panel 2 : moyennes C0 vs C1 par critère
ax2 = axes[1]
x     = np.arange(len(CRITERES))
width = 0.35

for i, cond in enumerate(["C0", "C1"]):
    moyennes = df[df["Condition"] == cond][CRITERES].mean()
    offset   = (i - 0.5) * width
    ax2.bar(x + offset, moyennes, width, label=cond,
            color=couleurs[cond], alpha=0.75)

ax2.set_xticks(x)
ax2.set_xticklabels(CRITERES, fontsize=9, rotation=45, ha="right")
ax2.set_ylabel("Score moyen", fontsize=10)
ax2.set_title("Moyenne par critère\nselon la condition", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)


# Panel 3 : Cohen's d par critère
# Un d positif = C1 > C0, négatif = C0 > C1
ax4 = axes[2]
ds_triees  = ds_series.sort_values()
couleurs_d = ["#DE4503" if v >= 0 else "#438FD2" for v in ds_triees]
ax4.barh(ds_triees.index, ds_triees.values, color=couleurs_d, alpha=0.8)
ax4.axvline(0,    color="#555", linewidth=1)
ax4.axvline(0.5,  color="#aaa", linewidth=0.8, linestyle="--")  # seuil petit effet
ax4.axvline(-0.5, color="#aaa", linewidth=0.8, linestyle="--")
ax4.axvline(0.8,  color="#aaa", linewidth=0.8, linestyle=":")   # seuil grand effet
ax4.axvline(-0.8, color="#aaa", linewidth=0.8, linestyle=":")
ax4.set_xlabel("Cohen's d  [C1 − C0]", fontsize=10)
ax4.set_title("Quel critère différencie\nle plus C0 et C1 ?", fontsize=11)
ax4.spines[["top", "right"]].set_visible(False)
ax4.grid(axis="x", alpha=0.3)
ax4.text(0.52, -0.7, "petit", fontsize=7, color="#999")
ax4.text(0.82, -0.7, "grand", fontsize=7, color="#999")


plt.tight_layout()
plt.savefig("criteres_scoring.png", dpi=150, bbox_inches="tight")
plt.close()
