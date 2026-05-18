import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Chargement
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])

ordre   = ["Jeune enfant", "Enfant", "Adolescent", "Adulte"]
couleurs = {"C0": "#438FD2", "C1": "#DE4503"}


# Statistiques descriptives par âge
print("Moyennes par tranche d'âge et condition")
print(df.groupby(["Tranche_age", "Condition"])["Total"]
        .agg(["mean", "std", "count"])
        .round(2))
print()
print("Moyennes par tranche d'âge (toutes conditions)")
print(df.groupby("Tranche_age")["Total"]
        .agg(["mean", "std", "count"])
        .round(2))
print()
print("Adolescent (N=2) et Adulte (N=3) : effectifs trop petits")
print("pour des tests statistiques, résultats purement descriptifs")


# Graphique
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Score TCT-DP en fonction de la tranche d'âge",
             fontsize=13, fontweight="bold")


# Graphique 1 : barres C0 vs C1 par tranche d'âge 
ax = axes[0]
x     = np.arange(len(ordre))
width = 0.35

for i, cond in enumerate(["C0", "C1"]):
    moyennes, erreurs, ns = [], [], []
    for age in ordre:
        groupe = df[(df["Tranche_age"] == age) & (df["Condition"] == cond)]["Total"]
        moyennes.append(groupe.mean() if len(groupe) > 0 else 0)
        # Erreur standard = SD / √N (donne la précision de la moyenne)
        erreurs.append(groupe.std() / np.sqrt(len(groupe)) if len(groupe) > 1 else 0)
        ns.append(len(groupe))

    offset = (i - 0.5) * width
    ax.bar(x + offset, moyennes, width, label=cond,
           color=couleurs[cond], alpha=0.75, zorder=3)
    # Barres d'erreur (± erreur standard)
    ax.errorbar(x + offset, moyennes, yerr=erreurs,
                fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
    # N affiché dans chaque barre
    for j, (m, n) in enumerate(zip(moyennes, ns)):
        if n > 0:
            ax.text(x[j] + offset, 1, f"N={n}", ha="center", va="bottom",
                    fontsize=8, color="white", fontweight="bold", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(ordre, fontsize=10)
ax.set_ylabel("Score moyen (± SE)", fontsize=10)
ax.set_title("Moyenne par âge et condition", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)


# Graphique 2 : boxplot par tranche d'âge (toutes conditions)
ax2 = axes[1]

data_par_age = [df[df["Tranche_age"] == age]["Total"].dropna().values
                for age in ordre]

palette = ["#7EC8E3", "#438FD2", "#1B5E9E", "#0D3B6E"]

bp = ax2.boxplot(data_par_age, positions=range(len(ordre)), widths=0.5,
                 patch_artist=True,
                 medianprops=dict(color="white", linewidth=2.5),
                 whiskerprops=dict(linewidth=1.5),
                 capprops=dict(linewidth=1.5),
                 flierprops=dict(marker="o", alpha=0.5))

for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for element in ["whiskers", "caps"]:
    for i, line in enumerate(bp[element]):
        line.set_color(palette[i // 2])

# Points individuels avec jitter
for i, (age, color) in enumerate(zip(ordre, palette)):
    vals   = df[df["Tranche_age"] == age]["Total"].dropna()
    jitter = np.random.normal(0, 0.07, size=len(vals))
    ax2.scatter([i] + jitter, vals, color=color, s=40, alpha=0.7, zorder=5)
 
# N affiché AU-DESSUS de chaque boxplot pour éviter le chevauchement avec les étiquettes
for i, age in enumerate(ordre):
    n      = len(df[df["Tranche_age"] == age]["Total"].dropna())
    y_max  = df[df["Tranche_age"] == age]["Total"].max()
    ax2.text(i, y_max + 1.5, f"N={n}", ha="center", fontsize=8,
             color=palette[i], fontweight="bold")
 
ax2.set_xticks(range(len(ordre)))
ax2.set_xticklabels(ordre, fontsize=10)
ax2.set_ylabel("Score Total", fontsize=10)
ax2.set_title("Distribution par âge (toutes conditions)", fontsize=11)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)
 
plt.tight_layout()
plt.savefig("../figures/scores_par_age.png", dpi=150, bbox_inches="tight")
plt.close()