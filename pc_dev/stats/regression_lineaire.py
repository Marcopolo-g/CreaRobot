import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Chargement
df = pd.read_csv("scores_tctdp.csv").dropna(subset=["Total"])
df["Condition_num"] = df["Condition"].map({"C0": 0, "C1": 1})

y = df["Total"].values
x = df["Condition_num"].values

# Modèle Contraint (MC) — prédit la moyenne pour tout le monde
b0_mc = np.mean(y)
SCR_mc = np.sum((y - b0_mc)**2)

# Modèle Augmenté (MA) — prédit en fonction de la condition
b1, b0_ma, r, p, se = stats.linregress(x, y)
SCR_ma = np.sum((y - (b0_ma + b1 * x))**2)
R2 = 1 - SCR_ma / SCR_mc

# Résultats
print(f"MC  : Score_i = {b0_mc:.2f}")
print(f"MA  : Score_i = {b0_ma:.2f} + {b1:.2f} × Condition_i")
print()
print(f"b0 = {b0_ma:.2f}  →  score prédit pour C0")
print(f"b1 = {b1:.2f}   →  augmentation du score en passant de C0 à C1")
print()
print(f"p-value = {p:.6f}")
print(f"R²      = {R2:.3f}  ({R2*100:.1f}% de variance expliquée par la condition)")

# Graphique
fig, ax = plt.subplots(figsize=(7, 5))
couleurs = {0: "#438FD2", 1: "#DE4503"}
for xi, yi in zip(x, y):
    ax.scatter(xi + np.random.normal(0, 0.02), yi,
               color=couleurs[xi], s=60, alpha=0.75, zorder=3)

ax.axhline(b0_mc, color="gray", linestyle="--", linewidth=1.5, label=f"MC : ŷ = {b0_mc:.2f}")
x_line = np.linspace(-0.2, 1.2, 100)
ax.plot(x_line, b0_ma + b1 * x_line, color="darkred", linewidth=2,
        label=f"MA : ŷ = {b0_ma:.2f} + {b1:.2f}×Condition  (p={p:.6f})")

ax.set_xticks([0, 1])
ax.set_xticklabels(["C0", "C1"], fontsize=12)
ax.set_ylabel("Score TCT-DP", fontsize=11)
ax.set_title("Régression linéaire", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("regression.png", dpi=150, bbox_inches="tight")
plt.close()