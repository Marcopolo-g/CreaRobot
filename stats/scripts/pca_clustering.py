import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CRITERES  = ["Cn", "Cm", "Ne", "Cl", "Cth", "Bfd", "Bfi", "Pe", "Hu", "Uc_b", "Uc_c", "Uc_d"]
MAXSCORES = {"Cn": 6, "Cm": 6, "Ne": 6, "Cl": 6, "Cth": 6,
             "Bfd": 3, "Bfi": 3, "Pe": 6, "Hu": 3,
             "Uc_b": 2, "Uc_c": 2, "Uc_d": 2}
COULEURS  = {"C0": "#438FD2", "C1": "#DE4503"}

# Chargement — Adultes exclus
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df = df[df["Tranche_age"] != "Adulte"].copy()

# Normalisation + score total
CRITERES_N = [c + "_n" for c in CRITERES]
for c in CRITERES:
    df[c + "_n"] = df[c] / MAXSCORES[c]
df["Total_n"] = df[CRITERES_N].sum(axis=1)

# Standardisation pour la PCA
X = StandardScaler().fit_transform(df[CRITERES_N].values)
labels_cond = df["Condition"].values


# ACP
pca        = PCA()
scores_pca = pca.fit_transform(X)
loadings   = pd.DataFrame(pca.components_.T, index=CRITERES,
                          columns=[f"PC{i+1}" for i in range(len(CRITERES))])
variance_exp = pca.explained_variance_ratio_ * 100
eigenvalues  = pca.explained_variance_

N_COMP = 3  # coude du scree plot après PC3 — 56% de variance cumulée

print("Variance expliquée par composante :")
print(f"  {'PC':<5} {'Eigenvalue':>11}  {'Var %':>7}  {'Cumulé':>8}  {'Retenu':>8}")
print("  " + "-" * 48)
for i in range(len(CRITERES)):
    keep = "✓" if i < N_COMP else ""
    print(f"  PC{i+1:<3} {eigenvalues[i]:>11.3f}  {variance_exp[i]:>7.1f}%  "
          f"{variance_exp[:i+1].sum():>7.1f}%  {keep:>8}")
print()
print(f"→ {N_COMP} composantes retenues (coude scree plot, {variance_exp[:N_COMP].sum():.1f}% cumulé)")
print()

print("Loadings PC1–PC3 :")
print(loadings[["PC1", "PC2", "PC3"]].round(3))
print()

contrib = loadings[["PC1", "PC2", "PC3"]].apply(lambda col: col**2).round(3)
print("Contributions normalisées (cos²) par critère — somme = 1 par PC :")
print(contrib.to_string())
print()


# FIGURE — 3 panels
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Analyse en Composantes Principales (ACP) — critères TCT-DP normalisés",
             fontsize=13, fontweight="bold")


# Panel 1 : Scree plot
ax = axes[0]
colors = ["#DE4503" if i < N_COMP else "#B0C4DE" for i in range(8)]
ax.bar(range(1, 9), variance_exp[:8], color=colors, alpha=0.85)
ax.plot(range(1, 9), variance_exp[:8], "o-", color="#333", linewidth=1.5, markersize=6)
ax.axvline(N_COMP + 0.5, color="#555", linewidth=1.2, linestyle="--",
           label=f"Coude après PC{N_COMP} ({variance_exp[:N_COMP].sum():.0f}% cumulé)")
for i, v in enumerate(variance_exp[:8]):
    ax.text(i+1, v+0.3, f"{v:.1f}%", ha="center", fontsize=8)
ax.set_xlabel("Composante principale", fontsize=10)
ax.set_ylabel("Variance expliquée (%)", fontsize=10)
ax.set_title("Scree plot", fontsize=10)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)


# Panel 2 : Participants PC1 vs PC2, colorés par condition
ax2 = axes[1]
for cond in ["C0", "C1"]:
    mask = labels_cond == cond
    ax2.scatter(scores_pca[mask, 0], scores_pca[mask, 1],
                color=COULEURS[cond], alpha=0.75, s=55, label=cond, zorder=5)
ax2.axhline(0, color="#ccc", linewidth=0.7)
ax2.axvline(0, color="#ccc", linewidth=0.7)
ax2.set_xlabel(f"PC1 ({variance_exp[0]:.1f}% de variance)", fontsize=10)
ax2.set_ylabel(f"PC2 ({variance_exp[1]:.1f}% de variance)", fontsize=10)
ax2.set_title("Participants dans l'espace PCA (C0 vs C1)", fontsize=10)
ax2.legend(fontsize=9)
ax2.spines[["top", "right"]].set_visible(False)
ax2.grid(alpha=0.2)


# Panel 3 : Heatmap contributions normalisées (cos²)
ax3 = axes[2]
contrib_mat = loadings.iloc[:, :N_COMP].values ** 2
im = ax3.imshow(contrib_mat, cmap="Blues", vmin=0, vmax=contrib_mat.max(), aspect="auto")
ax3.set_xticks(range(N_COMP))
ax3.set_xticklabels([f"PC{i+1}\n({variance_exp[i]:.1f}%)" for i in range(N_COMP)], fontsize=10)
ax3.set_yticks(range(len(CRITERES)))
ax3.set_yticklabels(CRITERES, fontsize=9)
ax3.set_title("Contribution de chaque critère aux composantes\n(Contributions normalisées : loadings²)", fontsize=10)
plt.colorbar(im, ax=ax3, shrink=0.7, label="cos² (contribution)")
for i in range(len(CRITERES)):
    for j in range(N_COMP):
        val = contrib_mat[i, j]
        ax3.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=7.5, color="white" if val > 0.15 else "black")


plt.tight_layout()
plt.savefig("../figures/pca_clustering.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure sauvegardée : figures/pca_clustering.png")
