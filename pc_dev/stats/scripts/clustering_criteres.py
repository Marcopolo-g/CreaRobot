import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


CRITERES  = ["Cn", "Cm", "Ne", "Cl", "Cth", "Bfd", "Bfi", "Pe", "Hu", "Uc_b", "Uc_c", "Uc_d"]
MAXSCORES = {"Cn": 6, "Cm": 6, "Ne": 6, "Cl": 6, "Cth": 6,
             "Bfd": 3, "Bfi": 3, "Pe": 6, "Hu": 3,
             "Uc_b": 2, "Uc_c": 2, "Uc_d": 2}
COULEURS  = {"C0": "#438FD2", "C1": "#DE4503"}

# Chargement — Adultes exclus
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df = df[df["Tranche_age"] != "Adulte"].copy()

# Normalisation
CRITERES_N = [c + "_n" for c in CRITERES]
for c in CRITERES:
    df[c + "_n"] = df[c] / MAXSCORES[c]
df["Total_n"] = df[CRITERES_N].sum(axis=1)

# Standardisation pour le clustering
X = StandardScaler().fit_transform(df[CRITERES_N].values)


# Choix de k par silhouette
print("Silhouette scores par k :")
silhouettes = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    silhouettes[k] = silhouette_score(X, km.fit_predict(X))
    print(f"  k={k} : {silhouettes[k]:.3f}")
print()

# On utilise k=3 : profils bas / moyen / haut — plus interprétable
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X)

# Renommer les clusters par score croissant (A = plus bas → D = plus haut)
ordre = df.groupby("Cluster")["Total_n"].mean().sort_values().index.tolist()
rename = {old: new for new, old in enumerate(ordre)}
df["Cluster"] = df["Cluster"].map(rename)
NOMS_CLUSTERS    = {0: "Profil A", 1: "Profil B", 2: "Profil C", 3: "Profil D"}
COULEURS_CLUSTER = {0: "#E74C3C", 1: "#8E44AD", 2: "#1ABC9C", 3: "#27AE60"}

# Profil moyen normalisé par cluster
profils = df.groupby("Cluster")[CRITERES_N].mean()
profils.columns = CRITERES

print("Profil moyen normalisé (0–1) par cluster :")
print(profils.round(3))
print()

# Score total moyen par cluster
stats_cluster = df.groupby("Cluster")["Total_n"].agg(["mean", "std", "count"])
stats_cluster.index = [NOMS_CLUSTERS[i] for i in stats_cluster.index]
print("Score total normalisé par cluster (/12) :")
print(stats_cluster.round(3))
print()

# Composition C0/C1 par cluster
comp = df.groupby(["Cluster", "Condition"]).size().unstack(fill_value=0)
comp.index = [NOMS_CLUSTERS[i] for i in comp.index]
comp["Total"] = comp.sum(axis=1)
comp["% C1"]  = (comp.get("C1", 0) / comp["Total"] * 100).round(1)
print("Composition C0/C1 par cluster :")
print(comp)
print()

# Test chi²
contingency = df.groupby(["Cluster", "Condition"]).size().unstack(fill_value=0)
chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
print(f"Test χ² C0/C1 entre clusters : χ²={chi2:.3f}  ddl={dof}  p={p_chi2:.4f}")
print()


# Projection PCA pour visualisation
pca_2d  = PCA(n_components=2)
scores_2d = pca_2d.fit_transform(X)
var_exp = pca_2d.explained_variance_ratio_ * 100

# FIGURE
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Clustering sur les critères TCT-DP (k=4)", fontsize=13, fontweight="bold")


# Panel 1 : Profil moyen de chaque cluster sur les critères
ax = axes[0, 0]
x   = np.arange(len(CRITERES))
w   = 0.19
for i in range(K):
    vals = profils.loc[i].values
    ax.bar(x + (i - 1.5)*w, vals, w, label=NOMS_CLUSTERS[i],
           color=COULEURS_CLUSTER[i], alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(CRITERES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Score moyen normalisé (0–1)", fontsize=10)
ax.set_title("Profil créatif moyen\npar cluster", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)


# Panel 2 : Score total par cluster (boxplot)
ax2 = axes[0, 1]
for i in range(K):
    vals   = df[df["Cluster"] == i]["Total_n"]
    jitter = np.random.normal(0, 0.06, size=len(vals))
    bp = ax2.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2.5),
                     whiskerprops=dict(color=COULEURS_CLUSTER[i], linewidth=1.5),
                     capprops=dict(color=COULEURS_CLUSTER[i], linewidth=1.5),
                     flierprops=dict(marker="o", color=COULEURS_CLUSTER[i], alpha=0.5))
    bp["boxes"][0].set_facecolor(COULEURS_CLUSTER[i])
    bp["boxes"][0].set_alpha(0.7)
    ax2.scatter([i] + jitter, vals, color=COULEURS_CLUSTER[i], s=40, alpha=0.7, zorder=5)
    n = len(vals)
    ax2.text(i, -0.13, f"N={n}", ha="center", fontsize=9, color=COULEURS_CLUSTER[i],
             transform=ax2.get_xaxis_transform(), va="top", clip_on=False)

ax2.set_xticks(range(K))
ax2.set_xticklabels([NOMS_CLUSTERS[i] for i in range(K)], fontsize=10)
ax2.set_ylabel("Score total normalisé (/12)", fontsize=10)
ax2.set_title("Distribution du score total\npar cluster", fontsize=11)
ax2.grid(alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)


# Panel 3 : % C0 / C1 par cluster (barres empilées)
ax3 = axes[1, 0]
n_c0 = [comp.iloc[i].get("C0", 0) for i in range(K)]
n_c1 = [comp.iloc[i].get("C1", 0) for i in range(K)]
x3   = np.arange(K)

bars_c0 = ax3.bar(x3, n_c0, label="C0", color=COULEURS["C0"], alpha=0.85)
bars_c1 = ax3.bar(x3, n_c1, bottom=n_c0, label="C1", color=COULEURS["C1"], alpha=0.85)

# Pourcentage C1 affiché sur chaque barre
for i in range(K):
    total = n_c0[i] + n_c1[i]
    pct   = n_c1[i] / total * 100 if total > 0 else 0
    ax3.text(i, total + 0.3, f"{pct:.0f}% C1", ha="center", fontsize=9, fontweight="bold")

ax3.set_xticks(x3)
ax3.set_xticklabels([NOMS_CLUSTERS[i] for i in range(K)], fontsize=10)
ax3.set_ylabel("Nombre de participants", fontsize=10)
ax3.set_title(f"Répartition C0/C1 par cluster\n(χ²={chi2:.2f}, p={p_chi2:.4f})", fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)


# Panel 4 : Scatter — couleur = cluster, forme = condition (C0 rond / C1 triangle)
ax4 = axes[1, 1]
MARKERS_COND = {"C0": "o", "C1": "^"}

for i in range(K):
    for cond, marker in MARKERS_COND.items():
        mask = (df["Cluster"].values == i) & (df["Condition"].values == cond)
        ax4.scatter(scores_2d[mask, 0], scores_2d[mask, 1],
                    color=COULEURS_CLUSTER[i], marker=marker,
                    s=120, alpha=0.9, edgecolors="white", linewidths=1.2, zorder=5)

# Légende cluster (couleurs)
for i in range(K):
    ax4.scatter([], [], color=COULEURS_CLUSTER[i], marker="o", s=80,
                label=NOMS_CLUSTERS[i])
# Légende condition (formes)
ax4.scatter([], [], marker="o", color="#555", s=80, label="C0")
ax4.scatter([], [], marker="^", color="#555", s=80, label="C1")

ax4.axhline(0, color="#ccc", linewidth=0.7)
ax4.axvline(0, color="#ccc", linewidth=0.7)
ax4.set_xlabel(f"PC1 ({var_exp[0]:.1f}% de variance)", fontsize=10)
ax4.set_ylabel(f"PC2 ({var_exp[1]:.1f}% de variance)", fontsize=10)
ax4.set_title("Participants dans l'espace PCA", fontsize=10)
ax4.legend(fontsize=8, loc="upper left", framealpha=0.7)
ax4.spines[["top", "right"]].set_visible(False)
ax4.grid(alpha=0.2)


plt.tight_layout()
plt.savefig("../figures/clustering_criteres.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure sauvegardée : figures/clustering_criteres.png")
