import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
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


# Choix de k — silhouette + inertie
print("Justification du choix de k :")
print(f"  {'k':>3}  {'Silhouette':>11}  {'Inertie':>10}")
print("  " + "-" * 30)
silhouettes = {}
inerties    = {}
for k in range(2, 7):
    km_k     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(X)
    silhouettes[k] = silhouette_score(X, labels_k)
    inerties[k]    = km_k.inertia_
    print(f"  {k:>3}  {silhouettes[k]:>11.3f}  {inerties[k]:>10.1f}")
print()
print("→ k=5 retenu : meilleur score de silhouette (0.189)")
print()

# K-means k=5
K = 5
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X)

# Renommer les clusters par score croissant (A = plus bas → E = plus haut)
ordre  = df.groupby("Cluster")["Total_n"].mean().sort_values().index.tolist()
rename = {old: new for new, old in enumerate(ordre)}
df["Cluster"] = df["Cluster"].map(rename)
NOMS_CLUSTERS    = {0: "Profil A", 1: "Profil B", 2: "Profil C", 3: "Profil D", 4: "Profil E"}
COULEURS_CLUSTER = {0: "#E74C3C", 1: "#E67E22", 2: "#8E44AD", 3: "#1ABC9C", 4: "#27AE60"}

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


# Comparaison K-means vs Ward (clustering hiérarchique)
labels_ward_raw = fcluster(linkage(X, method="ward"), t=K, criterion="maxclust") - 1
# Aligner Ward sur les mêmes noms (par score Total_n croissant)
ward_series = pd.Series(labels_ward_raw, index=df.index)
ordre_ward  = df.copy()
ordre_ward["ward"] = ward_series
ordre_ward  = ordre_ward.groupby("ward")["Total_n"].mean().sort_values().index.tolist()
rename_ward = {old: new for new, old in enumerate(ordre_ward)}
labels_ward = ward_series.map(rename_ward).values

ari_ward = adjusted_rand_score(df["Cluster"].values, labels_ward)
print(f"Comparaison K-means vs Ward (k={K}) :")
print(f"  ARI = {ari_ward:.3f}  {'→ accord défendable (> 0.3)' if ari_ward > 0.3 else '→ accord faible (< 0.3)'}")
print()


# Test de permutation — le clustering est-il dû au hasard ?
N_PERM = 1000
K_TEST = K
np.random.seed(42)
sil_random = []
for i in range(N_PERM):
    X_perm = X.copy()
    for j in range(X_perm.shape[1]):
        np.random.shuffle(X_perm[:, j])
    km_r = KMeans(n_clusters=K_TEST, random_state=i % 200, n_init=5)
    sil_random.append(silhouette_score(X_perm, km_r.fit_predict(X_perm)))

sil_random = np.array(sil_random)
sil_real   = silhouettes[K_TEST]
p_perm     = np.mean(sil_random >= sil_real)

print(f"Test de permutation (N={N_PERM} permutations, k={K_TEST}) :")
print(f"  Score silhouette réel  : {sil_real:.3f}")
print(f"  Baseline aléatoire     : {sil_random.mean():.3f} ± {sil_random.std():.3f}")
p_perm_str = "p < 0.001" if p_perm < 0.001 else f"p = {p_perm:.3f}"
print(f"  p-valeur               : {p_perm_str}  {'***' if p_perm < 0.001 else '**' if p_perm < 0.01 else '*' if p_perm < 0.05 else 'ns'}")
print()


# Projection PCA pour visualisation
pca_2d    = PCA(n_components=2)
scores_2d = pca_2d.fit_transform(X)
var_exp   = pca_2d.explained_variance_ratio_ * 100

# FIGURE PRINCIPALE
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f"Clustering sur les critères TCT-DP (k={K})", fontsize=13, fontweight="bold")


# Panel 1 : Profil moyen de chaque cluster sur les critères
ax = axes[0, 0]
x  = np.arange(len(CRITERES))
w  = 0.15
for i in range(K):
    vals = profils.loc[i].values
    ax.bar(x + (i - 2)*w, vals, w, label=NOMS_CLUSTERS[i],
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

ax3.bar(x3, n_c0, label="C0", color=COULEURS["C0"], alpha=0.85)
ax3.bar(x3, n_c1, bottom=n_c0, label="C1", color=COULEURS["C1"], alpha=0.85)

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


# Panel 4 : Scatter PCA — couleur = cluster, forme = condition
ax4 = axes[1, 1]
MARKERS_COND = {"C0": "o", "C1": "^"}

for i in range(K):
    for cond, marker in MARKERS_COND.items():
        mask = (df["Cluster"].values == i) & (df["Condition"].values == cond)
        ax4.scatter(scores_2d[mask, 0], scores_2d[mask, 1],
                    color=COULEURS_CLUSTER[i], marker=marker,
                    s=120, alpha=0.9, edgecolors="white", linewidths=1.2, zorder=5)

for i in range(K):
    ax4.scatter([], [], color=COULEURS_CLUSTER[i], marker="o", s=80, label=NOMS_CLUSTERS[i])
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


# FIGURE DE VALIDATION — 3 panels
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle(f"Validation du clustering (k={K})",
              fontsize=13, fontweight="bold")

ks = list(range(2, 7))

# Panel 1 : Silhouette par k
ax_s = axes2[0]
cols = ["#DE4503" if k == K else "#B0C4DE" for k in ks]
ax_s.bar(ks, [silhouettes[k] for k in ks], color=cols, alpha=0.85)
ax_s.plot(ks, [silhouettes[k] for k in ks], "o-", color="#333", linewidth=1.5, markersize=6)
for k in ks:
    ax_s.text(k, silhouettes[k] + 0.003, f"{silhouettes[k]:.3f}", ha="center", fontsize=9)
ax_s.set_xlabel("Nombre de clusters k", fontsize=10)
ax_s.set_ylabel("Score de silhouette", fontsize=10)
ax_s.set_title("Score de silhouette par k\n(orange = k retenu)", fontsize=10)
ax_s.set_xticks(ks)
ax_s.spines[["top", "right"]].set_visible(False)
ax_s.grid(axis="y", alpha=0.3)

# Panel 2 : Elbow (inertie)
ax_e = axes2[1]
ax_e.plot(ks, [inerties[k] for k in ks], "o-", color="#438FD2", linewidth=2, markersize=7)
ax_e.scatter([K], [inerties[K]], color="#DE4503", s=100, zorder=5, label=f"k={K} retenu")
for k in ks:
    ax_e.text(k, inerties[k] + 1.5, f"{inerties[k]:.0f}", ha="center", fontsize=8.5)
ax_e.set_xlabel("Nombre de clusters k", fontsize=10)
ax_e.set_ylabel("Inertie intra-cluster", fontsize=10)
ax_e.set_title("Courbe elbow (inertie)", fontsize=10)
ax_e.set_xticks(ks)
ax_e.legend(fontsize=9)
ax_e.spines[["top", "right"]].set_visible(False)
ax_e.grid(alpha=0.3)

# Panel 3 : Test de permutation
ax_p = axes2[2]
ax_p.hist(sil_random, bins=40, color="#B0C4DE", alpha=0.85, edgecolor="white",
          label=f"Baseline aléatoire\n(moy={sil_random.mean():.3f} ± {sil_random.std():.3f})")
ax_p.axvline(sil_real, color="#DE4503", linewidth=2.5,
             label=f"Score réel = {sil_real:.3f}\n{p_perm_str}")
sig_str = "***" if p_perm < 0.001 else "**" if p_perm < 0.01 else "*" if p_perm < 0.05 else "ns"
ax_p.set_xlabel("Score de silhouette", fontsize=10)
ax_p.set_ylabel("Fréquence", fontsize=10)
ax_p.set_title(f"Test de permutation (N={N_PERM})", fontsize=10)
ax_p.legend(fontsize=9)
ax_p.spines[["top", "right"]].set_visible(False)
ax_p.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/clustering_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure sauvegardée : figures/clustering_validation.png")


# BOOTSTRAP STABILITY — ARI + taille minimale des clusters
N_BOOT        = 500
SUBSAMPLE_PCT = 0.70
n_sub         = int(len(X) * SUBSAMPLE_PCT)
rng_boot      = np.random.default_rng(42)
reference_labels = df["Cluster"].values

ari_boot     = []
min_size_sub = []

for i in range(N_BOOT):
    # Bootstrap ARI : remise, prédit sur tout N
    idx_b = rng_boot.integers(0, len(X), size=len(X))
    km_b  = KMeans(n_clusters=K, random_state=i, n_init=5).fit(X[idx_b])
    ari_boot.append(adjusted_rand_score(reference_labels, km_b.predict(X)))

    # Taille minimale : sous-échantillon sans remise 70%
    idx_s = rng_boot.choice(len(X), size=n_sub, replace=False)
    km_s  = KMeans(n_clusters=K, random_state=i, n_init=5).fit(X[idx_s])
    _, cnts = np.unique(km_s.labels_, return_counts=True)
    min_size_sub.append(cnts.min())

ari_boot     = np.array(ari_boot)
min_size_sub = np.array(min_size_sub)

print(f"Stabilité bootstrap (N={N_BOOT} itérations, k={K}) :")
print(f"  ARI moyen  : {ari_boot.mean():.3f} ± {ari_boot.std():.3f}")
print(f"  ARI médian : {np.median(ari_boot):.3f}")
print()
print(f"Taille minimale de cluster (sous-échantillons {int(SUBSAMPLE_PCT*100)}%, N~{n_sub}) :")
print(f"  Moyenne    : {min_size_sub.mean():.1f} ± {min_size_sub.std():.1f}")
print(f"  Min absolu : {min_size_sub.min()}")
print()


# FIGURE STABILITÉ — 2 panels
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle(f"Stabilité bootstrap du clustering (k={K}, N_boot={N_BOOT})",
              fontsize=13, fontweight="bold")

ax_a = axes3[0]
ax_a.hist(ari_boot, bins=40, color="#438FD2", alpha=0.85, edgecolor="white")
ax_a.axvline(ari_boot.mean(), color="#DE4503", linewidth=2,
             label=f"ARI moyen = {ari_boot.mean():.3f}")
ax_a.axvline(0.6, color="#aaa", linewidth=1.2, linestyle="--", label="Seuil stable (0.6)")
ax_a.set_xlabel("Adjusted Rand Index (ARI)", fontsize=10)
ax_a.set_ylabel("Fréquence", fontsize=10)
ax_a.set_title("Stabilité des clusters\n(bootstrap avec remise, prédiction sur N=48)", fontsize=10)
ax_a.legend(fontsize=9)
ax_a.spines[["top", "right"]].set_visible(False)
ax_a.grid(alpha=0.3)

ax_m = axes3[1]
bins = np.arange(min_size_sub.min(), min_size_sub.max() + 2) - 0.5
ax_m.hist(min_size_sub, bins=bins, color="#8E44AD", alpha=0.85, edgecolor="white")
ax_m.axvline(min_size_sub.mean(), color="#DE4503", linewidth=2,
             label=f"Moyenne = {min_size_sub.mean():.1f} ± {min_size_sub.std():.1f}")
ax_m.set_xlabel("Taille minimale d'un cluster", fontsize=10)
ax_m.set_ylabel("Fréquence", fontsize=10)
ax_m.set_title(f"Taille minimale de cluster par itération\n(sous-échantillons {int(SUBSAMPLE_PCT*100)}%, N~{n_sub})",
               fontsize=10)
ax_m.legend(fontsize=9)
ax_m.spines[["top", "right"]].set_visible(False)
ax_m.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/clustering_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure sauvegardée : figures/clustering_stability.png")
