import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CRITERES  = ["Cn", "Cm", "Ne", "Cl", "Cth", "Bfd", "Bfi", "Pe", "Hu", "Uc_b", "Uc_c", "Uc_d"]
MAXSCORES = {"Cn": 6, "Cm": 6, "Ne": 6, "Cl": 6, "Cth": 6,
             "Bfd": 3, "Bfi": 3, "Pe": 6, "Hu": 3,
             "Uc_b": 2, "Uc_c": 2, "Uc_d": 2}
N_COMP = 3

# Chargement — Adultes exclus
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df = df[df["Tranche_age"] != "Adulte"].copy()

for c in CRITERES:
    df[c + "_n"] = df[c] / MAXSCORES[c]


def run_pca(criteres):
    cols = [c + "_n" for c in criteres]
    X = StandardScaler().fit_transform(df[cols].values)
    pca = PCA()
    pca.fit(X)
    variance_exp = pca.explained_variance_ratio_ * 100
    loadings = pd.DataFrame(pca.components_[:N_COMP].T, index=criteres,
                             columns=[f"PC{i+1}" for i in range(N_COMP)])
    # Orienter chaque composante de facon coherente (signe arbitraire sinon)
    for pc in loadings.columns:
        if loadings[pc].sum() < 0:
            loadings[pc] = -loadings[pc]
    return variance_exp[:N_COMP], loadings


scenarios = {
    "12 criteres (reference)": CRITERES,
    "Sans Cm": [c for c in CRITERES if c != "Cm"],
    "Sans Hu": [c for c in CRITERES if c != "Hu"],
    "Sans Cm + Hu": [c for c in CRITERES if c not in ("Cm", "Hu")],
}

variances_pc = {}
loadings_by_scenario = {}

print(f"{'Scenario':<25} {'Total 3 PC':>10}   Delta")
print("-" * 50)
ref_total = None
for name, criteres in scenarios.items():
    variance_pc, loadings = run_pca(criteres)
    variances_pc[name] = variance_pc
    loadings_by_scenario[name] = loadings
    total = variance_pc.sum()
    if ref_total is None:
        ref_total = total
        print(f"{name:<25} {total:>9.1f}%   —")
    else:
        print(f"{name:<25} {total:>9.1f}%   {total - ref_total:+.1f}%")

for name, loadings in loadings_by_scenario.items():
    print()
    print(f"Loadings PC1-PC3 — {name} :")
    print(loadings.round(3).to_string())

print()
print("Stabilite de PC1 — correlation des loadings avec la reference")
print("(criteres communs uniquement) :")
ref_pc1 = loadings_by_scenario["12 criteres (reference)"]["PC1"]
for name, loadings in loadings_by_scenario.items():
    if name == "12 criteres (reference)":
        continue
    common = loadings.index
    corr = np.corrcoef(ref_pc1[common], loadings["PC1"][common])[0, 1]
    print(f"  {name:<25} r = {corr:.3f}")


# FIGURE — heatmap des contributions (cos²) par scenario, cote a cote
fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 7))
fig.suptitle("Analyse de sensibilite PCA — contribution des criteres aux composantes (cos²)",
             fontsize=13, fontweight="bold")

for ax, (name, loadings) in zip(axes, loadings_by_scenario.items()):
    criteres = list(loadings.index)
    variance_pc = variances_pc[name]
    contrib_mat = loadings.values ** 2
    im = ax.imshow(contrib_mat, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(N_COMP))
    ax.set_xticklabels([f"PC{i+1}\n({variance_pc[i]:.1f}%)" for i in range(N_COMP)], fontsize=9)
    ax.set_yticks(range(len(criteres)))
    ax.set_yticklabels(criteres, fontsize=8)
    ax.set_title(name, fontsize=10)
    for i in range(len(criteres)):
        for j in range(N_COMP):
            val = contrib_mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.25 else "black")

fig.colorbar(im, ax=axes, shrink=0.6, label="cos² (contribution)")
plt.savefig("../figures/pca_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close()
print()
print("Figure sauvegardee : figures/pca_sensitivity.png")
