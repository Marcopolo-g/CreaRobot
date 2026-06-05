import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Chargement — Adultes exclus
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df = df[df["Tranche_age"] != "Adulte"]

CRITERES = ["Cn", "Cm", "Ne", "Cl", "Cth", "Bfd", "Bfi", "Pe", "Hu", "Uc_b", "Uc_c", "Uc_d"]
MAXSCORES = {"Cn": 6, "Cm": 6, "Ne": 6, "Cl": 6, "Cth": 6,
             "Bfd": 3, "Bfi": 3, "Pe": 6, "Hu": 3,
             "Uc_b": 2, "Uc_c": 2, "Uc_d": 2}
couleurs = {"C0": "#438FD2", "C1": "#DE4503"}

# Normalisation : chaque critère divisé par son score max → valeur entre 0 et 1
CRITERES_N = [c + "_n" for c in CRITERES]
for c in CRITERES:
    df[c + "_n"] = df[c] / MAXSCORES[c]
df["Total_n"] = df[CRITERES_N].sum(axis=1)

# Suppression outliers (même logique que t_test.py, sur Total_n par groupe)
def retirer_outliers(serie, seuil=2.5):
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z <= seuil]

idx_keep = []
for cond in ["C0", "C1"]:
    sub = df[df["Condition"] == cond]["Total_n"]
    idx_keep.extend(retirer_outliers(sub).index.tolist())
df = df.loc[idx_keep]


# Statistiques par critère (normalisés)
RENAME = {cn: c for c, cn in zip(CRITERES, CRITERES_N)}

print("Moyennes normalisées par critère (global)")
moyennes_globales = df[CRITERES_N].mean().round(3).rename(RENAME).sort_values(ascending=False)
print(moyennes_globales)
print()

print("Moyennes normalisées par critère selon la condition")
tab = df.groupby("Condition")[CRITERES_N].mean().round(3).T.rename(index=RENAME)
print(tab)
print()

C0_df = df[df["Condition"] == "C0"]
C1_df = df[df["Condition"] == "C1"]

print("Différences de moyennes normalisées C1 − C0 par critère")
diff_moyennes = (C1_df[CRITERES_N].mean() - C0_df[CRITERES_N].mean()).round(3).rename(RENAME).sort_values(ascending=False)
print(diff_moyennes)
print()

# Cohen's d par critère (sur valeurs normalisées)
print("Cohen's d par critère — valeurs normalisées (C1 vs C0)")
def cohens_d(a, b):
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    return (b.mean() - a.mean()) / pooled if pooled != 0 else 0

ds = {}
for c, cn in zip(CRITERES, CRITERES_N):
    d = cohens_d(C0_df[cn].dropna(), C1_df[cn].dropna())
    ds[c] = round(d, 3)
ds_series = pd.Series(ds).sort_values(ascending=False)
print(ds_series)


# Graphique (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(22, 5))
fig.suptitle("Analyse des critères de scoring (normalisés 0–1)", fontsize=13, fontweight="bold")


# Panel 1 : score moyen normalisé par critère
ax = axes[0]
moy_triees      = moyennes_globales.sort_values()
couleurs_barres = ["#B0C4DE" if v < 0.33 else "#438FD2" for v in moy_triees]
ax.barh(moy_triees.index, moy_triees.values, color=couleurs_barres, alpha=0.85)
for i, v in enumerate(moy_triees.values):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlabel("Score moyen normalisé (0–1)", fontsize=10)
ax.set_title("Score moyen normalisé\npar critère", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(right=ax.get_xlim()[1] + 0.08)


# Panel 2 : moyennes normalisées C0 vs C1 par critère
ax2 = axes[1]
x     = np.arange(len(CRITERES))
width = 0.35

for i, cond in enumerate(["C0", "C1"]):
    cn_cols  = CRITERES_N
    moyennes = df[df["Condition"] == cond][cn_cols].mean()
    offset   = (i - 0.5) * width
    ax2.bar(x + offset, moyennes, width, label=cond,
            color=couleurs[cond], alpha=0.75)

ax2.set_xticks(x)
ax2.set_xticklabels(CRITERES, fontsize=9, rotation=45, ha="right")
ax2.set_ylabel("Score normalisé moyen", fontsize=10)
ax2.set_title("Moyenne normalisée par critère\nselon la condition", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)


# Panel 3 : Cohen's d par critère
ax4 = axes[2]
ds_triees  = ds_series.sort_values()
couleurs_d = ["#DE4503" if v >= 0 else "#438FD2" for v in ds_triees]
ax4.barh(ds_triees.index, ds_triees.values, color=couleurs_d, alpha=0.8)
ax4.axvline(0,    color="#555", linewidth=1)
ax4.axvline(0.5,  color="#aaa", linewidth=0.8, linestyle="--")
ax4.axvline(-0.5, color="#aaa", linewidth=0.8, linestyle="--")
ax4.axvline(0.8,  color="#aaa", linewidth=0.8, linestyle=":")
ax4.axvline(-0.8, color="#aaa", linewidth=0.8, linestyle=":")
ax4.set_xlabel("Cohen's d  [C1 − C0]", fontsize=10)
ax4.set_title("Taille d'effet par critère normalisé", fontsize=11)
ax4.spines[["top", "right"]].set_visible(False)
ax4.grid(axis="x", alpha=0.3)
ax4.text(0.52, -0.7, "petit", fontsize=7, color="#999")
ax4.text(0.82, -0.7, "grand", fontsize=7, color="#999")


plt.tight_layout()
plt.savefig("../figures/criteres_scoring.png", dpi=150, bbox_inches="tight")
plt.close()
