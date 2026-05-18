import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# Chargement — on garde uniquement Enfant et Jeune enfant
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df_enf = df[df["Tranche_age"].isin(["Enfant", "Jeune enfant"])].copy()

COULEURS = {"C0": "#438FD2", "C1": "#DE4503"}
AGES     = ["Enfant", "Jeune enfant"]


# Effectifs par cellule
print("Effectifs par cellule (Condition × Tranche_age)")
print(df_enf.groupby(["Tranche_age", "Condition"])["Total"].count().unstack(fill_value=0))
print()

# Moyennes et écarts-types par cellule
print("Moyennes (SD) par cellule")
tab = df_enf.groupby(["Tranche_age", "Condition"])["Total"].agg(["mean", "std", "count"])
tab.columns = ["Moy", "SD", "N"]
print(tab.round(2))
print()


# ANOVA à deux facteurs (Type III — adapté aux effectifs déséquilibrés)
#
# Modèle : Total ~ Condition + Tranche_age + Condition:Tranche_age
# Type III : chaque effet est testé en contrôlant les autres → correct
# pour des cellules de tailles inégales.

model  = ols("Total ~ C(Condition) * C(Tranche_age)", data=df_enf).fit()
anova3 = anova_lm(model, typ=3)

# Calcul du eta² partiel : SS_effet / (SS_effet + SS_résiduelle)
ss_res = anova3.loc["Residual", "sum_sq"]
anova3["eta2_partiel"] = anova3["sum_sq"] / (anova3["sum_sq"] + ss_res)

print("=" * 70)
print("ANOVA à deux facteurs — Type III (Enfant + Jeune enfant)")
print("=" * 70)
print(f"  {'Facteur':<35} {'F':>7}  {'p':>8}  {'η²p':>7}")
print("  " + "-" * 62)
labels = {
    "Intercept":                       "Intercept",
    "C(Condition)":                    "Condition (C0 vs C1)",
    "C(Tranche_age)":                  "Tranche d'âge (Enfant vs Jeune enfant)",
    "C(Condition):C(Tranche_age)":     "Interaction Condition × Tranche_age",
}
for key, label in labels.items():
    if key == "Intercept":
        continue
    row = anova3.loc[key]
    sig = "***" if row["PR(>F)"] < 0.001 else "**" if row["PR(>F)"] < 0.01 \
          else "*" if row["PR(>F)"] < 0.05 else "ns"
    print(f"  {label:<35} {row['F']:>7.3f}  {row['PR(>F)']:>8.4f}  "
          f"{row['eta2_partiel']:>7.3f}  {sig}")
print()

print("Interprétation :")
p_cond = anova3.loc["C(Condition)", "PR(>F)"]
p_age  = anova3.loc["C(Tranche_age)", "PR(>F)"]
p_int  = anova3.loc["C(Condition):C(Tranche_age)", "PR(>F)"]

if p_cond < 0.05:
    print("  → Effet significatif de la CONDITION : C1 > C0 indépendamment de l'âge.")
else:
    print("  → Pas d'effet significatif de la condition après contrôle de l'âge.")

if p_age < 0.05:
    print("  → Effet significatif de l'ÂGE : les tranches ne scorent pas pareil.")
else:
    print("  → Pas d'effet significatif de l'âge une fois la condition contrôlée.")

if p_int < 0.05:
    print("  → Interaction significative : l'effet de C1 DIFFÈRE selon l'âge.")
else:
    print("  → Pas d'interaction : l'effet de C1 est SIMILAIRE quel que soit l'âge.")
print()


# Graphique 3 panels : effet Condition | effet Âge | Interaction
#   Panel 1 & 2 : boxplots des effets principaux
#   Panel 3 : interaction plot (lignes parallèles = pas d'interaction)

moyennes = df_enf.groupby(["Tranche_age", "Condition"])["Total"].mean().unstack()
erreurs  = df_enf.groupby(["Tranche_age", "Condition"])["Total"].sem().unstack()
ns       = df_enf.groupby(["Tranche_age", "Condition"])["Total"].count().unstack()

COULEURS_AGE = {"Enfant": "#2ECC71", "Jeune enfant": "#9B59B6"}

sig_cond = "***" if p_cond < 0.001 else "**" if p_cond < 0.01 else "*" if p_cond < 0.05 else "ns"
sig_age  = "***" if p_age  < 0.001 else "**" if p_age  < 0.01 else "*" if p_age  < 0.05 else "ns"
sig_int  = "***" if p_int  < 0.001 else "**" if p_int  < 0.01 else "*" if p_int  < 0.05 else "ns"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("ANOVA 2×2 — Effets principaux et interaction", fontsize=12, fontweight="bold")


def boxplot_simple(ax, groupes, couleurs, xlabel_list, titre, F, p, sig, eta2):
    all_vals = [v for g in groupes for v in g.tolist()]

    for i, (col, vals) in enumerate(zip(couleurs, groupes)):
        bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5),
                        whiskerprops=dict(color=col, linewidth=1.5),
                        capprops=dict(color=col, linewidth=1.5),
                        flierprops=dict(marker="o", color=col, alpha=0.5))
        bp["boxes"][0].set_facecolor(col)
        bp["boxes"][0].set_alpha(0.6)
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter([i] + jitter, vals, color=col, s=45, alpha=0.75, zorder=5)

    y_bar  = max(all_vals) + 2
    y_text = y_bar + 1.0
    ax.plot([0, 1], [y_bar, y_bar], color="#444", linewidth=1.2)
    ax.text(0.5, y_text,
            f"F={F:.2f}  p={p:.4f} {sig}\nη²p={eta2:.3f}",
            ha="center", va="bottom", fontsize=8.5, linespacing=1.6)
    ax.set_ylim(top=y_text + 5)
    ax.set_xticks(range(len(xlabel_list)))
    ax.set_xticklabels([f"{lbl}\n(N={len(g)})" for lbl, g in zip(xlabel_list, groupes)],
                       fontsize=10)
    ax.set_title(titre, fontsize=11)
    ax.set_ylabel("Score TCT-DP", fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# Panel 1 — Effet principal Condition (C0 vs C1, tous âges confondus)
boxplot_simple(
    axes[0],
    groupes=[df_enf[df_enf["Condition"] == c]["Total"].dropna() for c in ["C0", "C1"]],
    couleurs=[COULEURS["C0"], COULEURS["C1"]],
    xlabel_list=["C0", "C1"],
    titre="Effet Condition",
    F=anova3.loc["C(Condition)", "F"],
    p=p_cond, sig=sig_cond,
    eta2=anova3.loc["C(Condition)", "eta2_partiel"]
)

# Panel 2 — Effet principal Tranche_age (Enfant vs Jeune enfant, toutes conditions)
boxplot_simple(
    axes[1],
    groupes=[df_enf[df_enf["Tranche_age"] == a]["Total"].dropna() for a in AGES],
    couleurs=[COULEURS_AGE[a] for a in AGES],
    xlabel_list=["Enfant", "Jeune enfant"],
    titre="Effet Tranche d'âge",
    F=anova3.loc["C(Tranche_age)", "F"],
    p=p_age, sig=sig_age,
    eta2=anova3.loc["C(Tranche_age)", "eta2_partiel"]
)

# Panel 3 — Interaction plot (moyennes ± SEM par cellule)
ax = axes[2]
x_pos = [0, 1]
for cond in ["C0", "C1"]:
    moys = [moyennes.loc[age, cond] for age in AGES]
    errs = [erreurs.loc[age, cond]  for age in AGES]
    ax.plot(x_pos, moys, marker="o", color=COULEURS[cond], linewidth=2,
            markersize=8, label=cond)
    ax.errorbar(x_pos, moys, yerr=errs, fmt="none",
                color=COULEURS[cond], capsize=5, linewidth=1.5)
    for xi, age in zip(x_pos, AGES):
        ax.text(xi + 0.04, moyennes.loc[age, cond] + 0.4,
                f"N={ns.loc[age, cond]}", fontsize=8.5, color=COULEURS[cond])

ax.set_xticks(x_pos)
ax.set_xticklabels(["Enfant", "Jeune enfant"], fontsize=10)
ax.set_ylabel("Score TCT-DP moyen (± SEM)", fontsize=10)
ax.set_title("Interaction Condition × Tranche d'âge", fontsize=11)
ax.legend(title="Condition", fontsize=9)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.5, -0.12,
        f"F={anova3.loc['C(Condition):C(Tranche_age)','F']:.2f}  "
        f"p={p_int:.4f} {sig_int}  η²p={anova3.loc['C(Condition):C(Tranche_age)','eta2_partiel']:.3f}",
        ha="center", fontsize=8.5, color="#333", transform=ax.transAxes)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("../figures/anova2.png", dpi=150, bbox_inches="tight")
plt.close()
