import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Chargement — on exclut Adolescent et Adulte (effectifs trop faibles)
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df = df[df["Tranche_age"].isin(["Enfant", "Jeune enfant"])]

CRITERES = ["Cn", "Cm", "Ne", "Cl", "Cth", "Bfd", "Bfi", "Pe", "Hu", "Uc_b", "Uc_c", "Uc_d"]
COULEURS  = {"C0": "#438FD2", "C1": "#DE4503"}

C0_df = df[df["Condition"] == "C0"]
C1_df = df[df["Condition"] == "C1"]



# Fonctions utilitaires


def welch_complet(serie_C0, serie_C1, label=""):
    """T-test de Welch avec IC 95% et Cohen's d. Retourne un dict de résultats."""
    t, p   = stats.ttest_ind(serie_C0, serie_C1, equal_var=False)
    diff   = serie_C1.mean() - serie_C0.mean()
    se     = np.sqrt(serie_C0.std()**2/len(serie_C0) + serie_C1.std()**2/len(serie_C1))
    df_w   = (serie_C0.std()**2/len(serie_C0) + serie_C1.std()**2/len(serie_C1))**2 / \
             ((serie_C0.std()**2/len(serie_C0))**2/(len(serie_C0)-1) +
              (serie_C1.std()**2/len(serie_C1))**2/(len(serie_C1)-1))
    t_crit = stats.t.ppf(0.975, df=df_w)
    ic     = (diff - t_crit * se, diff + t_crit * se)
    pooled = np.sqrt((serie_C0.std()**2 + serie_C1.std()**2) / 2)
    d      = diff / pooled if pooled != 0 else 0
    sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"  {label}")
    print(f"    Moy C0={serie_C0.mean():.3f}  Moy C1={serie_C1.mean():.3f}  "
          f"Δ={diff:.3f}  t={t:.3f}  p={p:.4f} {sig}  "
          f"IC95%=[{ic[0]:.3f} ; {ic[1]:.3f}]  d={d:.3f}")

    return {"t": t, "p": p, "diff": diff, "ic": ic, "d": d, "sig": sig,
            "C0": serie_C0, "C1": serie_C1}



# Récupération des p-values et Cohen's d par critère
#    (recalculés ici pour construire les scores — la visualisation
#     complète est dans criteres_scoring.py)

resultats = {}
for c in CRITERES:
    a = C0_df[c].dropna()
    b = C1_df[c].dropna()

    if a.std() == 0 and b.std() == 0:
        resultats[c] = {"p": np.nan, "d": 0.0, "sig": "ns"}
        continue

    t, p   = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    d      = (b.mean() - a.mean()) / pooled if pooled != 0 else 0
    sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    resultats[c] = {"p": p, "d": d, "sig": sig}

res_df = pd.DataFrame(resultats).T

# Critères significatifs en faveur de C1
crit_sig = res_df[(res_df["p"] < 0.05) & (res_df["d"] > 0)].index.tolist()
print(f"Critères retenus (p < 0.05 et d > 0) : {crit_sig}")
print()

print(f"  {'Critère':<10}  {'p':>8}  {'sig':>5}")
print("  " + "-" * 28)
for c, row in res_df.sort_values("p").iterrows():
    print(f"  {c:<10}  {row['p']:>8.4f}  {row['sig']:>5}")
print()



# Construction des scores alternatifs
#
# Score restreint : somme des critères significatifs uniquement (poids égaux).
#   On isole les critères qui portent réellement le signal.
#
# Score pondéré   : somme de tous les critères multipliés par leur Cohen's d
#   (d ≤ 0 → poids=0 pour ne pas pénaliser C1). Les poids sont normalisés
#   pour que leur somme = nombre de critères avec d > 0.


# Score restreint
if crit_sig:
    df["Score_restreint"] = df[crit_sig].sum(axis=1)
else:
    print("Aucun critère significatif → score restreint non calculé")
    df["Score_restreint"] = np.nan

# Score pondéré
poids_bruts = res_df["d"].clip(lower=0)
somme_poids = poids_bruts.sum()
n_positifs  = (poids_bruts > 0).sum()
poids_norm  = poids_bruts * (n_positifs / somme_poids) if somme_poids > 0 else poids_bruts

print("Poids par critère (proportionnels à Cohen's d, d ≤ 0 → 0) :")
for c, w in poids_norm.sort_values(ascending=False).items():
    mark = " ← sig" if c in crit_sig else ""
    print(f"  {c:<8}  {w:.4f}{mark}")
print()

df["Score_pondere"] = df[CRITERES].mul(poids_norm, axis=1).sum(axis=1)


# 3. T-tests sur les deux nouveaux scores

print("=" * 65)
print("T-tests sur les scores alternatifs (C0 vs C1)")
print("=" * 65)

res_pond = welch_complet(
    df[df["Condition"] == "C0"]["Score_pondere"],
    df[df["Condition"] == "C1"]["Score_pondere"],
    label=f"Score pondéré (poids ∝ Cohen's d, {n_positifs} critères actifs)"
)

res_restr = None
if crit_sig:
    res_restr = welch_complet(
        df[df["Condition"] == "C0"]["Score_restreint"],
        df[df["Condition"] == "C1"]["Score_restreint"],
        label=f"Score restreint ({len(crit_sig)} critères sig. : {crit_sig})"
    )
print()


# Tableau récapitulatif
print("Récapitulatif")
print(f"  {'Score':<50} {'p':>8}  {'Δ':>7}  {'IC 95%':^22}  {'d':>7}  {'sig':>5}")
print("  " + "-" * 97)
for label, res in [
    (f"Pondéré ({n_positifs} critères, poids=d)",  res_pond),
    (f"Restreint ({len(crit_sig)} crit. sig.)",     res_restr),
]:
    if res:
        ic_str = f"[{res['ic'][0]:.3f} ; {res['ic'][1]:.3f}]"
        print(f"  {label:<50} {res['p']:>8.4f}  {res['diff']:>7.3f}  "
              f"{ic_str:^22}  {res['d']:>7.3f}  {res['sig']:>5}")
    else:
        print(f"  {label:<50} {'N/A':>8}  {'N/A':>7}  {'N/A':^22}  {'N/A':>7}  {'N/A':>5}")
print()



# Graphique (2 panels)

def mini_boxplot(ax, res, titre, ylabel="Score"):
    """Boxplot C0 vs C1 avec annotation p-value et Cohen's d."""
    if res is None:
        ax.text(0.5, 0.5, "Non disponible", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#888")
        ax.set_title(titre, fontsize=11)
        return

    for i, (cond, vals) in enumerate({"C0": res["C0"], "C1": res["C1"]}.items()):
        bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5),
                        whiskerprops=dict(color=COULEURS[cond], linewidth=1.5),
                        capprops=dict(color=COULEURS[cond], linewidth=1.5),
                        flierprops=dict(marker="o", color=COULEURS[cond], alpha=0.5))
        bp["boxes"][0].set_facecolor(COULEURS[cond])
        bp["boxes"][0].set_alpha(0.6)
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter([i] + jitter, vals, color=COULEURS[cond], s=45, alpha=0.75, zorder=5)

    y_bar  = max(res["C0"].max(), res["C1"].max()) + res["C0"].std() * 0.3
    y_text = y_bar + res["C0"].std() * 0.3
    ic_str = f"[{res['ic'][0]:.2f} ; {res['ic'][1]:.2f}]"
    ax.plot([0, 1], [y_bar, y_bar], color="#444", linewidth=1.2)
    ax.text(0.5, y_text,
            f"p={res['p']:.4f} {res['sig']}  d={res['d']:.3f}\n"
            f"Δ={res['diff']:.2f}  IC95% {ic_str}",
            ha="center", va="bottom", fontsize=8.5, linespacing=1.6)
    ax.set_ylim(top=y_text + res["C0"].std())

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["C0", "C1"], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(titre, fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Scores alternatifs : pondéré vs restreint aux critères significatifs",
             fontsize=13, fontweight="bold")

mini_boxplot(axes[0], res_pond,
             f"Score pondéré\n(poids ∝ Cohen's d, {n_positifs} critères)",
             "Score pondéré")
mini_boxplot(axes[1], res_restr,
             f"Score restreint\n({len(crit_sig)} critères significatifs)",
             "Score restreint")

plt.tight_layout()
plt.savefig("../figures/criteres_ponderes.png", dpi=150, bbox_inches="tight")
plt.close()


# Graphique Cohen's d par critère (barres horizontales, trié par d décroissant)

ds_sorted = res_df["d"].sort_values(ascending=True)  # ascending → plus grand en haut du graphe
couleurs_bars = ["#DE4503" if c in crit_sig else "#9E9E9E" for c in ds_sorted.index]

fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(range(len(ds_sorted)), ds_sorted.values, color=couleurs_bars, alpha=0.85)

for ref, label in [(0.2, "petit"), (0.5, "moyen"), (0.8, "large")]:
    ax.axvline(ref, color="#bbb", linewidth=1, linestyle="--")
    ax.text(ref + 0.02, len(ds_sorted) - 0.5, label, fontsize=7.5, color="#999")

ax.axvline(0, color="#444", linewidth=1)

for i, (c, d) in enumerate(ds_sorted.items()):
    offset = 0.03 if d >= 0 else -0.03
    ha = "left" if d >= 0 else "right"
    col = "#DE4503" if c in crit_sig else "#555"
    ax.text(d + offset, i, f"{d:.3f}", va="center", ha=ha, fontsize=8.5, color=col)

ax.set_yticks(range(len(ds_sorted)))
ax.set_yticklabels(ds_sorted.index, fontsize=10)
ax.set_xlabel("Cohen's d", fontsize=10)
ax.set_title("Cohen's d par critère — Enfant + Jeune enfant\n"
             "Orange = significatif (p < 0.05 et d > 0)", fontsize=11)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("../figures/criteres_cohens_d.png", dpi=150, bbox_inches="tight")
plt.close()
