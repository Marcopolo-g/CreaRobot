import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Chargement
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])

COULEURS = {"C0": "#438FD2", "C1": "#DE4503"}


# ──────────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ──────────────────────────────────────────────────────────────────────

def retirer_outliers(serie, seuil=2.5):
    """Retire les valeurs dont le z-score dépasse le seuil (même logique que t_test.py)."""
    if serie.std() == 0 or len(serie) < 3:
        return serie
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z <= seuil]


def welch_ttest(C0_raw, C1_raw, label=""):
    """
    T-test de Welch avec retrait des outliers, IC 95% et Cohen's d.
    Retourne un dict de résultats, ou None si effectifs insuffisants.
    """
    C0 = retirer_outliers(C0_raw)
    C1 = retirer_outliers(C1_raw)

    if len(C0) < 3 or len(C1) < 3:
        print(f"  {label} : effectifs insuffisants (N C0={len(C0)}, N C1={len(C1)}) → test non réalisé")
        return None

    t_stat, p = stats.ttest_ind(C0, C1, equal_var=False)
    diff      = C1.mean() - C0.mean()

    # Erreur standard de la différence des moyennes
    se = np.sqrt(C0.std()**2 / len(C0) + C1.std()**2 / len(C1))

    # Degrés de liberté de Welch (formule de Satterthwaite)
    df_w = (C0.std()**2/len(C0) + C1.std()**2/len(C1))**2 / \
           ((C0.std()**2/len(C0))**2/(len(C0)-1) + (C1.std()**2/len(C1))**2/(len(C1)-1))

    t_crit = stats.t.ppf(0.975, df=df_w)
    ic     = (diff - t_crit * se, diff + t_crit * se)

    # Cohen's d : taille d'effet normalisée
    pooled = np.sqrt((C0.std()**2 + C1.std()**2) / 2)
    d      = diff / pooled if pooled != 0 else 0

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"  {label}")
    print(f"    N C0={len(C0)}, N C1={len(C1)}  |  "
          f"Moy C0={C0.mean():.2f} (SD={C0.std():.2f}),  Moy C1={C1.mean():.2f} (SD={C1.std():.2f})")
    print(f"    t={t_stat:.3f}  p={p:.4f} {sig}  "
          f"Δ={diff:.2f}  IC95%=[{ic[0]:.2f} ; {ic[1]:.2f}]  Cohen's d={d:.3f}")

    return {"t": t_stat, "p": p, "diff": diff, "ic": ic, "d": d, "sig": sig,
            "C0": C0, "C1": C1}



# Effectifs par tranche d'âge
print("Effectifs par tranche d'âge et condition")
print(df.groupby(["Tranche_age", "Condition"])["Total"].count().unstack(fill_value=0))
print()
print("→ Adolescent (N=2 total) et Adulte (0 en C0, 3 en C1) : groupes trop")
print("  déséquilibrés pour des tests valides → exclus des analyses ci-dessous.")
print()



# Enfant + Jeune enfant (sans Ado/Adulte)

df_enf = df[df["Tranche_age"].isin(["Enfant", "Jeune enfant"])]

print("=" * 60)
print("Analyse 1 — Enfant + Jeune enfant (Ado et Adulte exclus)")
print("=" * 60)
res_enf = welch_ttest(
    df_enf[df_enf["Condition"] == "C0"]["Total"],
    df_enf[df_enf["Condition"] == "C1"]["Total"],
    label="Enfant + Jeune enfant"
)
print()

# Enfant seul

print("=" * 60)
print("Analyse 2 — Enfant seul")
print("=" * 60)
res_enfant = welch_ttest(
    df[(df["Tranche_age"] == "Enfant") & (df["Condition"] == "C0")]["Total"],
    df[(df["Tranche_age"] == "Enfant") & (df["Condition"] == "C1")]["Total"],
    label="Enfant seul"
)
print()



# Jeune enfant seul

print("=" * 60)
print("Analyse 3 — Jeune enfant seul")
print("=" * 60)
res_jeune = welch_ttest(
    df[(df["Tranche_age"] == "Jeune enfant") & (df["Condition"] == "C0")]["Total"],
    df[(df["Tranche_age"] == "Jeune enfant") & (df["Condition"] == "C1")]["Total"],
    label="Jeune enfant seul"
)
print()



# Tableau récapitulatif
print("=" * 60)
print("Récapitulatif")
print("=" * 60)
print(f"  {'Population':<28} {'p':>8}  {'Δ':>7}  {'IC 95%':^20}  {'d':>7}  {'sig':>5}")
print("  " + "-" * 77)
for label, res in [
    ("Enfant + Jeune enfant", res_enf),
    ("Enfant seul",           res_enfant),
    ("Jeune enfant seul",     res_jeune),
]:
    if res:
        ic_str = f"[{res['ic'][0]:.2f} ; {res['ic'][1]:.2f}]"
        print(f"  {label:<28} {res['p']:>8.4f}  {res['diff']:>7.2f}  {ic_str:^20}  {res['d']:>7.3f}  {res['sig']:>5}")
    else:
        print(f"  {label:<28} {'N/A':>8}  {'N/A':>7}  {'N/A':^20}  {'N/A':>7}  {'N/A':>5}")
print()



# Graphique (3 panels)

def boxplot_panel(ax, df_sub, titre, res=None):
    """Boxplot C0 vs C1 avec p-value et Cohen's d en annotation."""
    all_vals = []
    for i, cond in enumerate(["C0", "C1"]):
        vals = retirer_outliers(df_sub[df_sub["Condition"] == cond]["Total"].dropna())
        all_vals.extend(vals.tolist())

        bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                        medianprops=dict(color="white", linewidth=2.5),
                        whiskerprops=dict(color=COULEURS[cond], linewidth=1.5),
                        capprops=dict(color=COULEURS[cond], linewidth=1.5),
                        flierprops=dict(marker="o", color=COULEURS[cond], alpha=0.5))
        bp["boxes"][0].set_facecolor(COULEURS[cond])
        bp["boxes"][0].set_alpha(0.6)

        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter([i] + jitter, vals, color=COULEURS[cond], s=50, alpha=0.75, zorder=5)

        n = len(df_sub[df_sub["Condition"] == cond]["Total"].dropna())
        ax.text(i, -2, f"N={n}", ha="center", fontsize=9, color=COULEURS[cond])

    if res and all_vals:
        y_bar  = max(all_vals) + 2
        y_text = y_bar + 1.0
        ic_str = f"[{res['ic'][0]:.2f} ; {res['ic'][1]:.2f}]"
        ax.plot([0, 1], [y_bar, y_bar], color="#444", linewidth=1.2)
        ax.text(0.5, y_text,
                f"p={res['p']:.4f} {res['sig']}  d={res['d']:.2f}\n"
                f"Δ={res['diff']:.2f}  IC95% {ic_str}",
                ha="center", va="bottom", fontsize=8.5, linespacing=1.6)
        ax.set_ylim(top=y_text + 5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["C0", "C1"], fontsize=11)
    ax.set_ylabel("Score TCT-DP", fontsize=10)
    ax.set_title(titre, fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Analyse par tranche d'âge (Ado et Adulte exclus)",
             fontsize=13, fontweight="bold", x=0.5, ha="center")

boxplot_panel(axes[0], df_enf,
              "Enfant + Jeune enfant", res_enf)
boxplot_panel(axes[1], df[df["Tranche_age"] == "Enfant"],
              "Enfant seul", res_enfant)
boxplot_panel(axes[2], df[df["Tranche_age"] == "Jeune enfant"],
              "Jeune enfant seul", res_jeune)

plt.tight_layout()
plt.savefig("../figures/analyse_age.png", dpi=150, bbox_inches="tight")
plt.close()

