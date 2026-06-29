import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Chargement — Adultes exclus
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df = df[df["Tranche_age"] != "Adulte"]

C0 = df[df["Condition"] == "C0"]["Total"]
C1 = df[df["Condition"] == "C1"]["Total"]


def flag_outliers(serie, seuil=2.5):
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z > seuil]

def retirer_outliers(serie, seuil=2.5):
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z <= seuil]

outliers_C0 = flag_outliers(C0)
outliers_C1 = flag_outliers(C1)

print("Outliers détectés")
if len(outliers_C0) == 0 and len(outliers_C1) == 0:
    print("Aucun outlier (|z| > 2.5)")
else:
    if len(outliers_C0) > 0: print(f"C0 : {outliers_C0.to_dict()}")
    if len(outliers_C1) > 0: print(f"C1 : {outliers_C1.to_dict()}")
print()

C0 = retirer_outliers(C0)
C1 = retirer_outliers(C1)

n_retires = len(df[df["Condition"] == "C0"]) + len(df[df["Condition"] == "C1"]) - len(C0) - len(C1)
print(f"{n_retires} participant(s) retiré(s) après détection des outliers")
print()


# Statistiques descriptives
print("Statistiques descriptives  (score brut TCT-DP)")
print(f"N C0 = {len(C0)}  |  Moyenne = {C0.mean():.3f}  |  SD = {C0.std():.3f}")
print(f"N C1 = {len(C1)}  |  Moyenne = {C1.mean():.3f}  |  SD = {C1.std():.3f}")
print()


# Test de Levene
lev_stat, lev_p = stats.levene(C0, C1)
print(f"Test de Levene : stat = {lev_stat:.3f}  |  p = {lev_p:.3f}")
print()


# T-test de Welch
t_stat, p = stats.ttest_ind(C0, C1, equal_var=False)
sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

print(f"t = {t_stat:.4f}")
print(f"p = {p:.6f}  {sig}")
print()


# Intervalle de confiance à 95%
diff    = C1.mean() - C0.mean()
se_diff = np.sqrt(C0.std()**2 / len(C0) + C1.std()**2 / len(C1))
df_welch = (C0.std()**2/len(C0) + C1.std()**2/len(C1))**2 / \
           ((C0.std()**2/len(C0))**2/(len(C0)-1) + (C1.std()**2/len(C1))**2/(len(C1)-1))
t_crit  = stats.t.ppf(0.975, df=df_welch)
ic_bas  = diff - t_crit * se_diff
ic_haut = diff + t_crit * se_diff

print(f"Différence des moyennes (C1 − C0) = {diff:.3f}")
print(f"IC 95% = [{ic_bas:.3f} ; {ic_haut:.3f}]")
print()


# Cohen's d
pooled_std = np.sqrt((C0.std()**2 + C1.std()**2) / 2)
d = diff / pooled_std

if abs(d) < 0.2:   taille = "négligeable"
elif abs(d) < 0.5: taille = "petit"
elif abs(d) < 0.8: taille = "moyen"
else:              taille = "grand"

print(f"Cohen's d = {d:.3f}  →  effet {taille}")


# Graphique
fig, ax = plt.subplots(figsize=(7, 5))
couleurs = {"C0": "#438FD2", "C1": "#DE4503"}

for i, (cond, vals) in enumerate({"C0": C0, "C1": C1}.items()):
    bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color=couleurs[cond], linewidth=1.5),
                    capprops=dict(color=couleurs[cond], linewidth=1.5),
                    flierprops=dict(marker="o", color=couleurs[cond], alpha=0.5))
    bp["boxes"][0].set_facecolor(couleurs[cond])
    bp["boxes"][0].set_alpha(0.6)
    jitter = np.random.normal(0, 0.04, size=len(vals))
    ax.scatter([i] + jitter, vals, color=couleurs[cond], s=50, alpha=0.75, zorder=5)

gap    = max(C0.max(), C1.max()) * 0.05
y_bar  = max(C0.max(), C1.max()) + gap
y_p    = y_bar + gap
y_info = y_bar + gap * 3

ax.plot([0, 1], [y_bar, y_bar], color="#444", linewidth=1.2)
ax.text(0.5, y_p,    f"p = {p:.6f} {sig}", ha="center", fontsize=9)
ax.text(0.5, y_info,
        f"Δ = {diff:.3f}  |  IC 95% [{ic_bas:.3f} ; {ic_haut:.3f}]  |  d = {d:.3f}",
        ha="center", fontsize=9, color="#333")

ax.set_xticks([0, 1])
ax.set_xticklabels([f"C0\n(N={len(C0)})", f"C1\n(N={len(C1)})"], fontsize=12)
ax.set_ylabel("Score TCT-DP brut", fontsize=11)
ax.set_title("T-test de Welch : C0 vs C1\n(score brut)", fontsize=11)
ax.set_ylim(0, y_info + gap * 2)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("../figures/ttest.png", dpi=150, bbox_inches="tight")
plt.close()
