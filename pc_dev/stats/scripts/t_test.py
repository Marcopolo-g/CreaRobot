import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Chargement 
# On lit le CSV et on supprime P31 qui n'a pas de score Total
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])

# On sépare les deux groupes
C0 = df[df["Condition"] == "C0"]["Total"]
C1 = df[df["Condition"] == "C1"]["Total"]


# Détection et retrait des outliers (z-score par groupe)
# Pour chaque participant, on calcule combien d'écarts-types il s'éloigne
# de la moyenne de SON groupe. Seuil classique : |z| > 2.5
def flag_outliers(serie, seuil=2.5):
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z > seuil]

outliers_C0 = flag_outliers(C0)
outliers_C1 = flag_outliers(C1)

print("Outliers détectés")
if len(outliers_C0) == 0 and len(outliers_C1) == 0:
    print("Aucun outlier détecté (|z| > 2.5) dans les deux groupes")
else:
    if len(outliers_C0) > 0:
        print(f"C0 : {outliers_C0.to_dict()}")
    if len(outliers_C1) > 0:
        print(f"C1 : {outliers_C1.to_dict()}")
print()

# On retire les outliers des deux groupes pour la suite de l'analyse
def retirer_outliers(serie, seuil=2.5):
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z <= seuil]

C0 = retirer_outliers(C0)
C1 = retirer_outliers(C1)

n_retires = len(df[df["Condition"] == "C0"]) + len(df[df["Condition"] == "C1"]) \
            - len(C0) - len(C1)
print(f"{n_retires} participant(s) retiré(s) après détection des outliers")
print()


# Statistiques descriptives
print("Statistiques descriptives")
print(f"N C0 = {len(C0)}  |  Moyenne = {C0.mean():.2f}  |  SD = {C0.std():.2f}")
print(f"N C1 = {len(C1)}  |  Moyenne = {C1.mean():.2f}  |  SD = {C1.std():.2f}")
print()


# Test de Levene
# Vérifie si les deux groupes ont des variances similaires
# Si p > 0.05 → les variances sont égales → c'est bon pour le t-test
lev_stat, lev_p = stats.levene(C0, C1)
print(f"Test de Levene : stat = {lev_stat:.3f}  |  p = {lev_p:.3f}")
print()


# T-test de Welch
# Compare les moyennes des deux groupes
# On utilise Welch (equal_var=False) car il fonctionne dans tous les cas,
# même si les variances ne sont pas exactement égales
t_stat, p_value = stats.ttest_ind(C0, C1, equal_var=False)

print(f"t = {t_stat:.4f}")
print(f"p = {p_value:.6f}  →  {'significatif ***' if p_value < 0.001 else 'significatif' if p_value < 0.05 else 'non significatif'}")
print()


# Intervalle de confiance à 95% sur la différence des moyennes
# IC = différence ± t_critique × erreur standard de la différence
# L'erreur standard tient compte de la taille et dispersion de chaque groupe
diff = C1.mean() - C0.mean()
se_diff = np.sqrt(C0.std()**2 / len(C0) + C1.std()**2 / len(C1))

# Degrés de liberté de Welch (formule approchée)
df_welch = (C0.std()**2/len(C0) + C1.std()**2/len(C1))**2 / \
           ((C0.std()**2/len(C0))**2/(len(C0)-1) + (C1.std()**2/len(C1))**2/(len(C1)-1))

# t_critique pour α = 0.05, bilatéral
t_crit = stats.t.ppf(0.975, df=df_welch)

ic_bas  = diff - t_crit * se_diff
ic_haut = diff + t_crit * se_diff

print(f"Différence des moyennes (C1 − C0) = {diff:.2f} points")
print(f"IC 95% = [{ic_bas:.2f} ; {ic_haut:.2f}]")
print()


# Taille d'effet : Cohen's d
# Mesure à quel point la différence est grande dans la réalité
# Seuils : petit < 0.5 | moyen < 0.8 | grand ≥ 0.8
pooled_std = np.sqrt((C0.std()**2 + C1.std()**2) / 2)
d = (C1.mean() - C0.mean()) / pooled_std

if abs(d) < 0.2:   taille = "négligeable"
elif abs(d) < 0.5: taille = "petit"
elif abs(d) < 0.8: taille = "moyen"
else:              taille = "grand"

print(f"Cohen's d = {d:.3f}  →  effet {taille}")


# Graphique
fig, ax = plt.subplots(figsize=(7, 5))

couleurs = {"C0": "#438FD2", "C1": "#DE4503"}

# Boxplot pour chaque condition
for i, (cond, vals) in enumerate({"C0": C0, "C1": C1}.items()):
    bp = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(color=couleurs[cond], linewidth=1.5),
                    capprops=dict(color=couleurs[cond], linewidth=1.5),
                    flierprops=dict(marker="o", color=couleurs[cond], alpha=0.5))
    bp["boxes"][0].set_facecolor(couleurs[cond])
    bp["boxes"][0].set_alpha(0.6)

    # Points individuels avec un léger décalage pour éviter la superposition
    jitter = np.random.normal(0, 0.04, size=len(vals))
    ax.scatter([i] + jitter, vals, color=couleurs[cond], s=50, alpha=0.75, zorder=5)

# Barre de significativité entre les deux groupes
y_max = max(C0.max(), C1.max()) + 2
ax.plot([0, 1], [y_max, y_max], color="#444", linewidth=1.2)
ax.text(0.5, y_max + 0.5, f"p = {p_value:.6f} ***", ha="center", fontsize=10)

# Intervalle de confiance affiché comme annotation sous la barre de significativité
# (l'IC porte sur la différence C1−C0, pas sur chaque groupe séparément)
ax.text(0.5, y_max - 1.5,
        f"Δ = {diff:.2f} pts  |  IC 95% [{ic_bas:.2f} ; {ic_haut:.2f}]",
        ha="center", fontsize=9, color="#333")

# Mise en forme
ax.set_xticks([0, 1])
ax.set_xticklabels(["C0", "C1"], fontsize=12)
ax.set_ylabel("Score TCT-DP", fontsize=11)
ax.set_title("T-test de Welch : C0 vs C1", fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../figures/ttest.png", dpi=150, bbox_inches="tight")
plt.close()