import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Chargement
df_tctdp = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df_todd  = pd.read_csv("../score_todd.csv")

# Ajout de la note expert ; on garde uniquement les participants notés
df_tctdp["note_crea"] = df_tctdp["Participant"].map(
    df_todd.set_index("Participant")["note_crea"]
)
df = df_tctdp.dropna(subset=["note_crea"])
df = df[df["Tranche_age"] != "Adulte"]

COULEURS = {"C0": "#438FD2", "C1": "#DE4503"}


def retirer_outliers(serie, seuil=2.5):
    if serie.std() == 0 or len(serie) < 3:
        return serie
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z <= seuil]


# Effectifs
print("Effectifs après filtrage (toutes tranches d'âge, notés par l'expert)")
print(df.groupby(["Tranche_age", "Condition"])["note_crea"].count().unstack(fill_value=0))
print()

# Statistiques descriptives
print("Statistiques descriptives — note expert")
print(df.groupby("Condition")["note_crea"].describe().round(2))
print()

# T-test de Welch C0 vs C1
C0 = retirer_outliers(df[df["Condition"] == "C0"]["note_crea"])
C1 = retirer_outliers(df[df["Condition"] == "C1"]["note_crea"])

t_stat, p_val = stats.ttest_ind(C0, C1, equal_var=False)
diff = C1.mean() - C0.mean()
se   = np.sqrt(C0.std()**2 / len(C0) + C1.std()**2 / len(C1))
df_w = (C0.std()**2/len(C0) + C1.std()**2/len(C1))**2 / \
       ((C0.std()**2/len(C0))**2/(len(C0)-1) + (C1.std()**2/len(C1))**2/(len(C1)-1))
t_crit = stats.t.ppf(0.975, df=df_w)
ic   = (diff - t_crit * se, diff + t_crit * se)
pooled = np.sqrt((C0.std()**2 + C1.std()**2) / 2)
d    = diff / pooled if pooled != 0 else 0
sig  = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

print("T-test de Welch — note expert (C0 vs C1)")
print(f"  N C0={len(C0)}, N C1={len(C1)}  |  "
      f"Moy C0={C0.mean():.2f} (SD={C0.std():.2f}),  Moy C1={C1.mean():.2f} (SD={C1.std():.2f})")
print(f"  t={t_stat:.3f}  p={p_val:.4f} {sig}  "
      f"Δ={diff:.2f}  IC95%=[{ic[0]:.2f} ; {ic[1]:.2f}]  Cohen's d={d:.3f}")
print()


# Graphique (3 panels)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Analyse du score expert (note créativité)", fontsize=13, fontweight="bold")


# Panel 1 : Boxplot C0 vs C1
ax = axes[0]
all_vals = []
n_labels = []

for i, cond in enumerate(["C0", "C1"]):
    vals = retirer_outliers(df[df["Condition"] == cond]["note_crea"])
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

    n = len(df[df["Condition"] == cond]["note_crea"])
    n_labels.append((i, n, COULEURS[cond]))

# Barre de significativité
if all_vals:
    y_bar  = max(all_vals) + 0.3
    y_text = y_bar + 0.2
    ic_str = f"[{ic[0]:.2f} ; {ic[1]:.2f}]"
    ax.plot([0, 1], [y_bar, y_bar], color="#444", linewidth=1.2)
    ax.text(0.5, y_text,
            f"p={p_val:.4f} {sig}  d={d:.2f}\nΔ={diff:.2f}  IC95% {ic_str}",
            ha="center", va="bottom", fontsize=8.5, linespacing=1.6)
    ax.set_ylim(top=y_text + 1.5)

for i, n, color in n_labels:
    ax.text(i, -0.13, f"N={n}", ha="center", fontsize=9, color=color,
            transform=ax.get_xaxis_transform(), va="top", clip_on=False)

ax.set_xticks([0, 1])
ax.set_xticklabels(["C0", "C1"], fontsize=11)
ax.set_ylabel("Note expert (créativité)", fontsize=10)
ax.set_title("Note expert\nC0 vs C1", fontsize=11)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)


# Panel 2 : Distribution de la note expert par condition
ax2 = axes[1]

notes = sorted(df["note_crea"].unique())
x     = np.arange(len(notes))
width = 0.35

for i, cond in enumerate(["C0", "C1"]):
    counts = [len(df[(df["Condition"] == cond) & (df["note_crea"] == n)]) for n in notes]
    ax2.bar(x + (i - 0.5) * width, counts, width,
            label=cond, color=COULEURS[cond], alpha=0.75)

ax2.set_xticks(x)
ax2.set_xticklabels([str(int(n)) for n in notes], fontsize=10)
ax2.set_xlabel("Note expert", fontsize=10)
ax2.set_ylabel("Nombre de participants", fontsize=10)
ax2.set_title("Distribution de la\nnote expert par condition", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)


# Panel 3 : Note expert par tranche d'âge
ax3 = axes[2]

tranches = [t for t in df["Tranche_age"].unique() if t in df["Tranche_age"].values]
x        = np.arange(len(tranches))
width    = 0.35

for i, cond in enumerate(["C0", "C1"]):
    moyennes = []
    erreurs  = []
    for t in tranches:
        sub = df[(df["Tranche_age"] == t) & (df["Condition"] == cond)]["note_crea"]
        moyennes.append(sub.mean())
        erreurs.append(sub.sem())

    offset = (i - 0.5) * width
    ax3.bar(x + offset, moyennes, width, yerr=erreurs, label=cond,
            color=COULEURS[cond], alpha=0.75,
            error_kw=dict(ecolor="#555", capsize=4))

    for j, t in enumerate(tranches):
        n = len(df[(df["Tranche_age"] == t) & (df["Condition"] == cond)]["note_crea"])
        ax3.text(x[j] + offset, -0.13, f"N={n}", ha="center", fontsize=8,
                 color=COULEURS[cond], transform=ax3.get_xaxis_transform(),
                 va="top", clip_on=False)

ax3.set_xticks(x)
ax3.set_xticklabels(tranches, fontsize=10)
ax3.set_ylabel("Note expert moyenne (± SEM)", fontsize=10)
ax3.set_title("Note expert\npar tranche d'âge (toutes)", fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)


plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
plt.savefig("../figures/score_expert.png", dpi=150, bbox_inches="tight")
plt.close()
