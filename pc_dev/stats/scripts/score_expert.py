import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Chargement
df_tctdp = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df_todd  = pd.read_csv("../score_todd.csv")

# Score expert = moyenne des deux notes (P56 n'a que crea1 → on prend crea1 seul)
df_todd["note_expert"] = df_todd[["note_crea1", "note_crea2"]].mean(axis=1)

# Fusion avec les données TCT-DP
df_tctdp = df_tctdp.merge(
    df_todd[["Participant", "note_crea1", "note_crea2", "note_expert"]],
    on="Participant", how="left"
)
df = df_tctdp.dropna(subset=["note_expert"])
df = df[df["Tranche_age"] != "Adulte"]

COULEURS = {"C0": "#438FD2", "C1": "#DE4503"}


def retirer_outliers(serie, seuil=2.5):
    if serie.std() == 0 or len(serie) < 3:
        return serie
    z = np.abs((serie - serie.mean()) / serie.std())
    return serie[z <= seuil]


# Statistiques descriptives
print("Effectifs")
print(df.groupby("Condition")["note_expert"].count())
print()
print("Statistiques descriptives — note expert (moyenne des deux juges)")
print(df.groupby("Condition")["note_expert"].describe().round(2))
print()

# T-test de Welch C0 vs C1
C0 = retirer_outliers(df[df["Condition"] == "C0"]["note_expert"])
C1 = retirer_outliers(df[df["Condition"] == "C1"]["note_expert"])

t_stat, p_val = stats.ttest_ind(C0, C1, equal_var=False)
diff   = C1.mean() - C0.mean()
se     = np.sqrt(C0.std()**2 / len(C0) + C1.std()**2 / len(C1))
df_w   = (C0.std()**2/len(C0) + C1.std()**2/len(C1))**2 / \
         ((C0.std()**2/len(C0))**2/(len(C0)-1) + (C1.std()**2/len(C1))**2/(len(C1)-1))
t_crit = stats.t.ppf(0.975, df=df_w)
ic     = (diff - t_crit * se, diff + t_crit * se)
pooled = np.sqrt((C0.std()**2 + C1.std()**2) / 2)
d      = diff / pooled if pooled != 0 else 0
sig    = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
p_str  = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"

print("T-test de Welch — note expert (C0 vs C1)")
print(f"  N C0={len(C0)}, N C1={len(C1)}")
print(f"  Moy C0={C0.mean():.2f} (SD={C0.std():.2f}),  Moy C1={C1.mean():.2f} (SD={C1.std():.2f})")
print(f"  t={t_stat:.3f}  {p_str} {sig}  Δ={diff:.2f}  IC95%=[{ic[0]:.2f} ; {ic[1]:.2f}]  d={d:.3f}")
print()

# Comparaison inter-experts (sur paires complètes)
both   = df.dropna(subset=["note_crea1", "note_crea2"])
r, p_r = stats.pearsonr(both["note_crea1"], both["note_crea2"])
rho, p_rho = stats.spearmanr(both["note_crea1"], both["note_crea2"])
diff_abs = (both["note_crea1"] - both["note_crea2"]).abs()

print(f"Accord inter-experts (N={len(both)} paires complètes) :")
print(f"  Pearson  r   = {r:.3f}  p = {p_r:.4f}")
print(f"  Spearman rho = {rho:.3f}  p = {p_rho:.4f}")
print(f"  Score identique : {(diff_abs==0).sum()}/{len(both)} ({(diff_abs==0).mean()*100:.1f}%)")
print(f"  Diff ≤ 1        : {(diff_abs<=1).sum()}/{len(both)} ({(diff_abs<=1).mean()*100:.1f}%)")
print(f"  Diff moyenne    : {diff_abs.mean():.2f} ± {diff_abs.std():.2f}")
print()


# FIGURE — 3 panels
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Analyse du score expert (moyenne note_crea1 & note_crea2)", fontsize=13, fontweight="bold")


# Panel 1 : Boxplot C0 vs C1
ax = axes[0]
all_vals = []
n_labels = []
for i, cond in enumerate(["C0", "C1"]):
    vals = retirer_outliers(df[df["Condition"] == cond]["note_expert"])
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
    n_labels.append((i, len(vals), COULEURS[cond]))

y_bar  = max(all_vals) + 0.3
y_text = y_bar + 0.2
ic_str = f"[{ic[0]:.2f} ; {ic[1]:.2f}]"
ax.plot([0, 1], [y_bar, y_bar], color="#444", linewidth=1.2)
ax.text(0.5, y_text,
        f"{p_str} {sig}  d={d:.2f}\nΔ={diff:.2f}  IC95% {ic_str}",
        ha="center", va="bottom", fontsize=8.5, linespacing=1.6)
ax.set_ylim(top=y_text + 1.5)
for i, n, color in n_labels:
    ax.text(i, -0.13, f"N={n}", ha="center", fontsize=9, color=color,
            transform=ax.get_xaxis_transform(), va="top", clip_on=False)
ax.set_xticks([0, 1])
ax.set_xticklabels(["C0", "C1"], fontsize=11)
ax.set_ylabel("Note expert moyenne", fontsize=10)
ax.set_title("Note expert (moyenne)\nC0 vs C1", fontsize=11)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)


# Panel 2 : Distribution de la note moyenne par condition
ax2 = axes[1]
notes_vals = sorted(df["note_expert"].unique())
x     = np.arange(len(notes_vals))
width = 0.35
for i, cond in enumerate(["C0", "C1"]):
    counts = [len(df[(df["Condition"] == cond) & (df["note_expert"] == n)]) for n in notes_vals]
    ax2.bar(x + (i - 0.5) * width, counts, width, label=cond,
            color=COULEURS[cond], alpha=0.75)
ax2.set_xticks(x)
ax2.set_xticklabels([str(n) for n in notes_vals], fontsize=10)
ax2.set_xlabel("Note expert (moyenne)", fontsize=10)
ax2.set_ylabel("Nombre de participants", fontsize=10)
ax2.set_title("Distribution de la note expert\npar condition", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)


# Panel 3 : Accord inter-experts (scatter note_crea1 vs note_crea2)
ax3 = axes[2]
for cond in ["C0", "C1"]:
    sub = both[both["Condition"] == cond]
    ax3.scatter(sub["note_crea1"], sub["note_crea2"],
                color=COULEURS[cond], alpha=0.75, s=55, label=cond, zorder=5)

# Droite identité (accord parfait)
lim = [1.5, 6.5]
ax3.plot(lim, lim, color="#aaa", linewidth=1.2, linestyle="--", label="Accord parfait")

# Droite de régression
m, b = np.polyfit(both["note_crea1"], both["note_crea2"], 1)
xs = np.linspace(both["note_crea1"].min(), both["note_crea1"].max(), 100)
ax3.plot(xs, m * xs + b, color="#333", linewidth=1.2, linestyle="-")

r_str = f"r = {r:.3f}  p = {p_r:.4f}"
ax3.text(0.05, 0.95, r_str, transform=ax3.transAxes,
         fontsize=9, va="top", color="#333")
ax3.text(0.05, 0.88,
         f"Diff ≤ 1 : {(diff_abs<=1).mean()*100:.0f}%  |  moy diff = {diff_abs.mean():.2f}",
         transform=ax3.transAxes, fontsize=9, va="top", color="#555")

ax3.set_xlabel("note_crea1 (juge 1)", fontsize=10)
ax3.set_ylabel("note_crea2 (juge 2)", fontsize=10)
ax3.set_title("Accord inter-experts\n(N={} paires complètes)".format(len(both)), fontsize=11)
ax3.set_xlim(lim)
ax3.set_ylim(lim)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)
ax3.spines[["top", "right"]].set_visible(False)


plt.tight_layout()
plt.subplots_adjust(bottom=0.13)
plt.savefig("../figures/score_expert.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure sauvegardée : figures/score_expert.png")
