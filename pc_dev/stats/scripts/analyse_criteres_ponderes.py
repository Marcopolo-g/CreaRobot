import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Chargement — Adultes exclus
df = pd.read_csv("../scores_tctdp.csv").dropna(subset=["Total"])
df = df[df["Tranche_age"] != "Adulte"]

CRITERES = ["Cn", "Cm", "Ne", "Cl", "Cth", "Bfd", "Bfi", "Pe", "Hu", "Uc_b", "Uc_c", "Uc_d"]
MAXSCORES = {"Cn": 6, "Cm": 6, "Ne": 6, "Cl": 6, "Cth": 6,
             "Bfd": 3, "Bfi": 3, "Pe": 6, "Hu": 3,
             "Uc_b": 2, "Uc_c": 2, "Uc_d": 2}
COULEURS = {"C0": "#438FD2", "C1": "#DE4503"}

# Normalisation des critères (0–1) + score total normalisé
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

C0_df = df[df["Condition"] == "C0"]
C1_df = df[df["Condition"] == "C1"]


# T-tests bilatéraux par critère (normalisés)
resultats = {}
for c, cn in zip(CRITERES, CRITERES_N):
    a = C0_df[cn].dropna()
    b = C1_df[cn].dropna()

    if a.std() == 0 and b.std() == 0:
        resultats[c] = {"p": np.nan, "d": 0.0, "sig": "ns"}
        continue

    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.std()**2 + b.std()**2) / 2)
    d      = (b.mean() - a.mean()) / pooled if pooled != 0 else 0
    sig    = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    resultats[c] = {"p": p, "d": d, "sig": sig}

res_df = pd.DataFrame(resultats).T

# Critères significatifs (p < 0.05, d > 0)
crit_sig = res_df[(res_df["p"] < 0.05) & (res_df["d"] > 0)].index.tolist()
print(f"Critères significatifs (p < 0.05, d > 0) : {crit_sig}")
print()

print(f"  {'Critère':<10}  {'p':>8}  {'d':>7}  {'sig':>5}")
print("  " + "-" * 35)
for c, row in res_df.sort_values("p").iterrows():
    print(f"  {c:<10}  {row['p']:>8.4f}  {row['d']:>7.3f}  {row['sig']:>5}")
print()


# Score restreint : somme des critères normalisés significatifs (max = nb critères sig.)
if crit_sig:
    crit_sig_n = [c + "_n" for c in crit_sig]
    df["Score_restreint"] = df[crit_sig_n].sum(axis=1)

    C0_r = df[df["Condition"] == "C0"]["Score_restreint"]
    C1_r = df[df["Condition"] == "C1"]["Score_restreint"]

    t, p  = stats.ttest_ind(C0_r, C1_r, equal_var=False)
    diff  = C1_r.mean() - C0_r.mean()
    se    = np.sqrt(C0_r.std()**2/len(C0_r) + C1_r.std()**2/len(C1_r))
    df_w  = (C0_r.std()**2/len(C0_r) + C1_r.std()**2/len(C1_r))**2 / \
            ((C0_r.std()**2/len(C0_r))**2/(len(C0_r)-1) + (C1_r.std()**2/len(C1_r))**2/(len(C1_r)-1))
    ic    = (diff - stats.t.ppf(0.975, df_w) * se, diff + stats.t.ppf(0.975, df_w) * se)
    pooled = np.sqrt((C0_r.std()**2 + C1_r.std()**2) / 2)
    d     = diff / pooled if pooled != 0 else 0
    sig   = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

    print(f"Score restreint ({len(crit_sig)} critères : {crit_sig})")
    print(f"  Moy C0={C0_r.mean():.3f} (SD={C0_r.std():.3f}),  Moy C1={C1_r.mean():.3f} (SD={C1_r.std():.3f})")
    print(f"  t={t:.3f}  p={p:.4f} {sig}  Δ={diff:.3f}  IC95%=[{ic[0]:.3f} ; {ic[1]:.3f}]  d={d:.3f}")
    print()

    # Boxplot score restreint
    fig_r, ax_r = plt.subplots(figsize=(6, 5))
    for i, (cond, vals) in enumerate({"C0": C0_r, "C1": C1_r}.items()):
        bp = ax_r.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                          medianprops=dict(color="white", linewidth=2.5),
                          whiskerprops=dict(color=COULEURS[cond], linewidth=1.5),
                          capprops=dict(color=COULEURS[cond], linewidth=1.5),
                          flierprops=dict(marker="o", color=COULEURS[cond], alpha=0.5))
        bp["boxes"][0].set_facecolor(COULEURS[cond])
        bp["boxes"][0].set_alpha(0.6)
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax_r.scatter([i] + jitter, vals, color=COULEURS[cond], s=45, alpha=0.75, zorder=5)

    y_bar  = max(C0_r.max(), C1_r.max()) + C0_r.std() * 0.3
    y_text = y_bar + C0_r.std() * 0.3
    ic_str = f"[{ic[0]:.2f} ; {ic[1]:.2f}]"
    ax_r.plot([0, 1], [y_bar, y_bar], color="#444", linewidth=1.2)
    p_str = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    ax_r.text(0.5, y_text,
              f"p={p_str} {sig}  d={d:.3f}\nΔ={diff:.3f}  IC95% {ic_str}",
              ha="center", va="bottom", fontsize=8.5, linespacing=1.6)
    ax_r.set_ylim(top=y_text + C0_r.std())
    ax_r.set_xticks([0, 1])
    ax_r.set_xticklabels([f"C0\n(N={len(C0_r)})", f"C1\n(N={len(C1_r)})"], fontsize=11)
    ax_r.set_ylabel(f"Score restreint normalisé (max = {len(crit_sig)})", fontsize=10)
    ax_r.set_title(f"Score restreint \n({len(crit_sig)} critères significatifs, max=7)", fontsize=10)
    ax_r.grid(alpha=0.3)
    ax_r.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("../figures/criteres_ponderes.png", dpi=150, bbox_inches="tight")
    plt.close()
else:
    print("Aucun critère significatif → score restreint non calculé")
    print()


# Graphique Cohen's d par critère (barres horizontales)
ds_sorted     = res_df["d"].sort_values(ascending=True)
couleurs_bars = ["#DE4503" if c in crit_sig else "#9E9E9E" for c in ds_sorted.index]

fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(range(len(ds_sorted)), ds_sorted.values, color=couleurs_bars, alpha=0.85)

for ref, label in [(0.2, "petit"), (0.5, "moyen"), (0.8, "large")]:
    ax.axvline(ref, color="#bbb", linewidth=1, linestyle="--")
    ax.text(ref + 0.02, len(ds_sorted) - 0.5, label, fontsize=7.5, color="#999")

ax.axvline(0, color="#444", linewidth=1)

for i, (c, d) in enumerate(ds_sorted.items()):
    offset = 0.03 if d >= 0 else -0.03
    ha     = "left" if d >= 0 else "right"
    col    = "#DE4503" if c in crit_sig else "#555"
    ax.text(d + offset, i, f"{d:.3f}", va="center", ha=ha, fontsize=8.5, color=col)

ax.set_yticks(range(len(ds_sorted)))
ax.set_yticklabels(ds_sorted.index, fontsize=10)
ax.set_xlabel("Cohen's d  [C1 − C0]", fontsize=10)
ax.set_title("Cohen's d par critère\nOrange = significatif (p < 0.05)", fontsize=11)
ax.grid(axis="x", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("../figures/criteres_cohens_d.png", dpi=150, bbox_inches="tight")
plt.close()
