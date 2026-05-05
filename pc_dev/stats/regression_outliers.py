import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Chargement 
df = pd.read_csv("scores_tctdp.csv").dropna(subset=["Total"]).copy()
df["Condition_num"] = df["Condition"].map({"C0": 0, "C1": 1})
y = df["Total"].values.astype(float)
x = df["Condition_num"].values.astype(float)
n = len(y)

# Regression lineaire
b1, b0, r, p, se = stats.linregress(x, y)
y_pred = b0 + b1 * x
residus = y - y_pred
sigma2 = np.sum(residus**2) / (n - 2)

# ----------------------- Determination des outliers avec levier, rstudent et cook ---------------
# Levier
x_mean = np.mean(x)
h = 1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2)
seuil_levier = 2 * np.mean(h)
out_levier = df["Participant"].values[h > seuil_levier]

# Resident studentisé
rstudent = []
for i in range(n):
    xi, yi = np.delete(x, i), np.delete(y, i)
    b1i, b0i, _, _, _ = stats.linregress(xi, yi)
    s2i = np.sum((yi - (b0i + b1i * xi))**2) / (n - 3)
    rstudent.append(residus[i] / np.sqrt(s2i * (1 - h[i])) if s2i > 0 else 0)
rstudent = np.array(rstudent)
out_rstudent = df["Participant"].values[np.abs(rstudent) > 3]

# Distance de cook
cook = (residus**2 / (2 * sigma2)) * (h / (1 - h)**2)
seuil_cook = 4 / n
out_cook = df["Participant"].values[cook > seuil_cook]

tous_outliers = list(set(list(out_levier) + list(out_rstudent) + list(out_cook)))

# Affichage des outliers
print(f"Levier   : {list(out_levier) if len(out_levier) else 'aucun'}")
print(f"RStudent : {list(out_rstudent) if len(out_rstudent) else 'aucun'}")
print(f"Cook     : {list(out_cook) if len(out_cook) else 'aucun'}")
print(f"\nOutliers : {sorted(tous_outliers) if tous_outliers else 'aucun'}")

df_clean = df[~df["Participant"].isin(tous_outliers)].copy()
y2, x2 = df_clean["Total"].values.astype(float), df_clean["Condition_num"].values.astype(float)
b1c, b0c, _, pc, _ = stats.linregress(x2, y2)

# Intervalle de confiance à 95% pour b1
t_critique = stats.t.ppf(0.975, df=n-2)
ic_bas  = b1 - t_critique * se
ic_haut = b1 + t_critique * se

print(f"b1     = {b1:.4f}")
print(f"IC 95% = [{ic_bas:.4f} ; {ic_haut:.4f}]")

R2  = 1 - np.sum(residus**2) / np.sum((y - np.mean(y))**2)
R2c = 1 - np.sum((y2-(b0c+b1c*x2))**2) / np.sum((y2-np.mean(y2))**2)

print(f"\n{'':20} {'AVEC':>10} {'SANS':>10}")
print(f"{'N':20} {n:>10} {len(df_clean):>10}")
print(f"{'b1':20} {b1:>10.2f} {b1c:>10.2f}")
print(f"{'p-value':20} {p:>10.6f} {pc:>10.6f}")
print(f"{'R²':20} {R2:>10.3f} {R2c:>10.3f}")

df_clean.drop(columns=["Condition_num"]).to_csv("scores_tctdp_sans_outliers.csv", index=False)

participants = df["Participant"].values



# --------------------- Figures ---------------------------------------------------

# Figure levier
plt.figure(figsize=(9, 4))
plt.bar(range(n), h, color=["#DE4503" if hi > seuil_levier else "#438FD2" for hi in h], alpha=0.8)
plt.axhline(seuil_levier, color="red", linestyle="--", label=f"Seuil = {seuil_levier:.3f}")
for i, pid in enumerate(participants):
    if h[i] > seuil_levier:
        plt.text(i, h[i] + 0.002, pid, ha="center", fontsize=8, color="darkred")
plt.xticks([]); plt.ylabel("Levier"); plt.title("Levier — outliers sur X"); plt.legend(); plt.tight_layout()
plt.savefig("fig_levier.png", dpi=150); plt.close()

# Figure rstudent
plt.figure(figsize=(9, 4))
plt.bar(range(n), np.abs(rstudent), color=["#DE4503" if abs(r) > 3 else "#438FD2" for r in rstudent], alpha=0.8)
plt.axhline(3, color="red", linestyle="--", label="|RStudent| = 3")
for i, pid in enumerate(participants):
    if abs(rstudent[i]) > 3:
        plt.text(i, abs(rstudent[i]) + 0.05, pid, ha="center", fontsize=8, color="darkred")
plt.xticks([]); plt.ylabel("|RStudent|"); plt.title("Résidus Studentisés — outliers sur Y"); plt.legend(); plt.tight_layout()
plt.savefig("fig_rstudent.png", dpi=150); plt.close()

# Figure distance de cook
plt.figure(figsize=(9, 4))
plt.bar(range(n), cook, color=["#DE4503" if c > seuil_cook else "#438FD2" for c in cook], alpha=0.8)
plt.axhline(seuil_cook, color="red", linestyle="--", label=f"Seuil 4/n = {seuil_cook:.3f}")
for i, pid in enumerate(participants):
    if cook[i] > seuil_cook:
        plt.text(i, cook[i] + 0.002, pid, ha="center", fontsize=8, color="darkred")
plt.xticks([]); plt.ylabel("D de Cook"); plt.title("Distance de Cook — outliers combinaison X-Y"); plt.legend(); plt.tight_layout()
plt.savefig("fig_cook.png", dpi=150); plt.close()

# Figure regression sans outliers
plt.figure(figsize=(7, 5))
x_line = np.linspace(-0.2, 1.2, 100)
for xi, yi, pid in zip(x, y, participants):
    is_out = pid in tous_outliers
    plt.scatter(xi + np.random.normal(0, 0.02), yi,
                color="red" if is_out else {0:"#438FD2",1:"#DE4503"}[xi],
                marker="X" if is_out else "o",
                s=90 if is_out else 55, alpha=0.9 if is_out else 0.65, zorder=4 if is_out else 3)
plt.plot(x_line, b0+b1*x_line,   color="gray",   linestyle="--", linewidth=2, label=f"Avec  b1={b1:.2f}, p={p:.6f}")
plt.plot(x_line, b0c+b1c*x_line, color="darkred", linestyle="-",  linewidth=2, label=f"Sans  b1={b1c:.2f}, p={pc:.6f}")
plt.xticks([0,1], ["C0","C1"], fontsize=12); plt.ylabel("Score TCT-DP")
plt.text(0.5, 1.5, f"IC 95% de b1 : [{ic_bas:.2f} ; {ic_haut:.2f}]", # Intervalle de confiance
         ha="center", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))
plt.title("Régression avec vs sans outliers"); plt.legend(fontsize=9); plt.tight_layout()
plt.savefig("fig_regression_comparaison.png", dpi=150); plt.close()