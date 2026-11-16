#!/usr/bin/env python3
"""Figura das distribuições: self vs blindado (LFW, YTF)."""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
RES = os.path.join(REPO, "app", "results")

def col(path, name):
    return np.array([float(r[name]) for r in csv.DictReader(open(path)) if r[name]])

def col1(path):  # arquivo de 1 coluna com header
    return np.loadtxt(path, skiprows=1)

# ---- LFW ----
lfw = {r: [] for r in ("gen", "imp")}
for r in csv.DictReader(open(os.path.join(RES, "lfw_scores_jeito12.csv"))):
    (lfw["gen"] if r["label"] == "1" else lfw["imp"]).append(float(r["score_j2"]))
lfw_gen, lfw_imp = np.array(lfw["gen"]), np.array(lfw["imp"])
lfw_self = col(os.path.join(RES, "lfw_swap_attack_jeito12.csv"), "similarity_j2")
lfw_blind = col1(os.path.join(RES, "lfw_attack_blindado_scores.csv"))

# ---- YTF ----
ytf = {r: [] for r in ("gen", "imp")}
for r in csv.DictReader(open(os.path.join(RES, "ytf_scores_jeito12.csv"))):
    (ytf["gen"] if r["label"] == "1" else ytf["imp"]).append(float(r["score_j2"]))
ytf_gen, ytf_imp = np.array(ytf["gen"]), np.array(ytf["imp"])
ytf_self = col(os.path.join(RES, "ytf_swap_attack_jeito12.csv"), "similarity_j2")
ytf_blind = col1(os.path.join(RES, "ytf_attack_blindado_scores.csv"))

# ---- plot ----
C = {"imp": "#6b8fc4", "gen": "#4a9e6f", "self": "#d98a3a", "blind": "#c0453f"}
panels = [
    ("LFW (foto)", lfw_imp, lfw_gen, lfw_self, lfw_blind, 0.156, "FAR@1% = 0.156"),
    ("YTF (vídeo)", ytf_imp, ytf_gen, ytf_self, ytf_blind, 0.162, "FAR@1% = 0.162"),
]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=False)
bins = np.linspace(-0.15, 1.0, 46)
for ax, (title, imp, gen, self_a, blind_a, thr, thr_lbl) in zip(axes, panels):
    ax.hist(imp, bins=bins, density=True, color=C["imp"], alpha=0.55, label="impostor")
    ax.hist(gen, bins=bins, density=True, color=C["gen"], alpha=0.55, label="genuíno")
    ax.hist(self_a, bins=bins, density=True, histtype="step", lw=2.0, ls="--",
            color=C["self"], label="ataque (self)")
    ax.hist(blind_a, bins=bins, density=True, color=C["blind"], alpha=0.62, label="ataque (blindado)")
    ax.axvline(thr, color="#555", lw=1.6, label=thr_lbl)
    ax.axvline(0.60, color="#111", lw=1.2, ls=":", label="0.60 (fixo)")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("similaridade cosseno")
    ax.set_xlim(-0.15, 1.0)
    ax.legend(fontsize=7.5, loc="upper center")
axes[0].set_ylabel("densidade")
fig.suptitle("Distribuições jeito 2 — deslocamento do ataque self → blindado (referência separada)",
             fontsize=13, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(RES, "distribuicoes_blindado.png")
fig.savefig(out, dpi=140, facecolor="white")
print(f"figura: {out}")
