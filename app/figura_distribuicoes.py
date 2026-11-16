#!/usr/bin/env python3
"""Figura das distribuições: self vs blindado (LFW, YTF, real-time)."""
import os, sys, csv, glob, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app")); sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
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

# ---- Real-time: baseline pessoal (AdaFace) + ataque do log ----
import cv2
from src import AdaFaceVerifier
from src.utils import get_similarity
v = AdaFaceVerifier(os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml"))
ORIG = os.path.join(REPO, "data", "dataset_pessoal", "originais")
IDS = sorted(["paulista", "mateus", "victor", "carol", "joao", "jones", "lucas", "maria", "paulo"], key=len, reverse=True)
def ident(f):
    b = os.path.basename(f).lower()
    return next((i for i in IDS if b.startswith(i)), None)
byid = {}
for f in sorted(glob.glob(ORIG + "/*")):
    i = ident(f)
    if i: byid.setdefault(i, []).append(f)
E = {}
for fs in byid.values():
    for f in fs:
        e = v.get_embedding(cv2.imread(f)); E[f] = None if e is None else e.cpu().numpy()
pg, pi = [], []
for i, fs in byid.items():
    for a, b in itertools.combinations(fs, 2):
        if E[a] is not None and E[b] is not None: pg.append(get_similarity(E[a], E[b]))
names = list(byid)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        for a in byid[names[i]]:
            for b in byid[names[j]]:
                if E[a] is not None and E[b] is not None: pi.append(get_similarity(E[a], E[b]))
rt_gen, rt_imp = np.array(pg), np.array(pi)
rt_blind = np.array([float(r["similarity"]) for r in csv.DictReader(open(os.path.join(RES, "realtime_live_log.csv"))) if r["mode"] == "ATAQUE"])

# ---- plot ----
C = {"imp": "#6b8fc4", "gen": "#4a9e6f", "self": "#d98a3a", "blind": "#c0453f"}
panels = [
    ("LFW (foto)", lfw_imp, lfw_gen, lfw_self, lfw_blind, 0.156, "FAR@1% = 0.156"),
    ("YTF (vídeo)", ytf_imp, ytf_gen, ytf_self, ytf_blind, 0.162, "FAR@1% = 0.162"),
    ("Real-time (ao vivo)", rt_imp, rt_gen, None, rt_blind, 0.698, "oper. = 0.698"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
bins = np.linspace(-0.15, 1.0, 46)
for ax, (title, imp, gen, self_a, blind_a, thr, thr_lbl) in zip(axes, panels):
    ax.hist(imp, bins=bins, density=True, color=C["imp"], alpha=0.55, label="impostor")
    ax.hist(gen, bins=bins, density=True, color=C["gen"], alpha=0.55, label="genuíno")
    if self_a is not None:
        ax.hist(self_a, bins=bins, density=True, histtype="step", lw=2.0, ls="--",
                color=C["self"], label="ataque (self)")
    else:
        ax.axvline(0.82, color=C["self"], ls="--", lw=2.0, label="ataque self (~0.82)")
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
print(f"real-time: genuina={rt_gen.mean():.3f} impostor={rt_imp.mean():.3f} ataque_blind={rt_blind.mean():.3f} n={len(rt_blind)}")
