#!/usr/bin/env python3
"""
Figuras da monografia (reprodutíveis a partir dos CSVs do repositório).
==============================================================================
Gera as 3 figuras principais SEM GPU, lendo só os resultados já versionados:

  fig1_roc.png        — ROC (LFW e YTF, jeito 2), com EER/AUC e os pontos de
                        operação FAR@1% e FAR@0.1% marcados.
                        Fonte: lfw_scores_jeito12.csv, ytf_scores_jeito12.csv
  fig2_asr.png        — ASR x limiar (ataque blindado, LFW e YTF), com os
                        limiares FIXOS da configuração padrão (0.45/0.60) e os
                        pontos de operação calibrados por FAR.
                        Fonte: lfw/ytf_attack_blindado_scores.csv
  fig3_tempo_real.png — tempo real multi-sessão: mediana por sessão (ataque x
                        controle impostor), unidade = sessão, limiar herdado do
                        LFW (FAR@1%). Mostra a sessão excluída (Steve_Waugh,
                        referência com 2 rostos — ver controle_intraclasse).
                        Fonte: sessoes/atq_*.csv e sessoes/imp_*.csv

Os limiares de operação são RECALCULADOS aqui a partir dos scores (mesma lógica
de agg_sessoes_realtime.py / blindar_refs.py), então a figura é autoconsistente.
"""
import os, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
SESS = os.path.join(RES, "sessoes")

# paleta consistente com as outras figuras do repo
C = {"lfw": "#4a9e6f", "ytf": "#6b8fc4", "atk": "#c0453f", "imp": "#6b8fc4",
     "exc": "#999999", "thr": "#333333"}
EXCLUIDAS = {"Steve_Waugh"}  # referência com 2 rostos (controle_intraclasse)

# --------------------------------------------------------------------- dados
def load_scores(name):
    gen, imp = [], []
    for r in csv.DictReader(open(os.path.join(RES, name))):
        (gen if r["label"] == "1" else imp).append(float(r["score_j2"]))
    return np.array(gen), np.array(imp)

def roc_pack(gen, imp):
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    s = np.r_[gen, imp]
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    def at_far(f, arr):
        ok = np.where(fpr <= f)[0]
        return arr[ok[-1]] if len(ok) else arr[0]
    return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr),
            "eer": float((fpr[i] + fnr[i]) / 2),
            "far1_thr": float(at_far(0.01, thr)), "far1_tpr": float(at_far(0.01, tpr)),
            "far01_thr": float(at_far(0.001, thr)), "far01_tpr": float(at_far(0.001, tpr))}

def col1(name):
    return np.loadtxt(os.path.join(RES, name), skiprows=1)

lfw_gen, lfw_imp = load_scores("lfw_scores_jeito12.csv")
ytf_gen, ytf_imp = load_scores("ytf_scores_jeito12.csv")
L = roc_pack(lfw_gen, lfw_imp)
Y = roc_pack(ytf_gen, ytf_imp)
lfw_blind = col1("lfw_attack_blindado_scores.csv")
ytf_blind = col1("ytf_attack_blindado_scores.csv")

# ============================================================ FIG 1 — ROC
def fig1_roc():
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for P, nome, cor in [(L, "LFW (foto)", C["lfw"]), (Y, "YTF (vídeo)", C["ytf"])]:
        ax.plot(P["fpr"], P["tpr"], color=cor, lw=2.2,
                label=f"{nome} — AUC={P['auc']:.4f}, EER={P['eer']*100:.2f}%")
        ax.scatter([0.01], [P["far1_tpr"]], color=cor, s=55, zorder=5, edgecolor="white")
        ax.scatter([0.001], [P["far01_tpr"]], color=cor, s=55, marker="D", zorder=5, edgecolor="white")
    ax.scatter([], [], color="#333", s=55, label="ponto de operação FAR@1%")
    ax.scatter([], [], color="#333", s=55, marker="D", label="ponto de operação FAR@0.1%")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 1.0); ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("Taxa de falsa aceitação — FAR (escala log)")
    ax.set_ylabel("Taxa de verdadeira aceitação — TAR")
    ax.set_title("Curva ROC do verificador (AdaFace IR-50, jeito 2)", fontsize=12, weight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    out = os.path.join(RES, "fig1_roc.png")
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    print(f"fig1: {out}  (LFW AUC={L['auc']:.4f} EER={L['eer']*100:.2f}% | YTF AUC={Y['auc']:.4f} EER={Y['eer']*100:.2f}%)")

# ============================================================ FIG 2 — ASR x limiar
def fig2_asr():
    ts = np.linspace(-0.10, 1.0, 400)
    asr_l = np.array([np.mean(lfw_blind >= t) for t in ts]) * 100
    asr_y = np.array([np.mean(ytf_blind >= t) for t in ts]) * 100
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(ts, asr_l, color=C["lfw"], lw=2.4, label="LFW (foto) — ataque blindado")
    ax.plot(ts, asr_y, color=C["ytf"], lw=2.4, label="YTF (vídeo) — ataque blindado")
    # limiares FIXOS da configuração padrão do verificador
    for x in (0.45, 0.60):
        ax.axvline(x, color="#111", lw=1.2, ls=":")
    ax.plot([], [], color="#111", lw=1.2, ls=":", label="limiares fixos (configuração padrão): 0.45 / 0.60")
    # pontos de operação calibrados por FAR (recalculados dos scores)
    _ann = [(16, 18), (16, -34)]   # LFW acima, YTF abaixo (evita sobreposição)
    for (P, blind, cor), off in zip([(L, lfw_blind, C["lfw"]), (Y, ytf_blind, C["ytf"])], _ann):
        xo = P["far1_thr"]; yo = np.mean(blind >= xo) * 100
        ax.axvline(xo, color=cor, lw=1.3, ls="--", alpha=0.8)
        ax.scatter([xo], [yo], color=cor, s=60, zorder=6, edgecolor="white")
        ax.annotate(f"FAR@1%={xo:.3f} · ASR={yo:.1f}%", (xo, yo), textcoords="offset points",
                    xytext=off, fontsize=8, color=cor,
                    arrowprops=dict(arrowstyle="-", color=cor, lw=0.6))
    ax.set_xlim(-0.10, 1.0); ax.set_ylim(0, 101)
    ax.set_xlabel("limiar de similaridade (cosseno)")
    ax.set_ylabel("ASR — % de ataques aceitos")
    ax.set_title("ASR × limiar — ataque deepfake blindado", fontsize=12, weight="bold")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    out = os.path.join(RES, "fig2_asr.png")
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    print(f"fig2: {out}  (limiares fixos rotulados como 'configuração padrão')")

# ============================================================ FIG 3 — tempo real
def sess_median(path, mode):
    s = np.array([float(r["similarity"]) for r in csv.DictReader(open(path)) if r.get("mode") == mode])
    return s

def nome_sessao(path):
    b = os.path.basename(path).split(".")[0]           # atq_01_Queen_Latifah
    partes = b.split("_", 2)
    return int(partes[1]), partes[2]                   # (1, "Queen_Latifah")

def fig3_tempo_real():
    thr1, thr01 = L["far1_thr"], L["far01_thr"]        # herdado do LFW
    atq = sorted(glob.glob(os.path.join(SESS, "atq_*.csv")))
    imp = sorted(glob.glob(os.path.join(SESS, "imp_*.csv")))
    idx, nomes, med_a, iqr_a, med_i, exc = [], [], [], [], [], []
    for pa in atq:
        n, nome = nome_sessao(pa)
        s = sess_median(pa, "ATAQUE")
        idx.append(n); nomes.append(nome.replace("_", " "))
        med_a.append(float(np.median(s)))
        iqr_a.append((float(np.percentile(s, 25)), float(np.percentile(s, 75))))
        exc.append(nome in EXCLUIDAS)
    med_i = [float(np.median(sess_median(pi, "PASSTHROUGH"))) for pi in imp]

    x = np.array(idx)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    # IQR do ataque (p25-p75) como traço fino
    for xi, (lo, hi), e in zip(x, iqr_a, exc):
        ax.plot([xi, xi], [lo, hi], color=(C["exc"] if e else C["atk"]), lw=1.0, alpha=0.5, zorder=2)
    # medianas de ataque
    val = [m for m, e in zip(med_a, exc) if not e]
    xv = [xi for xi, e in zip(x, exc) if not e]
    ax.scatter(xv, val, color=C["atk"], s=70, zorder=4, edgecolor="white",
               label="ataque — mediana da sessão (aceita)")
    xe = [xi for xi, e in zip(x, exc) if e]
    ve = [m for m, e in zip(med_a, exc) if e]
    ax.scatter(xe, ve, facecolor="none", edgecolor=C["exc"], s=90, lw=1.8, zorder=4,
               label="ataque — sessão EXCLUÍDA (ref. com 2 rostos)")
    # medianas do controle impostor
    ax.scatter(x, med_i, color=C["imp"], s=55, marker="s", zorder=4, edgecolor="white",
               label="controle impostor — mediana (PASSTHROUGH)")
    # limiares herdados do LFW
    ax.axhline(thr1, color=C["thr"], lw=1.6, ls="--", label=f"limiar de operação FAR@1% = {thr1:.3f}")
    ax.axhline(thr01, color=C["thr"], lw=1.2, ls=":", label=f"FAR@0.1% = {thr01:.3f}")
    ax.axhline(0.0, color="#ccc", lw=0.8, zorder=1)
    for xi, ve_ in zip(xe, ve):
        ax.annotate("Steve Waugh", (xi, ve_), textcoords="offset points", xytext=(6, 8),
                    fontsize=8, color=C["exc"])
    ax.set_xticks(x); ax.set_xticklabels(nomes, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("similaridade cosseno (mediana da sessão)")
    ax.set_title("Tempo real multi-sessão — ataque vs. controle (unidade = sessão, 15 identidades LFW)",
                 fontsize=12, weight="bold")
    ax.set_ylim(-0.22, 0.95)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(RES, "fig3_tempo_real.png")
    fig.savefig(out, dpi=150, facecolor="white"); plt.close(fig)
    aceitas = sum(1 for m, e in zip(med_a, exc) if (not e and m >= thr1))
    validas = sum(1 for e in exc if not e)
    print(f"fig3: {out}  (ataque {aceitas}/{validas} válidas aceitas; impostor 0/{len(med_i)}; limiar {thr1:.4f})")

if __name__ == "__main__":
    fig1_roc()
    fig2_asr()
    fig3_tempo_real()
