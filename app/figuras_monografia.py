#!/usr/bin/env python3
"""
Figuras da monografia (reprodutíveis a partir dos CSVs do repositório, sem GPU).
==============================================================================
Reproduz as três figuras do Capítulo 5 no mesmo desenho usado na monografia:

  fig1_roc.png        — ROC da linha de base (LFW e YTF, jeito 2), eixo de FAR em
                        escala log; círculo = FAR 1%, quadrado = FAR 0,1%.
                        Fonte: lfw_scores_jeito12.csv, ytf_scores_jeito12.csv
  fig2_asr.png        — ASR x limiar sob referência blindada; pontos de operação
                        calibrados (círculo/quadrado) e limiares fixos 0,45/0,60
                        da configuração padrão do verificador.
                        Fonte: lfw/ytf_attack_blindado_scores.csv
  fig3_tempo_real.png — mediana da similaridade por sessão ao vivo por câmera
                        virtual (ataque x controle, log do sistema-alvo), ordenada
                        pelo ataque, com a região de rejeição à esquerda do limiar
                        de operação (FAR 1% da LFW) e a mediana do ataque na
                        condição emulada sobreposta (losango).
                        Fonte: sessoes_vcam/atq_*_B.csv, sessoes_vcam/imp_*_B.csv
                        e sessoes/atq_*.csv

Os limiares são recalculados dos scores com a mesma regra de blindar_refs.py
(roc_curve; maior limiar com FPR <= alvo), de modo que as figuras são
autoconsistentes com as tabelas.

Uso:  .venv/bin/python app/figuras_monografia.py [pasta_de_saida]
      (padrão: app/results/)
"""
import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator
from sklearn.metrics import roc_curve, auc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
SESS = os.path.join(RES, "sessoes")
SESS_VCAM = os.path.join(RES, "sessoes_vcam")
OUT = sys.argv[1] if len(sys.argv) > 1 else RES
os.makedirs(OUT, exist_ok=True)

AZUL, LARANJA = "#1f4e79", "#c0561e"          # LFW, YTF
VERMELHO, CINZA, VERDE = "#b22418", "#555555", "#2e7d32"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titleweight": "normal",
    "legend.frameon": False, "savefig.dpi": 200,
})

def virgula(x, _pos=None, nd=None):
    """Formata número com vírgula decimal (padrão pt-BR)."""
    if nd is None:
        s = f"{x:g}"
    else:
        s = f"{x:.{nd}f}"
    return "0" if s in ("0", "-0", "0.0", "-0.0") else s.replace(".", ",")

def br(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")

# ------------------------------------------------------------------ dados
def carregar_scores(nome):
    lab, sc = [], []
    for r in csv.DictReader(open(os.path.join(RES, nome))):
        lab.append(int(r["label"])); sc.append(float(r["score_j2"]))
    lab, sc = np.array(lab), np.array(sc)
    return lab, sc

def limiares(lab, sc):
    fpr, tpr, thr = roc_curve(lab, sc); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    def far(t):
        ok = np.where(fpr <= t)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr),
            "eer": float((fpr[i] + fnr[i]) / 2) * 100,
            "far1": far(0.01), "far01": far(0.001)}

def ponto(lab, sc, t):
    """(FAR %, TAR %) efetivos no limiar t."""
    return 100 * np.mean(sc[lab == 0] >= t), 100 * np.mean(sc[lab == 1] >= t)

def col1(nome):
    return np.loadtxt(os.path.join(RES, nome), skiprows=1)

lfw_lab, lfw_sc = carregar_scores("lfw_scores_jeito12.csv")
ytf_lab, ytf_sc = carregar_scores("ytf_scores_jeito12.csv")
L, Y = limiares(lfw_lab, lfw_sc), limiares(ytf_lab, ytf_sc)
lfw_blind = col1("lfw_attack_blindado_scores.csv")
ytf_blind = col1("ytf_attack_blindado_scores.csv")

# ================================================================ FIG 1
def fig1_roc():
    fig, ax = plt.subplots(figsize=(8, 5.65))
    for P, lab, sc, nome, cor in [(L, lfw_lab, lfw_sc, "LFW (foto)", AZUL),
                                  (Y, ytf_lab, ytf_sc, "YTF (vídeo)", LARANJA)]:
        ax.step(P["fpr"] * 100, P["tpr"] * 100, where="post", color=cor, lw=2.2,
                label=f"{nome} — AUC {br(P['auc'], 3)}, EER {br(P['eer'], 2)}%")
        for t, m in [(P["far1"], "o"), (P["far01"], "s")]:
            fx, ty = ponto(lab, sc, t)
            ax.plot(fx, ty, m, color=cor, ms=8, mec="white", mew=1.5, zorder=5)
    ax.plot([], [], "o", color="#444", ms=8, label="ponto de operação FAR 1%")
    ax.plot([], [], "s", color="#444", ms=8, label="ponto de operação FAR 0,1%")
    ax.set_xscale("symlog", linthresh=1e-2, linscale=0.2)
    ax.set_xlim(0, 100); ax.set_ylim(86, 100.8)
    ax.xaxis.set_major_locator(FixedLocator([0, 1e-2, 1e-1, 1, 10, 100]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, p: "0" if x == 0 else f"$10^{{{int(round(np.log10(x)))}}}$"))
    ax.yaxis.set_major_formatter(FuncFormatter(virgula))
    ax.set_xlabel("Taxa de falsos aceites (%)")
    ax.set_ylabel("Taxa de aceitação genuína (%)")
    ax.set_title("O sistema separa bem genuínos de impostores nas duas bases")
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(OUT, "fig1_roc.png"); fig.savefig(out, facecolor="white"); plt.close(fig)
    print(f"fig1: {out}  LFW AUC {L['auc']:.4f} EER {L['eer']:.2f}% | YTF AUC {Y['auc']:.4f} EER {Y['eer']:.2f}%")

# ================================================================ FIG 2
def fig2_asr():
    ts = np.linspace(0, 0.8, 801)
    fig, ax = plt.subplots(figsize=(8, 5.65))
    for blind, P, nome, cor in [(lfw_blind, L, "LFW (foto)", AZUL), (ytf_blind, Y, "YTF (vídeo)", LARANJA)]:
        ax.plot(ts, [100 * np.mean(blind >= t) for t in ts], color=cor, lw=2.4, label=nome)
        for t, m in [(P["far1"], "o"), (P["far01"], "s")]:
            ax.plot(t, 100 * np.mean(blind >= t), m, color=cor, ms=8, mec="white", mew=1.5, zorder=5)
    for x in (0.45, 0.60):
        ax.axvline(x, color="#999", lw=1.2, ls=":", zorder=1)
        ax.text(x, 102, br(x, 2), ha="center", va="bottom", color="#777", fontsize=9.5)
    ax.text(0.525, 107, "limiares fixos (configuração padrão)", ha="center", va="bottom",
            color="#777", fontsize=9.5)
    ax.text(0.015, 60, "pontos de operação\ncalibrados por FAR\n(círculo 1%, quadrado 0,1%)",
            color="#444", fontsize=9.5, va="bottom")
    ax.set_xlim(0, 0.8); ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: virgula(x, nd=1)))
    ax.set_xlabel("Limiar de decisão (similaridade do cosseno)")
    ax.set_ylabel("Taxa de sucesso de ataque (%)")
    ax.set_title("A taxa medida depende de onde o limiar é colocado")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.28), fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT, "fig2_asr.png"); fig.savefig(out, facecolor="white"); plt.close(fig)
    print(f"fig2: {out}  ASR@FAR1% LFW {100*np.mean(lfw_blind >= L['far1']):.1f}% "
          f"YTF {100*np.mean(ytf_blind >= Y['far1']):.1f}%")

# ================================================================ FIG 3
def mediana_sessao(path, modo):
    s = [float(r["similarity"]) for r in csv.DictReader(open(path))
         if r["mode"] == modo and r["similarity"] != ""]
    return float(np.median(s))

def identidade(path):
    return os.path.basename(path).rsplit(".", 1)[0].split("_", 2)[2]

def mediana_vcam(path):
    """Regra principal do PROTOCOLO_VCAM.md (3.3/3.4): mediana dos quadros únicos com rosto detectado."""
    s = [float(r["similarity"]) for r in csv.DictReader(open(path))
         if r.get("duplicado") == "0" and r.get("detectado") == "1" and r.get("similarity")]
    return float(np.median(s))

def identidade_vcam(path):
    b = os.path.basename(path).rsplit(".", 1)[0]          # atq_01_Queen_Latifah_B
    return b.split("_", 2)[2].rsplit("_", 1)[0]          # Queen_Latifah

def fig3_tempo_real():
    """Cenário ao vivo por câmera virtual (sessoes_vcam/, log do sistema-alvo B), com a
    mediana da condição emulada (sessoes/, coleta anterior) sobreposta para comparação."""
    thr = L["far1"]
    atq = {identidade_vcam(p): mediana_vcam(p) for p in glob.glob(os.path.join(SESS_VCAM, "atq_*_B.csv"))}
    imp = {identidade_vcam(p): mediana_vcam(p) for p in glob.glob(os.path.join(SESS_VCAM, "imp_*_B.csv"))}
    emul = {identidade(p): mediana_sessao(p, "ATAQUE") for p in glob.glob(os.path.join(SESS, "atq_*.csv"))}
    nomes = sorted(atq, key=lambda n: atq[n], reverse=True)   # maior embaixo, menor no topo
    fig, ax = plt.subplots(figsize=(8.6, 6.5))
    ax.axvspan(-0.15, thr, color="#efefef", zorder=0)
    ax.axvline(thr, color=VERDE, lw=2, zorder=1)
    for y, n in enumerate(nomes):
        ax.plot([imp[n], atq[n]], [y, y], color="#dddddd", lw=1.5, zorder=2)
        ax.plot(imp[n], y, "o", mfc="white", mec=CINZA, mew=2, ms=8, zorder=3)
        ax.plot(atq[n], y, "o", color=VERMELHO, ms=10, zorder=4)
        if n in emul:
            ax.plot(emul[n], y, "D", color="#222222", ms=4.5, zorder=5)
    ax.plot([], [], "o", color=VERMELHO, ms=10, label="ataque por câmera virtual")
    ax.plot([], [], "o", mfc="white", mec=CINZA, mew=2, ms=8, label="controle por câmera virtual (rosto real)")
    ax.plot([], [], "D", color="#222222", ms=4.5, label="ataque na condição emulada")
    ax.set_yticks(range(len(nomes))); ax.set_yticklabels([n.replace("_", " ") for n in nomes])
    ax.set_ylim(-0.9, len(nomes) + 1.6)
    ax.set_xlim(-0.15, 0.78)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: virgula(x, nd=1)))
    ax.text(-0.14, -0.75, "região de rejeição", color="#888", fontsize=10)
    ax.text(thr + 0.012, -0.75, "limiar de operação (FAR 1%)", color=VERDE, fontsize=10)
    ax.set_xlabel("Mediana da similaridade na sessão")
    ax.set_title("Por câmera virtual: todo ataque é aceito; nenhum controle é")
    ax.legend(loc="upper right", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(OUT, "fig3_tempo_real.png"); fig.savefig(out, facecolor="white"); plt.close(fig)
    print(f"fig3: {out}  câmera virtual: ataque {sum(v >= thr for v in atq.values())}/{len(atq)}; "
          f"controle {sum(v >= thr for v in imp.values())}/{len(imp)}; limiar {thr:.4f}; "
          f"menor ataque {min(atq.values()):.3f} x maior controle {max(imp.values()):.3f}; "
          f"identidades com condição emulada sobreposta: {sum(n in emul for n in atq)}")

if __name__ == "__main__":
    fig1_roc(); fig2_asr(); fig3_tempo_real()
