#!/usr/bin/env python3
"""
ASR oficial do tempo real — reprodutível, a partir do log da sessão.
====================================================================
Responde ao parecer (itens 1, 2, 3):
  1. Define UMA sessão oficial (realtime_live_log.csv) e gera o número por SCRIPT.
  2. Usa o limiar de operação HERDADO do LFW (calibrado por FAR), não o 0.698 do
     conjunto pessoal de 8 pessoas (que tem separação perfeita e não é ponto de
     operação). Reporta 0.45/0.60 (literatura) e 0.698 (pessoal) só p/ comparação.
  3. Reporta a autocorrelação dos frames (rho lag-1) e o n efetivo, deixando claro
     que a unidade estatística é a SESSÃO (n=1 aqui) — IC por frame superestima.

Uso: .venv/bin/python app/realtime_asr_oficial.py [caminho_do_log]
Gera: app/results/realtime_asr_oficial.txt
"""
import os, sys, csv
import numpy as np
from sklearn.metrics import roc_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "realtime_live_log.csv")

# --- limiares de operação herdados do LFW (recalculados do baseline jeito2) ---
def lfw_thresholds():
    lab, sc = [], []
    for r in csv.DictReader(open(os.path.join(RES, "lfw_scores_jeito12.csv"))):
        lab.append(int(r["label"])); sc.append(float(r["score_j2"]))
    lab, sc = np.array(lab), np.array(sc)
    fpr, tpr, thr = roc_curve(lab, sc); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    def far(t):
        ok = np.where(fpr <= t)[0]; return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return {"EER_LFW": float(thr[i]), "FAR1%_LFW": far(0.01), "FAR0.1%_LFW": far(0.001)}

# --- sims de ataque do log da sessão ---
rows = [r for r in csv.DictReader(open(LOG)) if r.get("mode") == "ATAQUE"]
sims = np.array([float(r["similarity"]) for r in rows])
n = len(sims)

thr_lfw = lfw_thresholds()
# autocorrelação lag-1 e n efetivo
rho = float(np.corrcoef(sims[:-1], sims[1:])[0, 1]) if n > 2 else float("nan")
n_eff = n * (1 - rho) / (1 + rho) if np.isfinite(rho) else float("nan")

def asr(t): return 100.0 * np.mean(sims >= t)

out = []
def P(s=""):
    print(s); out.append(s)

P("=" * 66)
P("ASR OFICIAL DO TEMPO REAL (reprodutível)")
P("=" * 66)
P(f"log da sessão : {os.path.relpath(LOG, REPO)}")
P(f"frames de ataque: {n} | similaridade média: {sims.mean():.3f} (min {sims.min():.3f}, max {sims.max():.3f})")
P("")
P(">> Ponto de operação HERDADO do LFW (o correto):")
for name in ["FAR1%_LFW", "FAR0.1%_LFW", "EER_LFW"]:
    P(f"   {name:<12} thr={thr_lfw[name]:.4f}  ASR={asr(thr_lfw[name]):6.1f}%")
P("")
P(">> Só para comparação (NÃO são ponto de operação):")
P(f"   0.45 (lit.)   thr=0.4500  ASR={asr(0.45):6.1f}%")
P(f"   0.60 (lit.)   thr=0.6000  ASR={asr(0.60):6.1f}%")
P(f"   0.698 (pessoal, overfit 8 pessoas)  ASR={asr(0.698):6.1f}%")
P("")
P(">> Validade estatística:")
P(f"   autocorrelação lag-1 (rho) = {rho:.3f}")
P(f"   n efetivo ≈ {n_eff:.0f} de {n} frames (frames fortemente correlacionados)")
P(f"   unidade estatística = SESSÃO. Aqui n_sessões = 1 -> SEM IC por frame.")
P(f"   Para IC: repetir várias sessões/atacantes/vítimas; cada sessão = 1 tentativa.")

open(os.path.join(RES, "realtime_asr_oficial.txt"), "w").write("\n".join(out))
P("")
P(f"salvo: {os.path.join(RES, 'realtime_asr_oficial.txt')}")
