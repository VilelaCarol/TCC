#!/usr/bin/env python3
"""
Agregação multi-sessão do real-time (parecer #3) — unidade estatística = SESSÃO.
==============================================================================
Lê vários logs de sessão (cada sessão = uma tentativa de verificação por um
atacante contra uma vítima) e calcula o ASR com IC 95% (Wilson) tendo a SESSÃO
como unidade — não o frame (que é fortemente autocorrelacionado).

Critério: uma sessão é 'aceita' se a MEDIANA das similaridades dos frames de
ataque >= limiar de operação (herdado do LFW, FAR@1% = 0.1562).

Uso:
  .venv/bin/python app/agg_sessoes_realtime.py "app/results/sessoes/*.csv"
"""
import os, sys, csv, glob
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
from sklearn.metrics import roc_curve

def lfw_far1():
    lab, sc = [], []
    for r in csv.DictReader(open(os.path.join(RES, "lfw_scores_jeito12.csv"))):
        lab.append(int(r["label"])); sc.append(float(r["score_j2"]))
    lab, sc = np.array(lab), np.array(sc)
    fpr, tpr, thr = roc_curve(lab, sc)
    ok = np.where(fpr <= 0.01)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))

pattern = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RES, "sessoes", "*.csv")
files = sorted(glob.glob(pattern))
THR = lfw_far1()
print(f"limiar de operação (LFW FAR@1%): {THR:.4f}")
print(f"logs encontrados: {len(files)} em '{pattern}'")
if not files:
    print("Nenhum log de sessão. Rode várias sessões salvando com --log sessoes/sess01.csv etc. (ver protocolo).")
    sys.exit(0)

rows = []
for f in files:
    sims = np.array([float(r["similarity"]) for r in csv.DictReader(open(f)) if r.get("mode") == "ATAQUE"])
    if len(sims) == 0:
        continue
    med = float(np.median(sims)); frac = 100*np.mean(sims >= THR)
    accepted = med >= THR
    rows.append((os.path.basename(f), len(sims), med, frac, accepted))

print(f"\n{'sessão':<28}{'frames':>8}{'mediana':>9}{'%frames>=thr':>13}{'aceita?':>9}")
for name, n, med, frac, acc in rows:
    print(f"{name:<28}{n:>8}{med:>9.3f}{frac:>12.1f}%{'SIM' if acc else 'não':>9}")

N = len(rows); K = sum(1 for r in rows if r[4])
lo, hi = wilson(K, N)
print(f"\n=== ASR por SESSÃO (unidade estatística correta) ===")
print(f"sessões: {N} | aceitas: {K} | ASR_sessão = {100*K/N:.1f}%  (IC95% Wilson: {lo:.1f}–{hi:.1f})")
if N < 10:
    print(f">> AVISO: {N} sessões é pouco p/ um IC estreito. Alvo sugerido: >= 15-20 sessões,")
    print(f"   variando atacante (quem está na câmera) e vítima (--source-image/--ref-image).")
