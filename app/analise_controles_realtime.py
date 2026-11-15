#!/usr/bin/env python3
"""
Controles negativos do real-time (parecer #3) — especificidade ao vivo.
==============================================================================
Lê os logs de CONTROLE de app/results/sessoes/ (imp_*.csv e gen_*.csv), filtra
os frames mode == "PASSTHROUGH" (rosto REAL do atacante, SEM deepfake) e reporta,
por sessão e no agregado, se o sistema os teria ACEITADO como sendo a vítima.

Espera-se que quase nenhuma sessão de controle seja aceita: é o que mostra que o
ASR alto do ataque não é artefato de um verificador frouxo (especificidade).

Regra de decisão (a MESMA do ataque, definida antes da coleta): uma sessão é
'aceita' se a MEDIANA das similaridades dos frames válidos >= limiar de operação
herdado do LFW. Limiares: FAR 1% (~0.1562) e FAR 0.1% (~0.2289).

Observação de reuso: o limiar é a MESMA definição de
app/agg_sessoes_realtime.py::lfw_far1() — mesmo CSV (lfw_scores_jeito12.csv) e
mesmo roc_curve. A lógica é replicada aqui (generalizada p/ o alvo de FAR) porque
agg_sessoes_realtime.py executa no import (não tem guarda __main__) e o enunciado
pede p/ não alterá-lo. O script imprime o limiar calculado p/ conferência
(lfw_far(0.01) deve reproduzir o 0.1562 usado lá).

Saída: app/results/controles_realtime.txt
==============================================================================
"""
import os
import csv
import glob
import numpy as np
from sklearn.metrics import roc_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
SESS = os.path.join(RES, "sessoes")
LFW_SCORES = os.path.join(RES, "lfw_scores_jeito12.csv")


def lfw_far(target):
    """Limiar de operação calibrado por FAR no baseline do LFW (jeito 2).
    Espelha agg_sessoes_realtime.lfw_far1(), generalizado p/ o alvo de FAR."""
    lab, sc = [], []
    for r in csv.DictReader(open(LFW_SCORES)):
        lab.append(int(r["label"])); sc.append(float(r["score_j2"]))
    lab, sc = np.array(lab), np.array(sc)
    fpr, tpr, thr = roc_curve(lab, sc)
    ok = np.where(fpr <= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def wilson(k, n, z=1.96):
    """IC de Wilson (igual ao de agg_sessoes_realtime.py)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def load_passthrough(path):
    """Similaridades dos frames PASSTHROUGH de um log de sessão."""
    sims = []
    for r in csv.DictReader(open(path)):
        if r.get("mode") == "PASSTHROUGH":
            try:
                sims.append(float(r["similarity"]))
            except (KeyError, ValueError):
                pass
    return np.array(sims)


out = []
def P(s=""):
    print(s); out.append(s)


def main():
    THR1 = lfw_far(0.01)      # FAR 1%  (~0.1562)
    THR01 = lfw_far(0.001)    # FAR 0.1% (~0.2289)

    files = sorted(glob.glob(os.path.join(SESS, "imp_*.csv")) +
                   glob.glob(os.path.join(SESS, "gen_*.csv")))

    P("=" * 78)
    P("CONTROLES NEGATIVOS DO REAL-TIME (PASSTHROUGH: rosto real, sem deepfake)")
    P("=" * 78)
    P(f"limiares de operação (LFW jeito 2): FAR1% = {THR1:.4f} | FAR0.1% = {THR01:.4f}")
    P(f"regra: sessão aceita se MEDIANA das similaridades >= limiar")
    P(f"logs de controle (imp_*/gen_*): {len(files)} em {os.path.relpath(SESS, REPO)}/")

    if not files:
        P("")
        P("Nenhum log de controle ainda. Colete com, por exemplo:")
        P('  .venv/bin/python app/rodar_sessoes.py --modo PASSTHROUGH --prefixo imp --duration 30')
        _save()
        return

    rows = []
    for f in files:
        sims = load_passthrough(f)
        if len(sims) == 0:
            P(f"  (aviso) sem frames PASSTHROUGH: {os.path.basename(f)}")
            continue
        med = float(np.median(sims))
        rows.append({
            "nome": os.path.basename(f),
            "n": len(sims),
            "media": float(np.mean(sims)),
            "mediana": med,
            "max": float(np.max(sims)),
            "aceita1": med >= THR1,
            "aceita01": med >= THR01,
        })

    P("")
    P(f"{'sessão':<30}{'frames':>7}{'média':>9}{'mediana':>9}{'máx':>9}{'aceita@1%':>11}{'aceita@0.1%':>13}")
    for r in rows:
        P(f"{r['nome']:<30}{r['n']:>7}{r['media']:>9.3f}{r['mediana']:>9.3f}{r['max']:>9.3f}"
          f"{('SIM' if r['aceita1'] else 'não'):>11}{('SIM' if r['aceita01'] else 'não'):>13}")

    N = len(rows)
    K1 = sum(1 for r in rows if r["aceita1"])
    K01 = sum(1 for r in rows if r["aceita01"])
    lo1, hi1 = wilson(K1, N)
    lo01, hi01 = wilson(K01, N)

    # estatística agregada frame a frame (só descritiva)
    all_sims = np.concatenate([load_passthrough(os.path.join(SESS, r["nome"])) for r in rows])

    P("")
    P("=" * 78)
    P("AGREGADO (unidade = sessão)")
    P("=" * 78)
    P(f"sessões de controle: {N} | frames PASSTHROUGH somados: {len(all_sims)}")
    P(f"similaridade (frames): média {all_sims.mean():.3f} | mediana {np.median(all_sims):.3f} | máx {all_sims.max():.3f}")
    P(f"aceitas @FAR1%  ({THR1:.4f}): {K1}/{N} = {100*K1/N:.1f}%  (IC95% Wilson: {lo1:.1f}-{hi1:.1f})")
    P(f"aceitas @FAR0.1% ({THR01:.4f}): {K01}/{N} = {100*K01/N:.1f}%  (IC95% Wilson: {lo01:.1f}-{hi01:.1f})")
    P("")
    P("Leitura: controle bom = ~0% aceitas (o verificador REJEITA o rosto real do")
    P("atacante). Se muitas sessões de controle forem aceitas, o ASR do ataque perde")
    P("significado (verificador frouxo). Compare com o ASR do ataque (agg_sessoes_realtime.py).")
    _save()


def _save():
    path = os.path.join(RES, "controles_realtime.txt")
    open(path, "w").write("\n".join(out))
    P(f"\nsalvo: {path}")


if __name__ == "__main__":
    main()
