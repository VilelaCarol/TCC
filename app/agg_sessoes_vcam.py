#!/usr/bin/env python3
"""
Agregação do experimento de câmera virtual (PROTOCOLO_VCAM.md, 3.2 a 3.4).
==============================================================================
Unidade estatística = SESSÃO. Uma sessão é aceita quando a MEDIANA das
similaridades atinge o limiar de operação herdado da LFW (FAR 1%).

Duas contas, ambas registradas antes da coleta:
  * principal   : mediana sobre os quadros em que o YOLOv8-Face detectou rosto,
                  com a taxa de detecção reportada por sessão;
  * sensibilidade: quadro sem detecção conta como REJEITADO (similaridade -1).

Quadros marcados como duplicados (mesmo quadro entregue duas vezes pela câmera
virtual) são descartados nas duas contas.

Compara ainda, sessão a sessão, a medida EMULADA (log de A, no laço) com a
medida PELO CANAL (log de B) — comparação pareada que isola o efeito do canal.

Uso:
  .venv/bin/python app/agg_sessoes_vcam.py
  .venv/bin/python app/agg_sessoes_vcam.py --sessoes "app/results/sessoes_vcam/*_B.csv"
"""
import os
import sys
import csv
import glob
import argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
os.chdir(REPO)
from sklearn.metrics import roc_curve

SEM_DETECCAO = -1.0        # valor atribuído ao quadro sem rosto na análise de sensibilidade


def lfw_thresholds():
    """Limiares de operação derivados da linha de base da LFW (jeito 2)."""
    lab, sc = [], []
    for r in csv.DictReader(open(os.path.join(RES, "lfw_scores_jeito12.csv"))):
        lab.append(int(r["label"])); sc.append(float(r["score_j2"]))
    lab, sc = np.array(lab), np.array(sc)
    fpr, tpr, thr = roc_curve(lab, sc)
    def far(t):
        ok = np.where(fpr <= t)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return far(0.01), far(0.001)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def ler_log_b(path):
    """Lê o log do verificador. Devolve dict com as duas séries de similaridade."""
    sims_det, sims_sens, ts = [], [], []
    n_lidos = n_dup = n_unicos = n_det = 0
    for r in csv.DictReader(open(path)):
        n_lidos += 1
        if r.get("t_abs"):
            ts.append(float(r["t_abs"]))
        if r.get("duplicado") == "1":
            n_dup += 1
            continue
        n_unicos += 1
        if r.get("detectado") == "1" and r.get("similarity"):
            s = float(r["similarity"])
            sims_det.append(s)
            sims_sens.append(s)
            n_det += 1
        else:
            sims_sens.append(SEM_DETECCAO)
    return dict(n_lidos=n_lidos, n_dup=n_dup, n_unicos=n_unicos, n_det=n_det,
                sims_det=np.array(sims_det), sims_sens=np.array(sims_sens),
                janela=(min(ts), max(ts)) if ts else None)


def ler_log_a(path, janela=None):
    """Mediana da medida emulada (no laço do atacante), se o log existir.

    `janela` = (t_ini, t_fim) em tempo absoluto: só entram os quadros de A
    escritos enquanto B estava observando, para que a comparação pareada
    emulado x canal cubra o mesmo trecho da sessão (PROTOCOLO_VCAM.md, 3.7).
    """
    if not os.path.exists(path):
        return None
    sims = []
    for r in csv.DictReader(open(path)):
        if not r.get("similarity_emulado"):
            continue
        if janela and r.get("t_abs"):
            t = float(r["t_abs"])
            if not (janela[0] <= t <= janela[1]):
                continue
        sims.append(float(r["similarity_emulado"]))
    return float(np.median(sims)) if sims else None


def main():
    p = argparse.ArgumentParser(description="Agrega as sessões de câmera virtual")
    p.add_argument("--sessoes", default=os.path.join(RES, "sessoes_vcam", "*_B.csv"))
    p.add_argument("--prefixo-ataque", default="atq")
    p.add_argument("--prefixo-controle", default="imp")
    a = p.parse_args()

    thr1, thr01 = lfw_thresholds()
    arquivos = sorted(glob.glob(a.sessoes))
    if not arquivos:
        sys.exit(f"nenhum log encontrado em {a.sessoes}")

    out = []
    def P(s=""):
        print(s); out.append(s)

    P("=" * 94)
    P("INJEÇÃO POR CÂMERA VIRTUAL — AGREGAÇÃO POR SESSÃO")
    P("=" * 94)
    P(f"limiares de operação (LFW jeito 2): FAR1% = {thr1:.4f} | FAR0,1% = {thr01:.4f}")
    P("regra principal: mediana dos quadros COM detecção | sensibilidade: sem detecção = rejeitado")
    P(f"logs: {len(arquivos)} em '{a.sessoes}'")
    P("")

    linhas = []
    for fb in arquivos:
        base = os.path.basename(fb)
        nome = base[:-6] if base.endswith("_B.csv") else base[:-4]
        fa = fb[:-6] + "_A.csv" if base.endswith("_B.csv") else None
        d = ler_log_b(fb)
        med_det = float(np.median(d["sims_det"])) if len(d["sims_det"]) else float("nan")
        med_sens = float(np.median(d["sims_sens"])) if len(d["sims_sens"]) else float("nan")
        taxa_det = 100.0 * d["n_det"] / d["n_unicos"] if d["n_unicos"] else 0.0
        med_emul = ler_log_a(fa, d["janela"]) if fa else None
        linhas.append(dict(
            sessao=nome, grupo=nome.split("_")[0], n_unicos=d["n_unicos"], n_dup=d["n_dup"],
            n_det=d["n_det"], taxa_det=taxa_det, mediana_canal=med_det, mediana_sens=med_sens,
            mediana_emulada=med_emul,
            aceita=bool(med_det >= thr1) if np.isfinite(med_det) else False,
            aceita_sens=bool(med_sens >= thr1) if np.isfinite(med_sens) else False,
            aceita_far01=bool(med_det >= thr01) if np.isfinite(med_det) else False,
        ))

    P(f"{'sessão':<34}{'únicos':>7}{'dup':>6}{'det%':>7}{'mediana':>9}{'sens':>8}{'emul':>8}{'aceita':>8}")
    P("-" * 94)
    for r in linhas:
        emul = "   --" if r["mediana_emulada"] is None else f"{r['mediana_emulada']:8.3f}"
        P(f"{r['sessao']:<34}{r['n_unicos']:>7}{r['n_dup']:>6}{r['taxa_det']:>6.1f}%"
          f"{r['mediana_canal']:>9.3f}{r['mediana_sens']:>8.3f}{emul}"
          f"{'SIM' if r['aceita'] else 'não':>8}")

    for grupo, rotulo in [(a.prefixo_ataque, "ATAQUE"), (a.prefixo_controle, "CONTROLE IMPOSTOR")]:
        g = [r for r in linhas if r["grupo"] == grupo]
        if not g:
            continue
        N = len(g)
        K = sum(r["aceita"] for r in g)
        Ks = sum(r["aceita_sens"] for r in g)
        K01 = sum(r["aceita_far01"] for r in g)
        lo, hi = wilson(K, N)
        los, his = wilson(Ks, N)
        P("")
        P("=" * 94)
        P(f"{rotulo} — unidade = sessão (n = {N})")
        P("=" * 94)
        P(f"  principal     @FAR1%  : {K}/{N} = {100*K/N:.1f}%  (IC95% Wilson: {lo:.1f}–{hi:.1f})")
        P(f"  sensibilidade @FAR1%  : {Ks}/{N} = {100*Ks/N:.1f}%  (IC95% Wilson: {los:.1f}–{his:.1f})")
        P(f"  principal     @FAR0,1%: {K01}/{N} = {100*K01/N:.1f}%")
        P(f"  taxa de detecção por sessão: mín {min(r['taxa_det'] for r in g):.1f}% | "
          f"mediana {np.median([r['taxa_det'] for r in g]):.1f}% | máx {max(r['taxa_det'] for r in g):.1f}%")
        P(f"  quadros duplicados descartados: {sum(r['n_dup'] for r in g)} de "
          f"{sum(r['n_dup'] + r['n_unicos'] for r in g)} lidos")
        if K != Ks:
            P("  ATENÇÃO: as duas regras divergem — reportar as duas e discutir a diferença.")
        else:
            P("  As duas regras concordam.")

        pares = [(r["mediana_emulada"], r["mediana_canal"]) for r in g
                 if r["mediana_emulada"] is not None and np.isfinite(r["mediana_canal"])]
        if len(pares) >= 3:
            emul = np.array([x[0] for x in pares]); canal = np.array([x[1] for x in pares])
            dif = emul - canal
            P("")
            P(f"  comparação pareada emulado x canal (n = {len(pares)} sessões)")
            P(f"    mediana emulada: {np.median(emul):.3f} | mediana pelo canal: {np.median(canal):.3f}")
            P(f"    diferença (emulado - canal): mediana {np.median(dif):+.3f} | "
              f"mín {dif.min():+.3f} | máx {dif.max():+.3f}")
            try:
                from scipy import stats
                w = stats.wilcoxon(emul, canal, alternative="two-sided")
                P(f"    Wilcoxon pareado bilateral (análise secundária): W = {w.statistic:.1f} | p = {w.pvalue:.3g}")
            except Exception as e:
                P(f"    (Wilcoxon não calculado: {e})")

    csv_path = os.path.join(RES, "vcam_sessoes.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        for r in linhas:
            w.writerow(r)
    txt_path = os.path.join(RES, "vcam_resultados.txt")
    open(txt_path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print(f"\nsalvo: {csv_path}")
    print(f"salvo: {txt_path}")


if __name__ == "__main__":
    main()
