#!/usr/bin/env python3
"""
Controle de variação intraclasse do LFW (real-time, revisão do protocolo item 3).
==============================================================================
Antes de concluir que o ataque FALHOU numa identidade, é preciso saber se as
DUAS fotos daquela pessoa no LFW (fotoA = fonte do swap, fotoB = referência)
sequer casam entre si. Se a similaridade genuína A-B for baixa (idade/pose/luz
muito diferentes), nem um swap perfeito casaria — a "falha" seria variação
intraclasse do próprio dataset, não falha do ataque.

Para cada identidade de app/results/sessoes_plano.json:
  - extrai embedding de fotoA e fotoB com o MESMO AdaFaceVerifier/jeito 2 das sessões
  - similaridade cosseno A-B = similaridade genuína intraclasse
  - lê a similaridade do ataque da sessão correspondente (mediana dos frames
    ATAQUE de app/results/sessoes/atq_<n>_<id>.csv — a mesma regra do agg)
  - razão = ataque / (A-B)

Destaca identidades com A-B < 0.40: nelas um ataque malsucedido pode refletir a
variação intraclasse do LFW, não falha da transferência de identidade.

Saída: app/results/controle_intraclasse.txt
==============================================================================
"""
import os
import sys
import csv
import json
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
sys.path.insert(0, os.path.join(REPO, "app"))
os.chdir(REPO)
import cv2
from src import AdaFaceVerifier
from src.utils import get_similarity, detector

CONFIG = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")
RES = os.path.join(REPO, "app", "results")
SESS = os.path.join(RES, "sessoes")
PLANO = os.path.join(RES, "sessoes_plano.json")
AB_BAIXO = 0.40  # abaixo disso, a variação intraclasse do LFW compromete a leitura

out = []
def P(s=""):
    print(s); out.append(s)


def nfaces(path):
    """Quantos rostos o detector acha na foto (>1 = referência ambígua)."""
    img = cv2.imread(path)
    if img is None:
        return -1
    return int(len(detector(img, verbose=False)[0].boxes))


def ataque_mediana(n, idn):
    """Mediana das similaridades dos frames ATAQUE da sessão (regra do agg)."""
    f = os.path.join(SESS, f"atq_{n:02d}_{idn}.csv")
    if not os.path.exists(f):
        return None
    sims = [float(r["similarity"]) for r in csv.DictReader(open(f)) if r.get("mode") == "ATAQUE"]
    return float(np.median(sims)) if sims else None


def main():
    if not os.path.exists(PLANO):
        print(f"ERRO: plano não encontrado: {PLANO} (rode o Passo 0 primeiro)."); sys.exit(1)
    plano = sorted(json.load(open(PLANO)), key=lambda x: x["n"])

    print(" Carregando AdaFace + YOLO (jeito 2)...")
    verifier = AdaFaceVerifier(CONFIG)  # align=True por padrão (mesmo das sessões)
    print(f"   device: {verifier.device} | align(jeito 2): {verifier.align}\n")

    def emb(path):
        img = cv2.imread(path)
        if img is None:
            return None
        e = verifier.get_embedding(img)  # detecta + alinha (jeito 2), igual à referência das sessões
        return e.cpu().numpy() if e is not None else None

    linhas = []
    for x in plano:
        n, idn = x["n"], x["id"]
        ea, eb = emb(x["fotoA"]), emb(x["fotoB"])
        ab = get_similarity(ea, eb) if (ea is not None and eb is not None) else None
        atk = ataque_mediana(n, idn)
        razao = (atk / ab) if (atk is not None and ab not in (None, 0)) else None
        nA, nB = nfaces(x["fotoA"]), nfaces(x["fotoB"])
        linhas.append({"n": n, "id": idn, "ab": ab, "atk": atk, "razao": razao,
                       "nA": nA, "nB": nB,
                       "flag": (ab is not None and ab < AB_BAIXO),
                       "multi": (nA > 1 or nB > 1)})

    P("=" * 86)
    P("CONTROLE DE VARIAÇÃO INTRACLASSE DO LFW (fotoA x fotoB da mesma pessoa)")
    P("=" * 86)
    P(f"verificador: AdaFace IR-50 WebFace4M | jeito 2 (alinhamento ArcFace 5 pts, BGR)")
    P(f"ataque = mediana dos frames ATAQUE da sessão | flag = A-B < {AB_BAIXO:.2f}")
    P("")
    P(f"{'n':>3}  {'identidade':<24}{'A-B':>8}{'ataque':>8}{'razão':>7}{'rostos A/B':>12}  obs")
    P("-" * 86)
    def fmt(v, nd=3):
        return f"{v:.{nd}f}" if v is not None else "   -"
    for r in linhas:
        obs = []
        if r["flag"]: obs.append("A-B BAIXO")
        if r["multi"]: obs.append("ref. multi-rosto")
        obs = ("   " + "; ".join(obs)) if obs else ""
        P(f"{r['n']:>3}  {r['id']:<24}{fmt(r['ab']):>8}{fmt(r['atk']):>8}{fmt(r['razao'],2):>7}"
          f"{(str(r['nA'])+'/'+str(r['nB'])):>12}{obs}")

    # estatísticas da coluna A-B (referência do que é 'normal' de genuína no LFW)
    abs_ok = [r["ab"] for r in linhas if r["ab"] is not None]
    P("")
    P(f"A-B genuína (15 identidades): média {np.mean(abs_ok):.3f} | mediana {np.median(abs_ok):.3f} "
      f"| mín {np.min(abs_ok):.3f} | máx {np.max(abs_ok):.3f}")

    flagged = [r for r in linhas if r["flag"]]
    multi = [r for r in linhas if r["multi"]]
    P("")
    P("=" * 86)
    P("LEITURA")
    P("=" * 86)
    if multi:
        P("Referências com MAIS DE UM ROSTO (detecção pode ter pego a pessoa errada):")
        for r in multi:
            P(f"  - {r['id']}: fotoA={r['nA']} rosto(s), fotoB={r['nB']} rosto(s), A-B={fmt(r['ab'])}. "
              f"Se A-B for baixo, a referência provavelmente é de OUTRA pessoa -> sessão INVÁLIDA "
              f"(referência corrompida, não falha do ataque).")
        P("")
    if flagged:
        P(f"Identidades com A-B < {AB_BAIXO:.2f}:")
        for r in flagged:
            causa = ("referência multi-rosto (detecção pegou o rosto errado)" if r["multi"]
                     else "variação intraclasse alta no próprio LFW (idade/pose/luz)")
            P(f"  - {r['id']}: A-B={fmt(r['ab'])}, ataque={fmt(r['atk'])} -> {causa}.")
    else:
        P(f"Nenhuma identidade com A-B < {AB_BAIXO:.2f}.")
    P("")
    P("Como reportar cada caso:")
    P("  (a) A-B baixo por referência multi-rosto -> sessão INVÁLIDA (referência não é a vítima).")
    P("  (b) A-B baixo por variação intraclasse do LFW -> falha explicável pelo dataset, não pelo ataque.")
    P("  (c) A-B alto e ataque abaixo do limiar -> falha REAL do ataque (vale reportar: a")
    P("      transferência de identidade tem limites).")

    path = os.path.join(RES, "controle_intraclasse.txt")
    open(path, "w").write("\n".join(out))
    P(f"\nsalvo: {path}")


if __name__ == "__main__":
    main()
