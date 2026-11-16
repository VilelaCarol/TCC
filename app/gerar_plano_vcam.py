#!/usr/bin/env python3
"""
Gera o plano de sessões do experimento de câmera virtual (PROTOCOLO_VCAM.md, 3.1).
==============================================================================
Regras (fixadas antes da coleta):
  1. A amostra original (semente 42, app/results/sessoes_plano.json) NÃO é alterada.
  2. Validade: similaridade genuína fotoA x fotoB >= 0.40, aplicada ANTES de gravar.
     O número de rostos por foto é diagnóstico, não invalida sozinho.
  3. Substituta: o sorteio original é CONTINUADO (16ª extração, depois 17ª...), não
     refeito. random.Random(42).sample(nomes, k) tem os mesmos 15 primeiros elementos
     para qualquer k >= 15, então a continuação é determinística.
     Antes de usá-la, o script VERIFICA que as 15 primeiras extrações reproduzem o
     sessoes_plano.json; se não reproduzirem, nenhuma substituta é adicionada.

Saídas:
  app/results/sessoes_plano_vcam.json   (plano das sessões a gravar)
  app/results/validade_vcam.txt         (log do critério, por identidade)

Uso:
  .venv/bin/python app/gerar_plano_vcam.py
"""
import os
import sys
import json
import random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
sys.path.insert(0, os.path.join(REPO, "app"))
os.chdir(REPO)

import cv2
from src import AdaFaceVerifier
from src.utils import get_similarity, detector
from baseline_lfw import LFW_DIR
from run_swaps_lfw import get_lfw_identities, MIN_IMAGES_PER_PERSON, RANDOM_SEED

CONFIG = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")
RES = os.path.join(REPO, "app", "results")
PLANO_ORIG = os.path.join(RES, "sessoes_plano.json")
PLANO_VCAM = os.path.join(RES, "sessoes_plano_vcam.json")
LOG = os.path.join(RES, "validade_vcam.txt")

AB_MIN = 0.40          # critério de validade (PROTOCOLO_VCAM.md, 3.1)
N_ALVO = 15            # tamanho do conjunto de sessões
MAX_EXTRA = 30         # até onde continuar o sorteio procurando substituta

out = []
def P(s=""):
    print(s)
    out.append(s)

verifier = AdaFaceVerifier(CONFIG)
P(f"device: {verifier.device}")

def emb_path(path):
    img = cv2.imread(path)
    if img is None:
        return None
    e = verifier.get_embedding(img)          # jeito 2 (padrão)
    return None if e is None else e.cpu().numpy()

def nfaces(path):
    img = cv2.imread(path)
    if img is None:
        return -1
    return int(len(detector(img, verbose=False)[0].boxes))

def avalia(nome, fotoA, fotoB):
    """Devolve (ab_sim, nA, nB, valido). ab_sim = None se faltar detecção."""
    ea, eb = emb_path(fotoA), emb_path(fotoB)
    nA, nB = nfaces(fotoA), nfaces(fotoB)
    if ea is None or eb is None:
        return None, nA, nB, False
    ab = get_similarity(ea, eb)
    return ab, nA, nB, (ab >= AB_MIN)

# ---------------------------------------------------------------- plano original
plano = json.load(open(PLANO_ORIG, encoding="utf-8"))
ids_orig = [x["id"] for x in plano]
P("=" * 78)
P("PLANO DE SESSÕES — EXPERIMENTO DE CÂMERA VIRTUAL")
P("=" * 78)
P(f"plano original: {PLANO_ORIG} ({len(plano)} identidades)")
P(f"critério de validade: similaridade genuína fotoA x fotoB >= {AB_MIN:.2f}")
P("")

validos, invalidos = [], []
P(f"{'identidade':<26}{'A-B':>8}{'rostos A/B':>12}   situação")
P("-" * 78)
for item in plano:
    ab, nA, nB, ok = avalia(item["id"], item["fotoA"], item["fotoB"])
    ab_txt = "  s/det" if ab is None else f"{ab:8.3f}"
    P(f"{item['id']:<26}{ab_txt}{f'{nA}/{nB}':>12}   {'VÁLIDA' if ok else 'INVÁLIDA (abaixo do critério)'}")
    (validos if ok else invalidos).append(dict(item, ab=ab, nfaces=[nA, nB],
                                               origem="plano original (semente 42)"))

P("")
P(f"válidas: {len(validos)} | inválidas: {len(invalidos)}"
  + (f" -> {', '.join(x['id'] for x in invalidos)}" if invalidos else ""))

# ---------------------------------------------------------------- substituta
faltam = N_ALVO - len(validos)
if faltam <= 0:
    P("\nNenhuma substituta necessária.")
else:
    P(f"\nFaltam {faltam} identidade(s). Continuando o sorteio original (semente {RANDOM_SEED}).")
    ids_all = get_lfw_identities(LFW_DIR, MIN_IMAGES_PER_PERSON)
    nomes = list(ids_all.keys())
    repro = random.Random(RANDOM_SEED).sample(nomes, len(plano))
    if repro != ids_orig:
        P("  AVISO: o sorteio não reproduz o sessoes_plano.json com esta listagem do LFW.")
        P("  (identidades esperadas: " + ", ".join(ids_orig[:3]) + " ...)")
        P("  (identidades obtidas:   " + ", ".join(repro[:3]) + " ...)")
        P("  Nenhuma substituta será adicionada: o conjunto fica com as válidas acima.")
        P("  Verifique se data/lfw/lfw_home/lfw_funneled/ é a mesma pasta usada no sorteio.")
    else:
        P(f"  sorteio reproduzido: as {len(plano)} primeiras extrações conferem com o plano original.")
        k = len(plano)
        while faltam > 0 and k < len(plano) + MAX_EXTRA:
            k += 1
            cand = random.Random(RANDOM_SEED).sample(nomes, k)[-1]
            if cand in ids_orig or any(v["id"] == cand for v in validos):
                P(f"  {k}ª extração: {cand} — já está no plano, pulando.")
                continue
            fotos = ids_all[cand]
            fotoA, fotoB = fotos[0], fotos[1]
            ab, nA, nB, ok = avalia(cand, fotoA, fotoB)
            ab_txt = "s/det" if ab is None else f"{ab:.3f}"
            P(f"  {k}ª extração: {cand:<26} A-B={ab_txt:<7} rostos {nA}/{nB} -> "
              + ("ACEITA como substituta" if ok else "reprovada no critério"))
            if ok:
                validos.append(dict(id=cand, fotoA=fotoA, fotoB=fotoB, ab=ab,
                                    nfaces=[nA, nB], origem=f"substituta ({k}ª extração do mesmo sorteio)"))
                faltam -= 1

# ---------------------------------------------------------------- saída
plano_vcam = []
for i, v in enumerate(validos, start=1):
    plano_vcam.append({
        "n": i, "id": v["id"], "fotoA": v["fotoA"], "fotoB": v["fotoB"],
        "ab_genuina": None if v["ab"] is None else round(float(v["ab"]), 4),
        "rostos_fotoA_fotoB": v["nfaces"],
        "origem": v["origem"],
    })
json.dump(plano_vcam, open(PLANO_VCAM, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

P("")
P("=" * 78)
P(f"PLANO FINAL — {len(plano_vcam)} sessões")
P("=" * 78)
for x in plano_vcam:
    P(f"  {x['n']:02d}  {x['id']:<26} A-B={x['ab_genuina']}  {x['origem']}")
P("")
P(f"salvo: {PLANO_VCAM}")
P("Commite este plano e o PROTOCOLO_VCAM.md ANTES de gravar a primeira sessão.")
open(LOG, "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"\nsalvo: {LOG}")
