#!/usr/bin/env python3
"""
Controles do lado do ATAQUE (revisão do protocolo item 8) — LFW, jeito 2.
========================================================
Para cada deepfake swap_{src}_in_{tgt} (rosto do SRC colado na foto do TGT),
mede a similaridade contra três referências DIFERENTES das usadas:
  (1) ref do SRC (vítima personificada)  -> o ATAQUE (deve ser ALTO)
  (2) ref do TGT (dono do quadro)        -> quanto da identidade ALVO sobra (deve ser BAIXO)
  (3) ref de identidade ALEATÓRIA        -> deve se comportar como IMPOSTOR (BAIXO)

Se (1) alto, (2) e (3) baixos: o swap TRANSFERE a identidade do SRC, REMOVE a do
TGT e não casa com pessoas quaisquer. Limiar de referência: FAR@1% do LFW (0.1562).
"""
import os, sys, csv, glob, random
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app")); sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
os.chdir(REPO)
from baseline_lfw import LFW_DIR
from run_swaps_lfw import get_lfw_identities, generate_swap_pairs, NUM_SWAPS
from src import AdaFaceVerifier
from src.utils import get_similarity

RES = os.path.join(REPO, "app", "results")
LFW_SWAPS = os.path.join(REPO, "data", "lfw", "swaps")
THR = 0.1562  # FAR@1% do LFW (ponto de operação)
rng = random.Random(42)

v = AdaFaceVerifier(os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml"))
print(f"device: {v.device}")
ids = get_lfw_identities(LFW_DIR)
names = list(ids.keys())
pairs = generate_swap_pairs(ids, NUM_SWAPS)

_c = {}
def emb_path(p):
    if p not in _c:
        e = v.get_embedding(cv2.imread(p)); _c[p] = None if e is None else e.cpu().numpy()
    return _c[p]
def emb_img(img):
    e = v.get_embedding(img); return None if e is None else e.cpu().numpy()

s_src, s_tgt, s_rnd = [], [], []
n_ok = 0
for (src_name, src_path, tgt_name, tgt_path) in pairs:
    sb = os.path.basename(src_path).split('.')[0]; tb = os.path.basename(tgt_path).split('.')[0]
    sf = os.path.join(LFW_SWAPS, f"swap_{sb}_in_{tb}.jpg")
    if not os.path.exists(sf):
        continue
    e_sw = emb_img(cv2.imread(sf))
    if e_sw is None:
        continue
    # refs DIFERENTES das usadas
    src_ref = next((p for p in ids[src_name] if p != src_path), None)
    tgt_ref = next((p for p in ids[tgt_name] if p != tgt_path), None)
    rnd_name = rng.choice([n for n in names if n not in (src_name, tgt_name)])
    rnd_ref = ids[rnd_name][0]
    if not (src_ref and tgt_ref and rnd_ref):
        continue
    a, t, r = emb_path(src_ref), emb_path(tgt_ref), emb_path(rnd_ref)
    if a is None or t is None or r is None:
        continue
    s_src.append(get_similarity(e_sw, a)); s_tgt.append(get_similarity(e_sw, t)); s_rnd.append(get_similarity(e_sw, r))
    n_ok += 1

def stats(x, label):
    x = np.array(x)
    return (f"{label:<34} n={len(x)} media={x.mean():6.3f} mediana={np.median(x):6.3f} "
            f"| >=FAR1%(0.156): {100*np.mean(x>=THR):5.1f}%")

out = []
def P(s=""):
    print(s); out.append(s)
P("=" * 78)
P("CONTROLES DO LADO DO ATAQUE — LFW jeito 2 (limiar de operação FAR@1% = 0.1562)")
P("=" * 78)
P(f"swaps avaliados: {n_ok}")
P(stats(s_src, "(1) deepfake vs ref do SRC (ataque)"))
P(stats(s_tgt, "(2) deepfake vs ref do TGT (sobra do alvo)"))
P(stats(s_rnd, "(3) deepfake vs ref ALEATÓRIA (impostor)"))
P("")
P("Interpretação: (1) ALTO e (2)/(3) BAIXOS => o swap transfere a identidade do SRC,")
P("remove a do alvo e não casa com identidades quaisquer (comporta-se como impostor).")
open(os.path.join(RES, "controles_ataque.txt"), "w").write("\n".join(out))
P(f"\nsalvo: {os.path.join(RES, 'controles_ataque.txt')}")
