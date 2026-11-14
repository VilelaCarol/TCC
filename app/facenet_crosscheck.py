#!/usr/bin/env python3
"""
Segundo verificador: FaceNet (InceptionResnetV1 / VGGFace2) — parecer #9.
========================================================================
Ameaça à validade: o inswapper condiciona o swap num embedding da família ArcFace,
e o AdaFace tem espaço de identidade próximo. Este script repete a verificação do
LFW com FaceNet (outra arquitetura e outro treino) para ver se o ataque transfere.

CORREÇÃO (pós-parecer): usa o PIPELINE NATIVO do FaceNet — detecção+alinhamento
pelo MTCNN do facenet-pytorch para 160x160 RGB com a normalização própria dele
(post_process). Antes usava o recorte ArcFace 112->160 (alinhamento do AdaFace),
fora da especificação do FaceNet, o que inflava o EER.

Calibra o limiar por FAR no baseline do LFW e mede o ASR do ataque BLINDADO
(referência = outra foto da vítima). Imagens onde o MTCNN não detecta são puladas.
"""
import os, sys, csv
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import roc_curve

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app")); sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
os.chdir(REPO)
from baseline_lfw import LFW_DIR, PAIRS_FILE, parse_pairs
from run_swaps_lfw import get_lfw_identities, generate_swap_pairs, NUM_SWAPS

RES = os.path.join(REPO, "app", "results")
LFW_SWAPS = os.path.join(REPO, "data", "lfw", "swaps")
dev = "cuda" if torch.cuda.is_available() else "cpu"

# pipeline nativo FaceNet: MTCNN (detecção+alinhamento->160x160, prewhiten) + InceptionResnetV1
mtcnn = MTCNN(image_size=160, margin=14, post_process=True, selection_method="probability", device=dev)
fnet = InceptionResnetV1(pretrained="vggface2").eval().to(dev)
print(f"device: {dev} | FaceNet VGGFace2 + MTCNN (pipeline nativo)")

@torch.no_grad()
def fembed(img_bgr):
    if img_bgr is None: return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)       # MTCNN espera RGB
    face = mtcnn(rgb)                                    # (3,160,160) já padronizado, ou None
    if face is None: return None
    e = fnet(face.unsqueeze(0).to(dev))[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8)

def cos(a, b): return float(np.dot(a, b)) if (a is not None and b is not None) else None

_c = {}
def fembed_path(p):
    if p not in _c: _c[p] = fembed(cv2.imread(p))
    return _c[p]

# ---- baseline (calibração FaceNet) ----
print("[1/3] baseline FaceNet/MTCNN (6000 pares)...")
pairs = parse_pairs(PAIRS_FILE, LFW_DIR)
lab, sc = [], []
miss = 0
for i, (p1, p2, same) in enumerate(pairs):
    a, b = fembed_path(p1), fembed_path(p2)
    s = cos(a, b)
    if s is None:
        miss += 1; continue
    lab.append(1 if same else 0); sc.append(s)
    if (i + 1) % 1500 == 0: print(f"   {i+1}/{len(pairs)} (sem detecção: {miss})")
lab, sc = np.array(lab), np.array(sc)
fpr, tpr, thr = roc_curve(lab, sc); fnr = 1 - tpr
ie = int(np.nanargmin(np.abs(fpr - fnr)))
def far(t):
    ok = np.where(fpr <= t)[0]; return float(thr[ok[-1]]) if len(ok) else float(thr[0])
TH = {"EER": float(thr[ie]), "FAR1%": far(0.01), "FAR0.1%": far(0.001)}
eer = float((fpr[ie] + fnr[ie]) / 2) * 100
gen, imp = sc[lab == 1], sc[lab == 0]

# ---- ataque blindado ----
print("[2/3] ataque blindado FaceNet/MTCNN (reusa swaps do disco)...")
ids = get_lfw_identities(LFW_DIR)
atk = []; miss_a = 0
for (sn, sp, tn, tp) in generate_swap_pairs(ids, NUM_SWAPS):
    sb = os.path.basename(sp).split('.')[0]; tb = os.path.basename(tp).split('.')[0]
    sf = os.path.join(LFW_SWAPS, f"swap_{sb}_in_{tb}.jpg")
    src_ref = next((p for p in ids[sn] if p != sp), None)
    if not os.path.exists(sf) or src_ref is None: continue
    e_sw, e_ref = fembed(cv2.imread(sf)), fembed_path(src_ref)
    s = cos(e_sw, e_ref)
    if s is None:
        miss_a += 1; continue
    atk.append(s)
atk = np.array(atk)

print("[3/3] relatório...")
def asr(t): return 100.0 * np.mean(atk >= t)
out = []
def P(s=""): print(s); out.append(s)
P("=" * 72)
P("CROSS-CHECK FaceNet (VGGFace2 + MTCNN nativo) — ataque transfere p/ outra família?")
P("=" * 72)
P(f"baseline: genuína {gen.mean():.3f} | impostor {imp.mean():.3f} | EER {eer:.2f}% | n={len(sc)} (sem detecção: {miss})")
P(f"ataque blindado: n={len(atk)} | sim média {atk.mean():.3f} | mediana {np.median(atk):.3f} (sem detecção: {miss_a})")
P(f"  {'limiar':<10}{'valor':>9}{'ASR':>9}{'TPR genuína':>13}")
for k in ["EER", "FAR1%", "FAR0.1%"]:
    tpr_k = 100 * np.mean(gen >= TH[k])
    P(f"  {k:<10}{TH[k]:>9.4f}{asr(TH[k]):>8.1f}%{tpr_k:>12.1f}%")
P("")
P("Comparar com AdaFace (LFW blindado): FAR1% 91.8% | FAR0.1% 91.3%.")
P("Pré-processamento DENTRO da especificação do FaceNet (MTCNN 160x160 RGB + prewhiten).")
open(os.path.join(RES, "facenet_crosscheck.txt"), "w").write("\n".join(out))
P(f"\nsalvo: {os.path.join(RES, 'facenet_crosscheck.txt')}")
