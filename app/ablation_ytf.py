#!/usr/bin/env python3
"""
Ablação YTF 2x2 — isolar ALINHAMENTO vs COR (RGB/BGR)
=====================================================
Mesmos pares/frames nas 4 condições (a única diferença é o pré-processamento):
  A = cru + RGB     (= "jeito 1" atual, e também o legado do paper original)
  B = cru + BGR
  C = alinhado + RGB
  D = alinhado + BGR (= "jeito 2" atual)

Detecta UMA vez por frame (YOLO) e gera os 4 embeddings a partir do mesmo
recorte/ alinhamento, trocando só a ordem de cor -> isola cada efeito.
Reaproveita os swaps já gerados em data/YouTubeFaces/swaps/ (não re-swapa nem
carrega FaceFusion se os arquivos existirem).

Veredito: a COR correta é a de MENOR EER nos pares reais. Se for RGB, o jeito 2
(BGR) está errado e o ASR alto é suspeito -> isso é dito explicitamente.
"""
import os, sys, csv, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
os.chdir(REPO)

from baseline_ytf import (get_ytf_identities, generate_pairs, YTF_DATA_DIR,
                          NUM_GENUINE_PAIRS, NUM_IMPOSTOR_PAIRS, FRAMES_TO_SAMPLE)
from run_swaps_ytf import generate_attack_pairs, NUM_ATTACKS
from src import AdaFaceVerifier
from src.utils import align_face, FACE_SIZE
from sklearn.metrics import roc_curve

CFG = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")
RESULTS = os.path.join(REPO, "app", "results")
SWAPS_DIR = os.path.join(REPO, "data", "YouTubeFaces", "swaps")
SPOT = os.path.join(RESULTS, "ablation_spotcheck")
os.makedirs(SPOT, exist_ok=True)

N_BASE_EACH = int(os.environ.get("N_BASE_EACH", "500"))   # genuínos e impostores (cada)
N_ATTACK = int(os.environ.get("N_ATTACK", "300"))

COND = ["A_cru_RGB", "B_cru_BGR", "C_ali_RGB", "D_ali_BGR"]
COND_LABEL = {"A_cru_RGB": "A  cru+RGB (jeito1)", "B_cru_BGR": "B  cru+BGR",
              "C_ali_RGB": "C  alinhado+RGB", "D_ali_BGR": "D  alinhado+BGR (jeito2)"}

print("=" * 72)
print("ABLAÇÃO YTF 2x2 — alinhamento vs cor | mesmos pares nas 4 condições")
print("=" * 72)

verifier = AdaFaceVerifier(CFG)
model, device = verifier.model, verifier.device
print(f"device: {device} | N baseline: {N_BASE_EACH}+{N_BASE_EACH} | N ataque: {N_ATTACK}")

@torch.no_grad()
def embed(face112_bgr, color):
    if face112_bgr is None:
        return None
    f = face112_bgr.astype(np.float32) / 255.0
    if color == "RGB":
        f = f[:, :, ::-1]                      # BGR -> RGB
    f = (f - 0.5) / 0.5
    f = np.transpose(f, (2, 0, 1))
    t = torch.from_numpy(np.ascontiguousarray(f)).unsqueeze(0).to(device)
    e, _ = model(t)
    e = F.normalize(e, p=2, dim=1)
    return e[0].cpu().numpy()

def faces_112(img_bgr):
    """(crop_bgr_112, aligned_bgr_112) — detecta 1x; None quando não dá."""
    if img_bgr is None:
        return None, None
    bbox, lmk = verifier.detect(img_bgr)
    crop = None
    if bbox is not None:
        c = verifier._bbox_crop(img_bgr, bbox)
        if c is not None:
            crop = cv2.resize(c, FACE_SIZE)
    aligned = align_face(img_bgr, lmk) if lmk is not None else None
    return crop, aligned

def embs_from_img(img_bgr):
    crop, aligned = faces_112(img_bgr)
    return {"A_cru_RGB": embed(crop, "RGB"), "B_cru_BGR": embed(crop, "BGR"),
            "C_ali_RGB": embed(aligned, "RGB"), "D_ali_BGR": embed(aligned, "BGR")}

_cache = {}
def embs_path(path):
    if path not in _cache:
        _cache[path] = embs_from_img(cv2.imread(path))
    return _cache[path]

def sample_frames(frames, n=FRAMES_TO_SAMPLE):
    if len(frames) > n:
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in idx]
    return frames

def video_means(frame_paths=None, from_imgs=None):
    """Média normalizada por condição. from_imgs = lista de imagens (swaps) OU None."""
    if from_imgs is not None:
        items = [embs_from_img(im) for im in from_imgs]
    else:
        items = [embs_path(p) for p in sample_frames(frame_paths)]
    acc = {k: [] for k in COND}
    for dd in items:
        for k in COND:
            if dd[k] is not None:
                acc[k].append(dd[k])
    out = {}
    for k in COND:
        if acc[k]:
            m = np.mean(acc[k], axis=0); out[k] = m / (np.linalg.norm(m) + 1e-8)
        else:
            out[k] = None
    return out

# -------------------------------------------------- dataset + spot-check
identities = get_ytf_identities(YTF_DATA_DIR)
print(f"pessoas: {len(identities)} | vídeos: {sum(len(v) for v in identities.values())}")

print("\n[spot-check] salvando 4 recortes de detecção p/ inspeção visual...")
spot_done = 0
for person, vids in identities.items():
    for vid, frames in vids.items():
        fp = frames[len(frames) // 2]
        img = cv2.imread(fp)
        if img is None:
            continue
        bbox, lmk = verifier.detect(img)
        if bbox is None:
            continue
        vis = img.copy()
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if lmk is not None:
            for (x, y) in lmk.astype(int):
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(SPOT, f"{spot_done}_{person}_frame.jpg"), vis)
        al = align_face(img, lmk) if lmk is not None else None
        if al is not None:
            cv2.imwrite(os.path.join(SPOT, f"{spot_done}_{person}_aligned112.jpg"), al)
        spot_done += 1
        break
    if spot_done >= 4:
        break
print(f"   {spot_done} exemplos em {SPOT}")

# ======================================================= BASELINE (subset)
print(f"\n[1/3] Baseline — {N_BASE_EACH} genuínos + {N_BASE_EACH} impostores (4 condições)...")
allpairs = generate_pairs(identities, NUM_GENUINE_PAIRS, NUM_IMPOSTOR_PAIRS)
gen_pairs = [p for p in allpairs if p[4] == 1][:N_BASE_EACH]
imp_pairs = [p for p in allpairs if p[4] == 0][:N_BASE_EACH]
sub = gen_pairs + imp_pairs
base = {k: {"label": [], "score": []} for k in COND}
t0 = time.time()
for i, (p1, v1, p2, v2, same) in enumerate(sub):
    m1 = video_means(identities[p1][v1]); m2 = video_means(identities[p2][v2])
    for k in COND:
        if m1[k] is not None and m2[k] is not None:
            base[k]["label"].append(1 if same else 0)
            base[k]["score"].append(float(np.dot(m1[k], m2[k])))
    if (i + 1) % 200 == 0:
        print(f"   {i+1}/{len(sub)}  ({time.time()-t0:.0f}s, cache={len(_cache)})")
print(f"   baseline ok ({time.time()-t0:.0f}s)")

# ======================================================= ATAQUE (subset, reusa swaps)
print(f"\n[2/3] Ataque — {N_ATTACK} pares (reusa swaps do disco; 4 condições)...")
attack_pairs = generate_attack_pairs(identities, NUM_ATTACKS)[:N_ATTACK]
atk = {k: [] for k in COND}
_swapper = None   # carrega FaceFusion só se faltar algum swap
t0 = time.time(); n_missing = 0
for i, (sn, sv, tn, tv) in enumerate(attack_pairs):
    sframes = identities.get(sn, {}).get(sv, []); tframes = identities.get(tn, {}).get(tv, [])
    if not sframes or not tframes:
        continue
    sampled = sample_frames(tframes)
    swap_imgs = []
    for j, tf in enumerate(sampled):
        out_path = os.path.join(SWAPS_DIR, f"swap_{sn}_{sv}_in_{tn}_{tv}_f{j}.jpg")
        im = cv2.imread(out_path) if os.path.exists(out_path) else None
        if im is None:
            n_missing += 1
            if _swapper is None:
                from neural_face_swap import swap_pair as _sp; _swapper = _sp
            src_ref = cv2.imread(sframes[len(sframes) // 2]); tgt = cv2.imread(tf)
            im = _swapper(src_ref, tgt) if (src_ref is not None and tgt is not None) else None
        if im is not None:
            swap_imgs.append(im)
    if not swap_imgs:
        continue
    sw = video_means(None, from_imgs=swap_imgs)
    src_m = video_means(sframes)
    for k in COND:
        if sw[k] is not None and src_m[k] is not None:
            atk[k].append(float(np.dot(sw[k], src_m[k])))
    if (i + 1) % 50 == 0:
        print(f"   {i+1}/{len(attack_pairs)}  ({time.time()-t0:.0f}s)")
print(f"   ataque ok ({time.time()-t0:.0f}s) | swaps faltando regenerados: {n_missing}")

# ======================================================= MÉTRICAS
def eer_of(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)

def asr(scores, t):
    s = np.asarray(scores); return float(np.mean(s >= t)) if len(s) else float("nan")

rows = {}
for k in COND:
    lab = np.asarray(base[k]["label"]); sc = np.asarray(base[k]["score"])
    gen = sc[lab == 1]; imp = sc[lab == 0]; a = atk[k]
    rows[k] = {"gen": float(gen.mean()), "imp": float(imp.mean()), "atk": float(np.mean(a)),
               "eer": eer_of(lab, sc) * 100, "asr45": asr(a, 0.45) * 100, "asr60": asr(a, 0.60) * 100,
               "n_base": len(lab), "n_atk": len(a)}

lines = []
def P(s=""):
    print(s); lines.append(s)

P("\n" + "=" * 92)
P(f"TABELA ABLAÇÃO YTF  (N baseline≈{rows['A_cru_RGB']['n_base']} pares, N ataque≈{rows['A_cru_RGB']['n_atk']})")
P("=" * 92)
P(f"{'condição':<26}{'genuína':>9}{'impostor':>10}{'ataque':>9}{'EER%':>8}{'ASR@.45':>9}{'ASR@.60':>9}")
for k in COND:
    r = rows[k]
    P(f"{COND_LABEL[k]:<26}{r['gen']:>9.3f}{r['imp']:>10.3f}{r['atk']:>9.3f}"
      f"{r['eer']:>8.2f}{r['asr45']:>8.1f}%{r['asr60']:>8.1f}%")

# decomposição
def d(a, b, f):  # efeito = cond_a - cond_b no campo f
    return rows[a][f] - rows[b][f]

P("\n--- DECOMPOSIÇÃO (em pontos percentuais de ASR@0.60 e em EER) ---")
P(f"ALINHAMENTO (cru->alinhado):")
P(f"   sob RGB (A->C): ASR@.60 {d('C_ali_RGB','A_cru_RGB','asr60'):+.1f} pp | EER {d('C_ali_RGB','A_cru_RGB','eer'):+.2f}")
P(f"   sob BGR (B->D): ASR@.60 {d('D_ali_BGR','B_cru_BGR','asr60'):+.1f} pp | EER {d('D_ali_BGR','B_cru_BGR','eer'):+.2f}")
P(f"COR (RGB->BGR):")
P(f"   no cru     (A->B): ASR@.60 {d('B_cru_BGR','A_cru_RGB','asr60'):+.1f} pp | EER {d('B_cru_BGR','A_cru_RGB','eer'):+.2f}")
P(f"   no alinhado(C->D): ASR@.60 {d('D_ali_BGR','C_ali_RGB','asr60'):+.1f} pp | EER {d('D_ali_BGR','C_ali_RGB','eer'):+.2f}")

best = min(COND, key=lambda k: rows[k]["eer"])
P(f"\n>>> MENOR EER (config 'correta' pelos pares reais): {COND_LABEL[best]}  (EER {rows[best]['eer']:.2f}%)")
color_best = "BGR" if rows["D_ali_BGR"]["eer"] <= rows["C_ali_RGB"]["eer"] else "RGB"
P(f">>> Cor com menor EER (no alinhado): {color_best}")
if best.endswith("BGR"):
    P(">>> VEREDITO: a melhor config usa BGR -> o jeito 2 (alinhado+BGR) é consistente.")
else:
    P(">>> ALERTA: a melhor config NÃO é BGR -> o jeito 2 pode estar com a cor errada; ASR alto é SUSPEITO.")

with open(os.path.join(RESULTS, "ablation_ytf.txt"), "w") as f:
    f.write("\n".join(lines))
print(f"\nresumo salvo: {os.path.join(RESULTS, 'ablation_ytf.txt')}")
