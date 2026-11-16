#!/usr/bin/env python3
"""
Blindagem da referência — LFW e YTF (jeito 2)
=============================================
Problema: no ataque, a referência de verificação era a MESMA imagem/vídeo usada
como fonte do swap -> inflava a similaridade (self-match). Aqui a referência passa
a ser SEMPRE uma imagem/vídeo DIFERENTE da mesma pessoa:
  - LFW: outra FOTO do source (!= foto usada no swap)
  - YTF: outro VÍDEO do source (!= vídeo usado no swap)

Só muda o lado da VERIFICAÇÃO. Os swaps (no disco) e o baseline (que já usa
imagens/vídeos diferentes por par) NÃO mudam -> thresholds/EER idênticos.
Reusa os swaps do disco (não carrega FaceFusion). Reporta ASR antes (self) vs
depois (blindado).
"""
import os, sys, csv, glob, time
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
os.chdir(REPO)

from baseline_lfw import LFW_DIR
from run_swaps_lfw import get_lfw_identities, generate_swap_pairs, NUM_SWAPS
from baseline_ytf import get_ytf_identities, YTF_DATA_DIR, FRAMES_TO_SAMPLE
from run_swaps_ytf import generate_attack_pairs, NUM_ATTACKS
from src import AdaFaceVerifier
from src.utils import get_similarity
from sklearn.metrics import roc_curve

CFG = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")
RESULTS = os.path.join(REPO, "app", "results")
LFW_SWAPS = os.path.join(REPO, "data", "lfw", "swaps")
YTF_SWAPS = os.path.join(REPO, "data", "YouTubeFaces", "swaps")

verifier = AdaFaceVerifier(CFG)
print(f"device: {verifier.device}")

def emb_img(img):
    e = verifier.get_embedding(img)          # jeito 2
    return None if e is None else e.cpu().numpy()

_cache = {}
def emb_path(p):
    if p not in _cache:
        _cache[p] = emb_img(cv2.imread(p))
    return _cache[p]

def sample_frames(frames, n=FRAMES_TO_SAMPLE):
    if len(frames) > n:
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in idx]
    return frames

def mean_emb(embs):
    xs = [e for e in embs if e is not None]
    if not xs: return None
    m = np.mean(xs, axis=0); return m / (np.linalg.norm(m) + 1e-8)

def thresholds_from_csv(csv_path):
    labels, scores = [], []
    for r in csv.DictReader(open(csv_path)):
        labels.append(int(r["label"])); scores.append(float(r["score_j2"]))
    labels = np.array(labels); scores = np.array(scores)
    fpr, tpr, thr = roc_curve(labels, scores); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    def far(t):
        ok = np.where(fpr <= t)[0]; return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return {"0.45": 0.45, "0.60": 0.60, "EER": float(thr[i]),
            "FAR1%": far(0.01), "FAR0.1%": far(0.001)}, float((fpr[i]+fnr[i])/2)*100

def asr(scores, t):
    s = np.array([x for x in scores if x is not None]); return float(np.mean(s >= t))*100 if len(s) else float("nan")

def report(name, thr, eer, new_scores, old_scores, n_total, n_skip):
    print(f"\n{'='*60}\n{name} — BLINDADO (referência separada)\n{'='*60}")
    print(f"  ataques usados: {len(new_scores)} | pulados (sem 2a img/vídeo): {n_skip} | EER baseline: {eer:.2f}%")
    print(f"  ataque média: self={np.mean([s for s in old_scores if s is not None]):.3f}  "
          f"blindado={np.mean(new_scores):.3f}")
    print(f"  {'threshold':<10}{'valor':>9}{'ASR self':>11}{'ASR blindado':>14}")
    for k in ["0.45", "0.60", "EER", "FAR1%", "FAR0.1%"]:
        print(f"  {k:<10}{thr[k]:>9.4f}{asr(old_scores, thr[k]):>10.1f}%{asr(new_scores, thr[k]):>13.1f}%")
    return {k: asr(new_scores, thr[k]) for k in thr}


# ---- identificação por par (permite comparar self x blindado nos MESMOS pares) ----
def src_names(csv_name):
    return [r["source_name"] for r in csv.DictReader(open(os.path.join(RESULTS, csv_name)))]

def save_pairs(path, header, meta):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header)
        for m in meta:
            if m[-2] is not None:
                w.writerow(list(m[:-2]) + [f"{m[-2]:.6f}", "" if m[-1] is None else f"{m[-1]:.6f}"])

def same_pairs_report(name, thr, meta):
    pares = [(m[-2], m[-1]) for m in meta if m[-2] is not None and m[-1] is not None]
    blind = [b for b, _ in pares]; self_ = [s for _, s in pares]
    linhas = [f"{name} — self x blindado nos MESMOS {len(pares)} pares",
              f"  similaridade média: self={np.mean(self_):.3f}  blindado={np.mean(blind):.3f}",
              f"  {'threshold':<10}{'valor':>9}{'ASR self':>11}{'ASR blindado':>14}"]
    for k in ["0.45", "0.60", "EER", "FAR1%", "FAR0.1%"]:
        linhas.append(f"  {k:<10}{thr[k]:>9.4f}{asr(self_, thr[k]):>10.1f}%{asr(blind, thr[k]):>13.1f}%")
    print("\n" + "\n".join(linhas))
    return linhas

# ============================================================ LFW
print("\n[LFW] re-verificando 3000 ataques com referência separada...")
lfw_thr, lfw_eer = thresholds_from_csv(os.path.join(RESULTS, "lfw_scores_jeito12.csv"))
lfw_ids = get_lfw_identities(LFW_DIR)
lfw_pairs = generate_swap_pairs(lfw_ids, NUM_SWAPS)
# scores "self" antigos (coluna similarity_j2 do CSV de ataque)
old_lfw = [float(r["similarity_j2"]) if r["similarity_j2"] else None
           for r in csv.DictReader(open(os.path.join(RESULTS, "lfw_swap_attack_jeito12.csv")))]
old_lfw_src = src_names("lfw_swap_attack_jeito12.csv")
lfw_meta, lfw_desalinhados = [], 0
new_lfw, skip_lfw = [], 0
t0 = time.time()
for i, (src_name, src_path, tgt_name, tgt_path) in enumerate(lfw_pairs):
    sb = os.path.basename(src_path).split('.')[0]; tb = os.path.basename(tgt_path).split('.')[0]
    swap_file = os.path.join(LFW_SWAPS, f"swap_{sb}_in_{tb}.jpg")
    # referência separada: outra foto do MESMO source
    others = [p for p in lfw_ids[src_name] if p != src_path]
    if not others or not os.path.exists(swap_file):
        skip_lfw += 1; continue
    e_swap = emb_img(cv2.imread(swap_file)); e_ref = emb_path(others[0])
    new_lfw.append(get_similarity(e_swap, e_ref) if (e_swap is not None and e_ref is not None) else None)
    if i >= len(old_lfw_src) or old_lfw_src[i] != src_name:
        lfw_desalinhados += 1
    lfw_meta.append((i, src_name, os.path.basename(src_path), tgt_name, os.path.basename(tgt_path),
                     new_lfw[-1], old_lfw[i] if (i < len(old_lfw) and lfw_desalinhados == 0) else None))
    if (i+1) % 1000 == 0: print(f"   LFW {i+1}/{len(lfw_pairs)} ({time.time()-t0:.0f}s)")
new_lfw = [s for s in new_lfw if s is not None]
np.savetxt(os.path.join(RESULTS, "lfw_attack_blindado_scores.csv"),
           np.array(new_lfw), delimiter=",", header="similarity_blindado", comments="")
lfw_blind = report("LFW (foto)", lfw_thr, lfw_eer, new_lfw, old_lfw, len(lfw_pairs), skip_lfw)
if lfw_desalinhados: print(f"  AVISO: {lfw_desalinhados} pares LFW desalinhados com lfw_swap_attack_jeito12.csv -> self por par omitido")
save_pairs(os.path.join(RESULTS, "lfw_attack_blindado_pares.csv"),
           ["pair_idx", "source_name", "source_img", "target_name", "target_img", "similarity_blindado", "similarity_self"], lfw_meta)
lfw_same = same_pairs_report("LFW (foto)", lfw_thr, lfw_meta)

# ============================================================ YTF
print("\n[YTF] re-verificando ataques com vídeo de referência diferente...")
ytf_thr, ytf_eer = thresholds_from_csv(os.path.join(RESULTS, "ytf_scores_jeito12.csv"))
ytf_ids = get_ytf_identities(YTF_DATA_DIR)
ytf_pairs = generate_attack_pairs(ytf_ids, NUM_ATTACKS)
old_ytf = [float(r["similarity_j2"]) if r["similarity_j2"] else None
           for r in csv.DictReader(open(os.path.join(RESULTS, "ytf_swap_attack_jeito12.csv")))]
old_ytf_src = src_names("ytf_swap_attack_jeito12.csv")
ytf_meta, ytf_desalinhados = [], 0
new_ytf, skip_ytf = [], 0
t0 = time.time()
for i, (src_name, src_vid, tgt_name, tgt_vid) in enumerate(ytf_pairs):
    other_vids = [v for v in ytf_ids[src_name].keys() if v != src_vid]
    swaps = sorted(glob.glob(os.path.join(YTF_SWAPS, f"swap_{src_name}_{src_vid}_in_{tgt_name}_{tgt_vid}_f*.jpg")))
    if not other_vids or not swaps:
        skip_ytf += 1; continue
    swap_mean = mean_emb([emb_img(cv2.imread(s)) for s in swaps])
    ref_mean = mean_emb([emb_path(p) for p in sample_frames(ytf_ids[src_name][other_vids[0]])])
    new_ytf.append(get_similarity(swap_mean, ref_mean) if (swap_mean is not None and ref_mean is not None) else None)
    if i >= len(old_ytf_src) or old_ytf_src[i] != src_name:
        ytf_desalinhados += 1
    ytf_meta.append((i, src_name, src_vid, tgt_name, tgt_vid, other_vids[0],
                     new_ytf[-1], old_ytf[i] if (i < len(old_ytf) and ytf_desalinhados == 0) else None))
    if (i+1) % 200 == 0: print(f"   YTF {i+1}/{len(ytf_pairs)} ({time.time()-t0:.0f}s)")
new_ytf = [s for s in new_ytf if s is not None]
np.savetxt(os.path.join(RESULTS, "ytf_attack_blindado_scores.csv"),
           np.array(new_ytf), delimiter=",", header="similarity_blindado", comments="")
ytf_blind = report("YTF (vídeo)", ytf_thr, ytf_eer, new_ytf, old_ytf, len(ytf_pairs), skip_ytf)
if ytf_desalinhados: print(f"  AVISO: {ytf_desalinhados} pares YTF desalinhados com ytf_swap_attack_jeito12.csv -> self por par omitido")
save_pairs(os.path.join(RESULTS, "ytf_attack_blindado_pares.csv"),
           ["pair_idx", "source_name", "source_vid", "target_name", "target_vid", "ref_vid", "similarity_blindado", "similarity_self"], ytf_meta)
ytf_same = same_pairs_report("YTF (vídeo)", ytf_thr, ytf_meta)

# salva
with open(os.path.join(RESULTS, "blindado_lfw_ytf.txt"), "w") as f:
    f.write("LFW blindado ASR: " + str({k: round(v,1) for k,v in lfw_blind.items()}) + "\n")
    f.write("YTF blindado ASR: " + str({k: round(v,1) for k,v in ytf_blind.items()}) + "\n")
print(f"\nsalvo: {os.path.join(RESULTS, 'blindado_lfw_ytf.txt')}")
with open(os.path.join(RESULTS, "blindado_mesmos_pares.txt"), "w") as f:
    f.write("\n".join(lfw_same + [""] + ytf_same) + "\n")
print(f"salvo: {os.path.join(RESULTS, 'blindado_mesmos_pares.txt')}")
