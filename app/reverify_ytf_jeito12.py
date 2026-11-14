#!/usr/bin/env python3
"""
YTF unificado no JEITO 2 (+ jeito 1 para o comparativo antes/depois)
====================================================================
Igual ao LFW, mas no protocolo YTF (vídeo-a-vídeo, embedding = média de N frames).
Usa os FRAMES CRUS (frame_images_DB) e aplica o NOSSO alinhamento ArcFace de 5
pontos por frame (detect YOLO -> align), NÃO o aligned_images_DB do YTF. A única
diferença entre jeito 1 e jeito 2 é o recorte (cru+RGB vs alinhado+BGR); os swaps
são os mesmos.

Reprodutível: mesma geração de pares/seed (generate_pairs / generate_attack_pairs)
dos scripts originais. TODOS os pares entram. Thresholds só por EER/FAR/literatura.

Saídas em app/results/:
  ytf_scores_jeito12.csv, ytf_swap_attack_jeito12.csv,
  ytf_thresholds_jeito12.txt, ytf_jeito1_vs_jeito2.png
"""
import os, sys, csv, time
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
os.chdir(REPO)

from baseline_ytf import (get_ytf_identities, generate_pairs, YTF_DATA_DIR,
                          NUM_GENUINE_PAIRS, NUM_IMPOSTOR_PAIRS, FRAMES_TO_SAMPLE)
from run_swaps_ytf import generate_attack_pairs, NUM_ATTACKS
from neural_face_swap import swap_pair
from src import AdaFaceVerifier
from src.utils import get_similarity
from sklearn.metrics import roc_curve

CFG = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")
RESULTS = os.path.join(REPO, "app", "results")
SWAPS_DIR = os.path.join(REPO, "data", "YouTubeFaces", "swaps")
os.makedirs(RESULTS, exist_ok=True); os.makedirs(SWAPS_DIR, exist_ok=True)

print("=" * 70)
print("YTF — reverificação jeito 1 (cru+RGB) vs jeito 2 (alinhado+BGR)")
print("   (frames CRUS + nosso alinhamento; protocolo vídeo-a-vídeo)")
print("=" * 70)

verifier = AdaFaceVerifier(CFG)
print(f"device: {verifier.device}")

# cache de embeddings por caminho de frame -> (emb_j1, emb_j2)
_emb_cache = {}

def embed_both_from_img(img):
    if img is None:
        return None, None
    bbox, lmk = verifier.detect(img)
    if bbox is None and lmk is None:
        return None, None
    e1 = verifier.get_embedding(img, bbox=bbox, align=False)
    e2 = verifier.get_embedding(img, bbox=bbox, landmarks=lmk, align=True)
    return e1, e2

def embed_both_path(path):
    if path not in _emb_cache:
        _emb_cache[path] = embed_both_from_img(cv2.imread(path))
    return _emb_cache[path]

def sample_frames(frames, n=FRAMES_TO_SAMPLE):
    if len(frames) > n:
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return [frames[i] for i in idx]
    return frames

def video_means(frame_paths):
    """(mean_j1, mean_j2) normalizados, a partir de N frames detect+align/crop."""
    e1s, e2s = [], []
    for fp in sample_frames(frame_paths):
        e1, e2 = embed_both_path(fp)
        if e1 is not None: e1s.append(e1.detach().cpu().numpy() if hasattr(e1, "detach") else np.asarray(e1))
        if e2 is not None: e2s.append(e2.detach().cpu().numpy() if hasattr(e2, "detach") else np.asarray(e2))
    def _mean(xs):
        if not xs: return None
        m = np.mean(xs, axis=0); return m / (np.linalg.norm(m) + 1e-8)
    return _mean(e1s), _mean(e2s)

# -------------------------------------------------- dataset
print(f"\n[0/4] Lendo identidades de {YTF_DATA_DIR} ...")
identities = get_ytf_identities(YTF_DATA_DIR)
n_people = len(identities); n_videos = sum(len(v) for v in identities.values())
print(f"   pessoas: {n_people} | vídeos: {n_videos}")
if n_people == 0:
    print("❌ dataset vazio — verifique a extração do frame_images_DB."); sys.exit(1)

# ======================================================= 1) BASELINE
_SMOKE = int(os.environ.get("YTF_SMOKE", "0"))   # >0 limita pares p/ teste rápido
print(f"\n[1/4] Baseline vídeo-a-vídeo ({NUM_GENUINE_PAIRS}+{NUM_IMPOSTOR_PAIRS} pares, 2 jeitos)...")
pairs = generate_pairs(identities, NUM_GENUINE_PAIRS, NUM_IMPOSTOR_PAIRS)
if _SMOKE:
    # pega alguns genuínos e alguns impostores
    g = [p for p in pairs if p[4] == 1][:_SMOKE]
    im = [p for p in pairs if p[4] == 0][:_SMOKE]
    pairs = g + im
    print(f"   [SMOKE] limitado a {len(pairs)} pares")
base_rows = []; t0 = time.time(); err = 0
for i, (p1, v1, p2, v2, same) in enumerate(pairs):
    f1 = identities.get(p1, {}).get(v1, []); f2 = identities.get(p2, {}).get(v2, [])
    a1, a2 = video_means(f1); b1, b2 = video_means(f2)
    if a1 is None or b1 is None:
        err += 1; continue
    base_rows.append((1 if same else 0, float(np.dot(a1, b1)), float(np.dot(a2, b2))))
    if (i + 1) % 500 == 0:
        print(f"   {i+1}/{len(pairs)}  ({time.time()-t0:.0f}s, cache={len(_emb_cache)})")
print(f"   baseline ok: {len(base_rows)} | erros: {err} | {time.time()-t0:.0f}s")

with open(os.path.join(RESULTS, "ytf_scores_jeito12.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id", "label", "score_j1", "score_j2"])
    for i, (lab, s1, s2) in enumerate(base_rows):
        w.writerow([i, lab, f"{s1:.6f}", f"{s2:.6f}"])

# ======================================================= 2) ATAQUE
print(f"\n[2/4] Ataque: {NUM_ATTACKS} pares, swap em {FRAMES_TO_SAMPLE} frames/vídeo (2 jeitos)...")
attack_pairs = generate_attack_pairs(identities, NUM_ATTACKS)
if _SMOKE:
    attack_pairs = attack_pairs[:_SMOKE]
    print(f"   [SMOKE] limitado a {len(attack_pairs)} ataques")
atk_rows = []; t0 = time.time(); n_fail = 0; n_regen = 0
for i, (sn, sv, tn, tv) in enumerate(attack_pairs):
    src_frames = identities.get(sn, {}).get(sv, [])
    tgt_frames = identities.get(tn, {}).get(tv, [])
    if not src_frames or not tgt_frames:
        atk_rows.append((sn, tn, None, None)); continue
    src_ref = cv2.imread(src_frames[len(src_frames) // 2])   # frame do meio = fonte do swap
    if src_ref is None:
        atk_rows.append((sn, tn, None, None)); continue

    swap_e1, swap_e2 = [], []
    for j, tf in enumerate(sample_frames(tgt_frames)):
        out_name = f"swap_{sn}_{sv}_in_{tn}_{tv}_f{j}.jpg"
        out_path = os.path.join(SWAPS_DIR, out_name)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            out = cv2.imread(out_path)
        else:
            tgt = cv2.imread(tf)
            out = swap_pair(src_ref, tgt) if tgt is not None else None
            if out is not None:
                cv2.imwrite(out_path, out); n_regen += 1
        if out is None:
            continue
        o1, o2 = embed_both_from_img(out)
        if o1 is not None: swap_e1.append(o1.detach().cpu().numpy() if hasattr(o1, "detach") else np.asarray(o1))
        if o2 is not None: swap_e2.append(o2.detach().cpu().numpy() if hasattr(o2, "detach") else np.asarray(o2))

    src_m1, src_m2 = video_means(src_frames)
    def _m(xs):
        if not xs: return None
        m = np.mean(xs, axis=0); return m / (np.linalg.norm(m) + 1e-8)
    sw1, sw2 = _m(swap_e1), _m(swap_e2)
    s1 = float(np.dot(sw1, src_m1)) if (sw1 is not None and src_m1 is not None) else None
    s2 = float(np.dot(sw2, src_m2)) if (sw2 is not None and src_m2 is not None) else None
    if s2 is None and s1 is None:
        n_fail += 1
    atk_rows.append((sn, tn, s1, s2))
    if (i + 1) % 100 == 0:
        print(f"   {i+1}/{len(attack_pairs)}  ({time.time()-t0:.0f}s, {n_regen} swaps novos)")
print(f"   ataque ok: {len(atk_rows)} | sem score: {n_fail} | swaps novos: {n_regen} | {time.time()-t0:.0f}s")

with open(os.path.join(RESULTS, "ytf_swap_attack_jeito12.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["attack_id", "source_name", "target_name",
                                   "similarity_j1", "similarity_j2"])
    for i, (sn, tn, s1, s2) in enumerate(atk_rows):
        w.writerow([i, sn, tn, "" if s1 is None else f"{s1:.6f}",
                    "" if s2 is None else f"{s2:.6f}"])

# ======================================================= 3) THRESHOLDS + ASR
print("\n[3/4] ROC/EER/FAR e ASR...")
def thresholds(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores)
    fpr, tpr, thr = roc_curve(labels, scores); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    def thr_at_far(t):
        ok = np.where(fpr <= t)[0]; return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return {"EER": float(thr[i]), "FAR1%": thr_at_far(0.01), "FAR0.1%": thr_at_far(0.001),
            "eer_value": float((fpr[i] + fnr[i]) / 2)}

def asr(scores, t):
    s = np.array([x for x in scores if x is not None])
    return float(np.mean(s >= t)) if len(s) else float("nan")

lines = []
def P(s=""):
    print(s); lines.append(s)
P("\n" + "=" * 70); P("RESUMO YTF (todos os pares; thresholds por EER/FAR/literatura)"); P("=" * 70)
summary = []
for jeito, col in [("jeito1 (cru+RGB)", 1), ("jeito2 (alinhado+BGR)", 2)]:
    labels = [r[0] for r in base_rows]; bscores = [r[col] for r in base_rows]
    ascores = [r[2 if col == 1 else 3] for r in atk_rows if r[2 if col == 1 else 3] is not None]
    gen = [s for lab, s in zip(labels, bscores) if lab == 1]
    imp = [s for lab, s in zip(labels, bscores) if lab == 0]
    th = thresholds(labels, bscores)
    thr_map = {"0.45": 0.45, "0.60": 0.60, "EER": th["EER"], "FAR1%": th["FAR1%"], "FAR0.1%": th["FAR0.1%"]}
    P(f"\n### {jeito}")
    P(f"  genuína média : {np.mean(gen):.3f}")
    P(f"  impostor média: {np.mean(imp):.3f}")
    P(f"  ataque média  : {np.mean(ascores):.3f}")
    P(f"  EER           : {th['eer_value']*100:.2f}%")
    P(f"  {'threshold':<10}{'valor':>10}{'ASR':>10}")
    for name in ["0.45", "0.60", "EER", "FAR1%", "FAR0.1%"]:
        P(f"  {name:<10}{thr_map[name]:>10.4f}{asr(ascores, thr_map[name])*100:>9.1f}%")
    summary.append((jeito, col, gen, imp, ascores))

with open(os.path.join(RESULTS, "ytf_thresholds_jeito12.txt"), "w") as f:
    f.write("\n".join(lines))

# ======================================================= 4) FIGURA
print("\n[4/4] Figura antes/depois...")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, (jeito, col, gen, imp, ascores) in zip(axes, summary):
    bins = np.linspace(-0.2, 1.0, 50)
    ax.hist(imp, bins=bins, alpha=0.55, label="impostor", color="#4C72B0", density=True)
    ax.hist(gen, bins=bins, alpha=0.55, label="genuíno", color="#55A868", density=True)
    ax.hist(ascores, bins=bins, alpha=0.55, label="ataque (deepfake)", color="#C44E52", density=True)
    ax.axvline(0.45, ls="--", c="gray", lw=1); ax.axvline(0.60, ls=":", c="black", lw=1)
    ax.set_title(jeito); ax.set_xlabel("similaridade cosseno"); ax.legend(fontsize=9)
axes[0].set_ylabel("densidade")
fig.suptitle("YTF: distribuições antes (jeito 1) vs depois (jeito 2) — mesmos vídeos/frames", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "ytf_jeito1_vs_jeito2.png"), dpi=130)
print(f"   figura salva: {os.path.join(RESULTS, 'ytf_jeito1_vs_jeito2.png')}")
print("\n✅ YTF concluído.")
