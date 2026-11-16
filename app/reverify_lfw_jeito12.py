#!/usr/bin/env python3
"""
LFW unificado no JEITO 2 (+ jeito 1 para o comparativo antes/depois)
====================================================================
Regenera os swaps IN-PROCESS (mesma seed do experimento original) e verifica
baseline (genuíno/impostor) e ataque medindo a similaridade nos DOIS jeitos na
MESMA imagem — a única diferença é o recorte (cru+RGB vs alinhado+BGR), nunca o
swap. Depois computa ROC/EER/thresholds em FAR fixo e o ASR em todos os limiares.

Travas de integridade (defesa na banca):
  * MESMA geração de pares/seed do run_swaps_lfw original (reprodutível).
  * TODOS os pares entram; nenhuma seleção de "swaps bons".
  * Thresholds só por EER / FAR fixo / fixos da config. padrão (0.45, 0.60) — nunca escolhidos
    para maximizar ASR.
  * jeito 1 e jeito 2 medidos na mesma imagem -> before/after limpo.

Saídas em app/results/:
  lfw_scores_jeito12.csv, lfw_swap_attack_jeito12.csv,
  lfw_thresholds_jeito12.txt, lfw_jeito1_vs_jeito2.png
"""
import os, sys, csv, time
import numpy as np
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
os.chdir(REPO)

from baseline_lfw import ensure_lfw_images, parse_pairs, LFW_DIR, PAIRS_FILE, PAIRS_URL, download_file
from run_swaps_lfw import get_lfw_identities, generate_swap_pairs, NUM_SWAPS, RANDOM_SEED
from neural_face_swap import swap_pair
from src import AdaFaceVerifier
from src.utils import get_similarity
from sklearn.metrics import roc_curve

CFG = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")
RESULTS = os.path.join(REPO, "app", "results")
SWAPS_DIR = os.path.join(REPO, "data", "lfw", "swaps")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(SWAPS_DIR, exist_ok=True)

print("=" * 70)
print("LFW — reverificação jeito 1 (cru+RGB) vs jeito 2 (alinhado+BGR)")
print("=" * 70)

# ---------------------------------------------------------------- setup dados
ensure_lfw_images()
if not os.path.exists(PAIRS_FILE):
    download_file(PAIRS_URL, PAIRS_FILE)

verifier = AdaFaceVerifier(CFG)
print(f"device: {verifier.device}\n")

# cache de embeddings por caminho -> (emb_jeito1, emb_jeito2)
_emb_cache = {}

def embed_both_from_img(img):
    """Detecta UMA vez; devolve (emb_j1, emb_j2) reusando bbox/landmarks."""
    if img is None:
        return None, None
    bbox, lmk = verifier.detect(img)
    if bbox is None and lmk is None:
        return None, None
    e1 = verifier.get_embedding(img, bbox=bbox, align=False)          # cru + RGB
    e2 = verifier.get_embedding(img, bbox=bbox, landmarks=lmk, align=True)  # alinhado + BGR
    return e1, e2

def embed_both_path(path):
    if path not in _emb_cache:
        _emb_cache[path] = embed_both_from_img(cv2.imread(path))
    return _emb_cache[path]

# ======================================================= 1) BASELINE (6000)
print("[1/4] Baseline genuíno/impostor (6000 pares, 2 jeitos)...")
pairs = parse_pairs(PAIRS_FILE, LFW_DIR)
base_rows = []   # (label, s1, s2)
t0 = time.time()
miss = 0
for k, (p1, p2, same) in enumerate(pairs):
    if not (os.path.exists(p1) and os.path.exists(p2)):
        miss += 1; continue
    a1, a2 = embed_both_path(p1)
    b1, b2 = embed_both_path(p2)
    if a1 is None or b1 is None:
        miss += 1; continue
    s1 = get_similarity(a1, b1)
    s2 = get_similarity(a2, b2)
    base_rows.append((1 if same else 0, s1, s2))
    if (k + 1) % 1000 == 0:
        print(f"   {k+1}/{len(pairs)}  ({time.time()-t0:.0f}s)")
print(f"   baseline ok: {len(base_rows)} pares, {miss} pulados, {time.time()-t0:.0f}s\n")

with open(os.path.join(RESULTS, "lfw_scores_jeito12.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id", "label", "score_j1", "score_j2"])
    for i, (lab, s1, s2) in enumerate(base_rows):
        w.writerow([i, lab, f"{s1:.6f}", f"{s2:.6f}"])

# ======================================================= 2) ATAQUE (3000 swaps)
print(f"[2/4] Ataque: regenerando {NUM_SWAPS} swaps (in-process) + verificando 2 jeitos...")
identities = get_lfw_identities(LFW_DIR)
swap_pairs = generate_swap_pairs(identities, NUM_SWAPS)   # seed=RANDOM_SEED
print(f"   identidades: {len(identities)} | pares de ataque: {len(swap_pairs)} (seed={RANDOM_SEED})")

atk_rows = []   # (source_name, target_name, swap_file, s1, s2)
t0 = time.time(); n_swap_fail = 0; n_regen = 0
for i, (sn, sp, tn, tp) in enumerate(swap_pairs):
    sb = os.path.basename(sp).split('.')[0]
    tb = os.path.basename(tp).split('.')[0]
    out_path = os.path.join(SWAPS_DIR, f"swap_{sb}_in_{tb}.jpg")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        out = cv2.imread(out_path)
    else:
        src = cv2.imread(sp); tgt = cv2.imread(tp)
        out = swap_pair(src, tgt) if (src is not None and tgt is not None) else None
        if out is not None:
            cv2.imwrite(out_path, out); n_regen += 1
    if out is None:
        n_swap_fail += 1
        atk_rows.append((sn, tn, os.path.basename(out_path), None, None))
        continue

    # ataque: o swap se passa pela pessoa-fonte (SN)? -> comparar com a foto-fonte
    o1, o2 = embed_both_from_img(out)
    r1, r2 = embed_both_path(sp)
    s1 = get_similarity(o1, r1) if (o1 is not None and r1 is not None) else None
    s2 = get_similarity(o2, r2) if (o2 is not None and r2 is not None) else None
    atk_rows.append((sn, tn, os.path.basename(out_path), s1, s2))
    if (i + 1) % 500 == 0:
        print(f"   {i+1}/{len(swap_pairs)}  ({time.time()-t0:.0f}s, {n_regen} novos)")
print(f"   ataque ok: {len(atk_rows)} | swaps sem face: {n_swap_fail} | "
      f"regenerados: {n_regen} | {time.time()-t0:.0f}s\n")

with open(os.path.join(RESULTS, "lfw_swap_attack_jeito12.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["swap_id", "source_name", "target_name", "swap_file",
                                   "similarity_j1", "similarity_j2"])
    for i, (sn, tn, sf, s1, s2) in enumerate(atk_rows):
        w.writerow([i, sn, tn, sf,
                    "" if s1 is None else f"{s1:.6f}",
                    "" if s2 is None else f"{s2:.6f}"])

# ======================================================= 3) THRESHOLDS + ASR
print("[3/4] ROC/EER/FAR e ASR...")

def thresholds(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores)
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer_thr = float(thr[i]); eer = float((fpr[i] + fnr[i]) / 2)
    def thr_at_far(target):
        ok = np.where(fpr <= target)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0])
    return {"EER": eer_thr, "FAR1%": thr_at_far(0.01), "FAR0.1%": thr_at_far(0.001),
            "eer_value": eer}

def asr(scores, t):
    s = np.array([x for x in scores if x is not None])
    return float(np.mean(s >= t)) if len(s) else float("nan")

summary = []
for jeito, col in [("jeito1 (cru+RGB)", 1), ("jeito2 (alinhado+BGR)", 2)]:
    labels = [r[0] for r in base_rows]
    bscores = [r[col] for r in base_rows]
    ascores = [r[3 if col == 1 else 4] for r in atk_rows]
    ascores = [x for x in ascores if x is not None]
    gen = [s for lab, s in zip(labels, bscores) if lab == 1]
    imp = [s for lab, s in zip(labels, bscores) if lab == 0]
    th = thresholds(labels, bscores)
    block = {
        "jeito": jeito,
        "genuina_media": float(np.mean(gen)), "impostor_media": float(np.mean(imp)),
        "ataque_media": float(np.mean(ascores)),
        "EER_value": th["eer_value"],
        "thr": {"0.45": 0.45, "0.60": 0.60, "EER": th["EER"],
                "FAR1%": th["FAR1%"], "FAR0.1%": th["FAR0.1%"]},
        "ASR": {name: asr(ascores, t) for name, t in
                {"0.45": 0.45, "0.60": 0.60, "EER": th["EER"],
                 "FAR1%": th["FAR1%"], "FAR0.1%": th["FAR0.1%"]}.items()},
    }
    summary.append(block)

# imprime e salva resumo
lines = []
def P(s=""):
    print(s); lines.append(s)

P("\n" + "=" * 70)
P("RESUMO LFW (todos os pares; thresholds por EER/FAR/fixos)")
P("=" * 70)
for b in summary:
    P(f"\n### {b['jeito']}")
    P(f"  genuína média : {b['genuina_media']:.3f}")
    P(f"  impostor média: {b['impostor_media']:.3f}")
    P(f"  ataque média  : {b['ataque_media']:.3f}")
    P(f"  EER           : {b['EER_value']*100:.2f}%")
    P(f"  {'threshold':<10}{'valor':>10}{'ASR':>10}")
    for name in ["0.45", "0.60", "EER", "FAR1%", "FAR0.1%"]:
        P(f"  {name:<10}{b['thr'][name]:>10.4f}{b['ASR'][name]*100:>9.1f}%")

with open(os.path.join(RESULTS, "lfw_thresholds_jeito12.txt"), "w") as f:
    f.write("\n".join(lines))

# ======================================================= 4) FIGURA antes/depois
print("\n[4/4] Figura antes/depois...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, (jeito, col) in zip(axes, [("Jeito 1 (recorte cru + RGB)", 1),
                                    ("Jeito 2 (alinhado 5-pts + BGR)", 2)]):
    labels = [r[0] for r in base_rows]
    bscores = [r[col] for r in base_rows]
    ascores = [r[3 if col == 1 else 4] for r in atk_rows if r[3 if col == 1 else 4] is not None]
    gen = [s for lab, s in zip(labels, bscores) if lab == 1]
    imp = [s for lab, s in zip(labels, bscores) if lab == 0]
    bins = np.linspace(-0.2, 1.0, 50)
    ax.hist(imp, bins=bins, alpha=0.55, label="impostor", color="#4C72B0", density=True)
    ax.hist(gen, bins=bins, alpha=0.55, label="genuíno", color="#55A868", density=True)
    ax.hist(ascores, bins=bins, alpha=0.55, label="ataque (deepfake)", color="#C44E52", density=True)
    ax.axvline(0.45, ls="--", c="gray", lw=1); ax.axvline(0.60, ls=":", c="black", lw=1)
    ax.set_title(jeito); ax.set_xlabel("similaridade cosseno"); ax.legend(fontsize=9)
axes[0].set_ylabel("densidade")
fig.suptitle("LFW: distribuições antes (jeito 1) vs depois (jeito 2) — mesmas imagens",
             fontsize=13)
fig.tight_layout()
fig_path = os.path.join(RESULTS, "lfw_jeito1_vs_jeito2.png")
fig.savefig(fig_path, dpi=130)
print(f"   figura salva: {fig_path}")
print("\n LFW concluído.")
