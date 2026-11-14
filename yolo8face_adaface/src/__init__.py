"""
API YOLOv8-Face + AdaFace IR-50 (reconstruída para o real-time, 6GB VRAM).

Expõe `AdaFaceVerifier`, com a mesma interface usada pelos scripts do projeto:
    verifier = AdaFaceVerifier(config_path)
    verifier.device                 # torch.device
    verifier.model                  # backbone AdaFace (nn.Module)
    emb = verifier.get_embedding(img_bgr)      # tensor (512,) normalizado, ou None
    match, sim = verifier.verify(img1_bgr, img2_bgr)

Modelo: net.py (IR-50, mode='ir') + checkpoint oficial adaface_ir50_webface4m.ckpt.
Pré-processamento idêntico ao do paper (ver src/utils.py).
"""

import os
import yaml
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import net
from .utils import (
    detector, FACE_SIZE, get_similarity,
    align_face, preprocess_aligned, preprocess_image,
)

THRESHOLD_LOW = 0.45
THRESHOLD_HIGH = 0.60


class AdaFaceVerifier:
    """
    Verificador YOLOv8-Face + AdaFace IR-50.

    Dois "jeitos" de recortar o rosto, selecionáveis por flag (`align`):
      * jeito 2 (align=True, PADRÃO)  -> alinhamento por 5 landmarks ao template
        canônico ArcFace 112x112, mantendo BGR. É a "régua" do real-time e o
        alvo para onde LFW/YTF devem convergir.
      * jeito 1 (align=False)         -> recorte cru da bounding box + RGB. Reproduz
        exatamente o pré-processamento legado (detect_and_crop_face/preprocess_face)
        usado no baseline original. Mantido só para o comparativo antes/depois.

    A escolha é a MESMA mudança no lado da verificação; não altera o ataque/swap.
    """

    def __init__(self, config_path: str, align: bool = True):
        # yolo8face_adaface/  (config fica em yolo8face_adaface/configs/config.yaml)
        self._pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        ada_cfg = cfg.get("adaface", {})
        arch = ada_cfg.get("architecture", "ir_50")
        mode = ada_cfg.get("mode", "ir")
        ckpt_rel = ada_cfg.get("checkpoint", "../data/models/adaface_ir50_webface4m.ckpt")
        ckpt_path = ckpt_rel if os.path.isabs(ckpt_rel) else os.path.normpath(os.path.join(self._pkg_dir, ckpt_rel))

        # jeito padrão de recorte (pode ser sobrescrito por chamada em get_embedding)
        self.align = bool(cfg.get("align", align))

        want = cfg.get("device", "cuda")
        self.device = torch.device("cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu")

        # ── Carrega o backbone AdaFace ───────────────────────────────────────
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint AdaFace não encontrado: {ckpt_path}")
        self.model = net.build_model(arch, mode=mode)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        # checkpoint Lightning: chaves têm prefixo 'model.'
        model_state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
        if not model_state:                      # ckpt já sem prefixo
            model_state = state
        self.model.load_state_dict(model_state, strict=True)
        self.model.eval().to(self.device)

        # ── Detector YOLO (mesma instância singleton de src.utils) ───────────
        self.detector = detector

    # ------------------------------------------------------------------ #
    # Detecção                                                           #
    # ------------------------------------------------------------------ #
    def detect(self, img_bgr: np.ndarray):
        """Retorna (bbox, landmarks5) da face de maior confiança, ou (None, None).

        bbox = (x1,y1,x2,y2); landmarks5 = np.array (5,2) [olho-e, olho-d, nariz,
        boca-e, boca-d] — usados para o alinhamento ArcFace.
        """
        results = self.detector(img_bgr, verbose=False)[0]
        if len(results.boxes) == 0:
            return None, None
        box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        lmk = None
        if results.keypoints is not None and len(results.keypoints) > 0:
            lmk = results.keypoints.xy[0].cpu().numpy().astype(np.float32)
        return bbox, lmk

    def detect_bbox(self, img_bgr: np.ndarray):
        """Compatibilidade: retorna só a bbox."""
        bbox, _ = self.detect(img_bgr)
        return bbox

    # ------------------------------------------------------------------ #
    # Embedding                                                          #
    # ------------------------------------------------------------------ #
    def _bbox_crop(self, img_bgr: np.ndarray, bbox):
        """Recorte cru da bounding box (BGR), com clamp nas bordas. None se degenerado."""
        h, w = img_bgr.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1)); x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1)); y2 = max(0, min(int(y2), h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return img_bgr[y1:y2, x1:x2]

    @torch.no_grad()
    def get_embedding(self, img_bgr: np.ndarray, bbox=None, landmarks=None, align=None):
        """
        Extrai o embedding AdaFace (512-d, L2-normalizado) de uma face no frame.

        `align` escolhe o jeito (None = usa o padrão self.align):
          * True  (jeito 2): alinhamento por 5 landmarks -> template ArcFace 112x112,
            mantendo BGR. Prioridade: landmarks dados -> detecta -> (fallback bbox cru).
          * False (jeito 1): recorte cru da bbox + RGB, idêntico ao baseline legado.

        Reaproveita landmarks/bbox já calculados (ex.: pelo FaceFusion) sem rodar o
        YOLO de novo. Retorna None se não houver face.
        """
        use_align = self.align if align is None else bool(align)
        face_crop = None            # crop BGR 112x112 (jeito 2) ou região crua (jeito 1)
        is_aligned = False

        if use_align:
            # ---- jeito 2: alinhamento por 5 landmarks ----
            lmk = landmarks
            if lmk is None and bbox is None:
                _, lmk = self.detect(img_bgr)
            if lmk is not None:
                aligned = align_face(img_bgr, lmk)
                if aligned is not None:
                    face_crop, is_aligned = aligned, True
            if face_crop is None and bbox is not None:
                # sem landmarks utilizáveis -> cai no recorte cru da bbox (último recurso)
                face_crop = self._bbox_crop(img_bgr, bbox)
        else:
            # ---- jeito 1: recorte cru da bbox + RGB (legado) ----
            bb = bbox
            if bb is None:
                bb, _ = self.detect(img_bgr)
            if bb is not None:
                face_crop = self._bbox_crop(img_bgr, bb)

        if face_crop is None:
            return None

        # jeito 2 alinhado -> preprocess_aligned (BGR); senão -> preprocess_image (crop + RGB)
        if is_aligned:
            tensor = preprocess_aligned(face_crop).to(self.device)
        else:
            tensor = preprocess_image(face_crop).to(self.device)

        emb, _ = self.model(tensor)
        emb = F.normalize(emb, p=2, dim=1)
        return emb[0]                              # (512,)

    # ------------------------------------------------------------------ #
    # Verificação                                                        #
    # ------------------------------------------------------------------ #
    def verify(self, img1_bgr, img2_bgr, threshold: float = THRESHOLD_LOW, align=None):
        """Retorna (match: bool, similaridade: float). `align` escolhe o jeito."""
        e1 = self.get_embedding(img1_bgr, align=align)
        e2 = self.get_embedding(img2_bgr, align=align)
        if e1 is None or e2 is None:
            return False, 0.0
        sim = get_similarity(e1, e2)
        return sim >= threshold, sim


__all__ = ["AdaFaceVerifier"]
