"""
Utilitários da API YOLOv8-Face + AdaFace.

Reconstruído para o real-time (6GB VRAM). Reproduz a interface que os scripts
do projeto importam:
    from src.utils import detector            # instância YOLO (ultralytics)
    from src.utils import preprocess_image, preprocess_batch, get_similarity

Pré-processamento idêntico ao que o paper usou (revelado pelas funções legadas
de baseline_lfw.py): recorte da bounding box -> resize 112x112 -> RGB -> [-1,1].
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Diretório raiz do pacote (yolo8face_adaface/)
_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YOLO_WEIGHTS = os.path.join(_PKG_DIR, "model_weights", "yolov8n-face.pt")

FACE_SIZE = (112, 112)

# Template de referência ArcFace/AdaFace para alinhamento (5 pontos, imagem 112x112):
# olho-esq, olho-dir, nariz, canto-boca-esq, canto-boca-dir.
ARCFACE_REF_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(img_bgr: np.ndarray, landmarks5) -> np.ndarray:
    """
    Alinha a face para 112x112 via transformação de similaridade dos 5 landmarks
    até o template ArcFace (mesmo alinhamento com que o AdaFace foi treinado).
    Retorna crop BGR 112x112, ou None se a estimativa falhar.
    """
    lmk = np.asarray(landmarks5, dtype=np.float32)
    if lmk.shape != (5, 2):
        return None
    M, _ = cv2.estimateAffinePartial2D(lmk, ARCFACE_REF_112, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, FACE_SIZE, borderValue=0.0)


def preprocess_aligned(aligned_bgr: np.ndarray) -> torch.Tensor:
    """
    Crop alinhado 112x112 (BGR) -> tensor (1,3,112,112) normalizado [-1,1].
    Mantém BGR (convenção do AdaFace oficial). Sem troca de canais.
    """
    face = aligned_bgr.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.transpose(face, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(face)).unsqueeze(0)

# ==============================================================================
# Detector YOLOv8-Face (singleton, carregado uma vez na importação)
# É a MESMA instância usada pelo AdaFaceVerifier e pelos scripts — evita
# carregar o YOLO duas vezes na VRAM.
# ==============================================================================
if not os.path.exists(_YOLO_WEIGHTS):
    raise FileNotFoundError(
        f"Pesos do YOLOv8-Face não encontrados: {_YOLO_WEIGHTS}\n"
        "Baixe de: https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
    )

detector = YOLO(_YOLO_WEIGHTS)


# ==============================================================================
# Pré-processamento
# ==============================================================================
def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR (crop de face, qualquer tamanho) -> tensor (1,3,112,112) normalizado [-1,1].
    Reproduz baseline_lfw.preprocess_face: resize 112, BGR->RGB, /255, (x-0.5)/0.5.
    """
    face = cv2.resize(img_bgr, FACE_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5                      # [-1, 1]
    face = np.transpose(face, (2, 0, 1))           # CHW
    return torch.from_numpy(face).unsqueeze(0)     # (1,3,112,112)


def preprocess_batch(imgs_bgr) -> torch.Tensor:
    """Lista de crops BGR -> tensor (N,3,112,112)."""
    return torch.cat([preprocess_image(im) for im in imgs_bgr], dim=0)


# ==============================================================================
# Similaridade
# ==============================================================================
def get_similarity(emb_a, emb_b) -> float:
    """Similaridade cosseno entre dois embeddings (numpy ou torch, 1-D)."""
    a = emb_a.detach().cpu().numpy() if isinstance(emb_a, torch.Tensor) else np.asarray(emb_a)
    b = emb_b.detach().cpu().numpy() if isinstance(emb_b, torch.Tensor) else np.asarray(emb_b)
    a = a.flatten(); b = b.flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
