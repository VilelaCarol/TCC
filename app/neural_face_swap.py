"""
Neural Face Swap — Integração Oficial FaceFusion (ajustado para 6GB VRAM)
=========================================================================
Usa o motor interno do FaceFusion (inswapper_128) para alinhar, processar e
mesclar a face.

Mudanças em relação à versão original (todas justificadas pelo hardware novo —
RTX 4050 Laptop 6GB — e pela reconstrução do ambiente):

  1. execution_providers: ['cuda','cpu']  (removido 'tensorrt': build de engine
     lento e consumo extra de VRAM, inviável em 6GB).
  2. execution_device_ids: [0]            (a máquina antiga tinha 2 GPUs: ['0','1'];
     aqui só existe a device 0, e o valor deve ser int).
  3. ort.preload_dlls()                   (faz o ONNXRuntime achar as libs CUDA/cuDNN
     que o PyTorch trouxe — substitui o hack de cuDNN do realtime_api_server.py).
  4. pre_check() dos módulos              (baixa/habilita os modelos; sem isso o
     get_inference_pool() devolve sessão None).
  5. swap_and_locate()                    (NOVO: devolve também a bbox que o FaceFusion
     já detectou, para reaproveitá-la na verificação AdaFace — evita re-detectar).
"""

import os
import sys
import cv2
import numpy as np

# Injeta facefusion no path
FF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "facefusion")
if FF_DIR not in sys.path:
    sys.path.insert(0, FF_DIR)

# Faz o ONNXRuntime localizar as libs CUDA/cuDNN empacotadas com o PyTorch.
# Precisa acontecer ANTES de qualquer sessão ONNX ser criada.
import onnxruntime as ort
try:
    ort.preload_dlls()
except Exception as _e:
    print(f"[neural_face_swap] aviso: ort.preload_dlls() falhou ({_e})")

from facefusion import state_manager
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.processors.modules.face_swapper.core import swap_face


# Config do FaceFusion ajustada para GPU única de 6GB
_FF_CONFIG = {
    'execution_providers': ['cuda', 'cpu'],
    'execution_device_ids': [0],
    'execution_thread_count': 1,
    'download_providers': ['github'],
    'face_detector_model': 'yolo_face',
    'face_detector_size': '640x640',
    'face_detector_score': 0.5,
    'face_detector_angles': [0],
    'face_detector_margin': (0, 0, 0, 0),
    'face_landmarker_model': '2dfan4',
    'face_landmarker_score': 0.5,
    'face_recognizer_model': 'arcface_w600k_r50',
    'face_swapper_model': 'inswapper_128',
    'face_swapper_pixel_boost': '128x128',
    'face_swapper_weight': 1.0,
    'face_mask_types': ['box'],
    'face_mask_blur': 0.3,
    'face_mask_padding': (0, 0, 0, 0),
    'face_mask_regions': ['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye',
                          'right-eye', 'eye-glasses', 'nose', 'mouth',
                          'upper-lip', 'lower-lip'],
}

_models_ready = False
_config_ready = False


def _ensure_config():
    """Aplica a config do FaceFusion no state_manager uma única vez (idempotente)."""
    global _config_ready
    if _config_ready:
        return
    for k, v in _FF_CONFIG.items():
        state_manager.init_item(k, v)
    _config_ready = True


def _ensure_models():
    """Baixa/habilita os modelos do FaceFusion uma única vez (pre_check)."""
    global _models_ready
    if _models_ready:
        return
    _ensure_config()
    from facefusion import face_detector, face_landmarker, face_recognizer, face_classifier
    from facefusion.processors.modules.face_swapper import core as _ff_swapper
    for mod in (face_detector, face_landmarker, face_recognizer, face_classifier, _ff_swapper):
        mod.pre_check()
    _models_ready = True


def swap_pair(source_img: np.ndarray, target_img: np.ndarray):
    """
    Swap in-process para um par (fonte -> alvo), sem subprocess.

    Os modelos do FaceFusion são carregados UMA vez (ficam na VRAM) e reusados a
    cada chamada — é isso que transforma o job de 'horas' (subprocess recarregando
    tudo por imagem) em 'minutos'. Substitui run_facefusion_swap() do LFW/YTF.

    Retorna o frame processado (BGR) ou None se não achar face na fonte ou no alvo.
    """
    _ensure_models()
    src_faces = get_many_faces([source_img])
    source_face = get_one_face(src_faces)
    if source_face is None:
        return None
    tgt_faces = get_many_faces([target_img])
    if not tgt_faces:
        return None
    return swap_face(source_face, tgt_faces[0], target_img)


class NeuralFaceSwapper:
    """Face swap neural usando a API interna do FaceFusion."""

    def __init__(self, source_image: np.ndarray, source_bbox: tuple = None):
        print("🧠 Inicializando motor FaceFusion (inswapper_128)...")
        _ensure_models()          # aplica config + carrega modelos (uma vez)

        # Face do atacante (fonte do swap)
        faces = get_many_faces([source_image])
        self.source_face = get_one_face(faces)
        if not self.source_face:
            print("⚠️ Face do atacante não encontrada. Tentando fallback (resize 640)...")
            resized = cv2.resize(source_image, (640, 640))
            self.source_face = get_one_face(get_many_faces([resized]))
        print(f"   ✅ Atacante processado: {'Sim' if self.source_face else 'Não'}")

    # ------------------------------------------------------------------ #
    def swap_and_locate(self, frame: np.ndarray):
        """
        Aplica o swap e devolve (frame_processado, bbox, landmarks5).

        bbox e landmarks5 vêm da face que o FaceFusion JÁ detectou internamente —
        o chamador os reaproveita no alinhamento/verificação AdaFace, sem rodar o
        YOLO de novo. Se não houver face, devolve (frame_original, None, None).
        """
        if self.source_face is None:
            return frame, None, None

        target_faces = get_many_faces([frame])
        if not target_faces:
            return frame, None, None

        target_face = target_faces[0]
        output_frame = swap_face(self.source_face, target_face, frame)

        bb = target_face.bounding_box.astype(int)
        bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        lmk = target_face.landmark_set.get('5')
        landmarks5 = None if lmk is None else np.asarray(lmk, dtype=np.float32)
        return output_frame, bbox, landmarks5

    # ------------------------------------------------------------------ #
    def swap(self, frame: np.ndarray, bbox: tuple = None) -> np.ndarray:
        """Compatibilidade com o código antigo: devolve só o frame processado."""
        output, _, _ = self.swap_and_locate(frame)
        return output
