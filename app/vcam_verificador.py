#!/usr/bin/env python3
"""
Processo B — sistema-alvo: lê a câmera virtual e verifica, sem saber do atacante.
==============================================================================
Abre /dev/video10 como abriria uma webcam, executa o pipeline completo do sistema
avaliado (YOLOv8-Face detecta e alinha -> AdaFace extrai -> similaridade cosseno
contra a referência cadastrada) e grava um CSV por quadro.

Este é o resultado OFICIAL do experimento de câmera virtual: o detector do
sistema-alvo vê o quadro manipulado, e o quadro atravessou o canal.

O CSV traz, por quadro, `detectado` e `n_rostos`, de modo que a regra principal
(mediana sobre os quadros detectados) e a análise de sensibilidade (quadro sem
detecção conta como rejeitado) saiam do MESMO arquivo, sem nova execução
(PROTOCOLO_VCAM.md, 3.3). Quadros repetidos pelo dispositivo são marcados em
`duplicado` e descartados na agregação (3.4).

O argumento --modo é apenas rótulo no log; não entra em nenhum cálculo.

Uso (normalmente chamado por app/rodar_sessoes_vcam.py):
  .venv/bin/python app/vcam_verificador.py --ref-image data/.../Id_0002.jpg \
      --duration 30 --log app/results/sessoes_vcam/atq_01_Id_B.csv
"""
import os
import sys
import csv
import time
import hashlib
import argparse

_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
if not _HAS_DISPLAY:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
sys.path.insert(0, os.path.join(REPO, "app"))
os.chdir(REPO)

import cv2
import numpy as np
from src import AdaFaceVerifier
from src.utils import get_similarity

CONFIG_PATH = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")

# limiares de operação herdados da LFW (jeito 2) — os mesmos do protocolo atual
THR_FAR1 = 0.1562
THR_FAR01 = 0.2289


def parse_args():
    p = argparse.ArgumentParser(description="Sistema-alvo: verifica o fluxo da câmera virtual")
    p.add_argument("--device", default="/dev/video10", help="dispositivo de entrada (câmera virtual)")
    p.add_argument("--ref-image", required=True, help="fotoB: referência cadastrada")
    p.add_argument("--modo", default="", help="rótulo do log (não entra em nenhum cálculo)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--duration", type=float, default=30.0, help="segundos, contados do 1º quadro")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--espera", type=float, default=20.0, help="segundos aguardando o 1º quadro")
    p.add_argument("--timeout", type=float, default=4.0, help="encerra após N s sem quadro novo")
    p.add_argument("--log", required=True)
    p.add_argument("--ocultar-similaridade", action="store_true",
                   help="não imprime a similaridade nem o nº de aceitos no console (§3.5). "
                        "Afeta só o print; o CSV continua gravando similarity/aceita_far1 igual.")
    return p.parse_args()


def detectar(verifier, frame):
    """Uma única passada do YOLOv8-Face: devolve (n_rostos, bbox, landmarks5).

    Mesma indexação de AdaFaceVerifier.detect() (face de maior confiança),
    mas preservando a contagem de rostos para o log.
    """
    res = verifier.detector(frame, verbose=False)[0]
    n = int(len(res.boxes))
    if n == 0:
        return 0, None, None
    box = res.boxes.xyxy[0].cpu().numpy().astype(int)
    bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    lmk = None
    if res.keypoints is not None and len(res.keypoints) > 0:
        lmk = res.keypoints.xy[0].cpu().numpy().astype(np.float32)
    return n, bbox, lmk


def main():
    a = parse_args()
    verifier = AdaFaceVerifier(CONFIG_PATH)
    ref_img = cv2.imread(a.ref_image)
    if ref_img is None:
        sys.exit(f"referência não encontrada: {a.ref_image}")
    e = verifier.get_embedding(ref_img)
    if e is None:
        sys.exit("nenhuma face detectada na imagem de referência")
    ref_emb = e.cpu().numpy()
    print(f"[B] referência: {os.path.basename(a.ref_image)} | device: {verifier.device}")

    dev = int(a.device) if str(a.device).isdigit() else a.device
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"não foi possível abrir {a.device}. Rodou o setup_vcam.sh? O atacante já está escrevendo?")

    log_path = a.log if os.path.isabs(a.log) else os.path.join(REPO, a.log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = open(log_path, "w", newline="", buffering=1)
    w = csv.writer(fh)
    w.writerow(["frame", "t_rel", "t_abs", "mode", "duplicado", "detectado", "n_rostos",
                "similarity", "aceita_far1", "aceita_far01", "fps"])

    print(f"[B] aguardando o 1º quadro em {a.device} (até {a.espera:g}s)...")
    t_wait = time.time()
    frame = None
    while time.time() - t_wait < a.espera:
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            break
        time.sleep(0.05)
    if frame is None:
        fh.close()
        cap.release()
        sys.exit(f"[B] nenhum quadro recebido em {a.device}: o atacante está escrevendo?")

    t0 = time.time()
    n = n_unicos = n_det = n_aceitos = 0
    last_hash = None
    last_new = time.time()
    fps_val, fps_c, fps_t = 0.0, 0, time.time()
    print("[B] recebendo quadros. >>> COMECE A SEQUÊNCIA DE MOVIMENTOS AGORA <<<")

    try:
        while True:
            if n > 0:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if time.time() - last_new > a.timeout:
                        print("[B] sem quadros novos; encerrando.")
                        break
                    time.sleep(0.005)
                    continue
            n += 1
            h = hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=8).hexdigest()
            elapsed = time.time() - t0
            dup = int(h == last_hash)
            if dup:
                w.writerow([n, f"{elapsed:.3f}", f"{time.time():.3f}", a.modo, 1, "", "", "", "", "", f"{fps_val:.1f}"])
                if time.time() - last_new > a.timeout:
                    print("[B] sem quadros novos; encerrando.")
                    break
                time.sleep(0.002)
                if a.duration and elapsed >= a.duration:
                    break
                continue

            last_hash = h
            last_new = time.time()
            n_unicos += 1

            n_rostos, bbox, lmk = detectar(verifier, frame)
            sim_txt, ac1, ac01 = "", "", ""
            if n_rostos > 0:
                emb = verifier.get_embedding(frame, bbox=bbox, landmarks=lmk)
                if emb is not None:
                    sim = get_similarity(emb.cpu().numpy(), ref_emb)
                    sim_txt = f"{sim:.6f}"
                    ac1, ac01 = int(sim >= THR_FAR1), int(sim >= THR_FAR01)
                    n_det += 1
                    n_aceitos += int(sim >= THR_FAR1)
                else:
                    n_rostos = 0          # detectou caixa mas não produziu embedding

            fps_c += 1
            if fps_c >= 15:
                fps_val = fps_c / (time.time() - fps_t)
                fps_c, fps_t = 0, time.time()

            w.writerow([n, f"{elapsed:.3f}", f"{time.time():.3f}", a.modo, 0, int(n_rostos > 0), n_rostos,
                        sim_txt, ac1, ac01, f"{fps_val:.1f}"])

            if n_unicos % 30 == 0:
                taxa = 100.0 * n_det / n_unicos
                msg = (f"[B] {elapsed:5.1f}s únicos={n_unicos} fps={fps_val:4.1f} "
                       f"detecção={taxa:5.1f}%")
                if not a.ocultar_similaridade:
                    msg += f" sim={sim_txt or '--'}"
                print(msg)

            if a.max_frames and n_unicos >= a.max_frames:
                break
            if a.duration and elapsed >= a.duration:
                break
    except KeyboardInterrupt:
        print("\n[B] interrompido.")
    finally:
        cap.release()
        fh.close()
        det = 100.0 * n_det / n_unicos if n_unicos else 0.0
        msg = f"[B] fim | lidos={n} únicos={n_unicos} detectados={n_det} ({det:.1f}%)"
        if not a.ocultar_similaridade:
            msg += f" | aceitos@FAR1%={n_aceitos}"
        msg += f" | log={log_path}"
        print(msg)


if __name__ == "__main__":
    main()
