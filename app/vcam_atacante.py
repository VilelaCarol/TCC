#!/usr/bin/env python3
"""
Processo A — atacante: webcam -> FaceFusion -> câmera virtual (v4l2loopback).
==============================================================================
Lê a webcam, aplica a troca de rosto (inswapper_128) e ESCREVE o quadro
manipulado em /dev/video10, de onde o processo B (app/vcam_verificador.py) o
consome como se fosse uma webcam comum.

No modo PASSTHROUGH escreve o quadro CRU (rosto real da atacante), sem
manipulação: é o controle de impostor, pelo mesmo canal.

Mede também, no laço, a similaridade contra a referência (--ref-image), do
mesmo jeito que app/realtime_live_attack.py faz hoje. Essa é a medida EMULADA;
a medida oficial deste experimento é a do processo B. As duas, por sessão,
formam a comparação pareada que isola o efeito do canal (PROTOCOLO_VCAM.md, 2).

Pré-requisitos:
  bash setup_vcam.sh              # cria /dev/video10 (v4l2loopback)
  .venv/bin/pip install pyvirtualcam

Uso (normalmente chamado por app/rodar_sessoes_vcam.py):
  .venv/bin/python app/vcam_atacante.py --modo ATAQUE \
      --source-image data/.../Id_0001.jpg --ref-image data/.../Id_0002.jpg \
      --duration 30 --log app/results/sessoes_vcam/atq_01_Id_A.csv
"""
import os
import sys
import csv
import time
import signal
import hashlib
import argparse

_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
if not _HAS_DISPLAY:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "yolo8face_adaface"))
os.chdir(REPO)

import cv2
import numpy as np

CONFIG_PATH = os.path.join(REPO, "yolo8face_adaface", "configs", "config.yaml")
RESULTS_DIR = os.path.join(REPO, "app", "results")


def parse_args():
    p = argparse.ArgumentParser(description="Atacante: swap ao vivo entregue por câmera virtual")
    p.add_argument("--source", default="0", help="webcam de entrada (0 = /dev/video0) ou arquivo")
    p.add_argument("--source-image", required=True, help="fotoA: rosto injetado (fonte do swap)")
    p.add_argument("--ref-image", default=None, help="fotoB: referência blindada (medida emulada)")
    p.add_argument("--modo", choices=["ATAQUE", "PASSTHROUGH"], default="ATAQUE")
    p.add_argument("--vcam", default="/dev/video10", help="dispositivo v4l2loopback de saída")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=15.0, help="fps nominal da câmera virtual")
    p.add_argument("--duration", type=float, default=0.0,
                   help="segundos; 0 = roda até ser encerrado (padrão: o executor encerra A quando B termina)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--log", required=True, help="CSV de saída (medida emulada)")
    p.add_argument("--sem-medida-emulada", action="store_true",
                   help="não mede a similaridade no laço (poupa GPU; perde a comparação pareada)")
    p.add_argument("--preview", action="store_true",
                   help="mostra numa janela OpenCV o quadro ENVIADO à câmera virtual "
                        "(deepfake no ATAQUE, rosto cru no PASSTHROUGH); sem similaridade, "
                        "sem veredito, sem número de resultado. Não altera o quadro enviado.")
    p.add_argument("--ocultar-similaridade", action="store_true",
                   help="não imprime a similaridade no console (cumpre o §3.5). "
                        "Afeta só o print; o CSV continua gravando similarity_emulado igual.")
    return p.parse_args()


def main():
    a = parse_args()
    # o executor encerra A com SIGTERM quando B termina: converte em SystemExit p/ o finally fechar o log
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        import pyvirtualcam
    except ImportError:
        sys.exit("pyvirtualcam não instalado. Rode: .venv/bin/pip install pyvirtualcam")

    src_img = cv2.imread(a.source_image)
    if src_img is None:
        sys.exit(f"fotoA não encontrada: {a.source_image}")

    verifier = None
    ref_emb = None
    if not a.sem_medida_emulada:
        from src import AdaFaceVerifier
        from src.utils import get_similarity
        verifier = AdaFaceVerifier(CONFIG_PATH)
        ref_path = a.ref_image or a.source_image
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            sys.exit(f"referência não encontrada: {ref_path}")
        e = verifier.get_embedding(ref_img)
        if e is None:
            sys.exit("nenhuma face detectada na imagem de referência")
        ref_emb = e.cpu().numpy()
        print(f"[A] referência: {os.path.basename(ref_path)} | device: {verifier.device}")
    else:
        get_similarity = None

    from neural_face_swap import NeuralFaceSwapper
    swapper = NeuralFaceSwapper(src_img) if a.modo == "ATAQUE" else None

    source = int(a.source) if str(a.source).isdigit() else a.source
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if isinstance(source, int) else cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"não foi possível abrir a webcam: {a.source}")

    log_path = a.log if os.path.isabs(a.log) else os.path.join(REPO, a.log)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = open(log_path, "w", newline="", buffering=1)   # linha a linha: o executor lê o log de A para saber que ele está no ar
    w = csv.writer(fh)
    w.writerow(["frame", "t_abs", "mode", "elapsed_s", "fps", "swap_ok", "similarity_emulado", "frame_hash"])

    n = 0
    n_swap_ok = 0
    fps_val, fps_c, fps_t = 0.0, 0, time.time()
    t0 = None
    print(f"[A] modo={a.modo} | saída={a.vcam} | {a.width}x{a.height} @ {a.fps:g} fps nominal")

    # preview: janela que mostra o quadro EXATAMENTE como é enviado à câmera virtual,
    # para a atacante se ver enquanto grava (como nas sessões antigas). É só visualização:
    # sem similaridade, sem aceito/rejeitado, sem número de resultado. Sem DISPLAY, segue
    # sem janela (não quebra a coleta headless).
    preview = bool(a.preview)
    if preview and not _HAS_DISPLAY:
        print("[A] --preview pedido, mas sem DISPLAY: seguindo sem janela de preview.")
        preview = False
    preview_win = f"Atacante -> {a.vcam}  [{a.modo}]"
    if preview:
        cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_win, a.width, a.height)

    try:
        with pyvirtualcam.Camera(width=a.width, height=a.height, fps=a.fps,
                                 device=a.vcam, print_fps=False) as cam:
            print(f"[A] câmera virtual aberta: {cam.device}")
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                if frame.shape[1] != a.width or frame.shape[0] != a.height:
                    frame = cv2.resize(frame, (a.width, a.height))
                if t0 is None:
                    t0 = time.time()
                n += 1

                bbox = lmk = None
                if a.modo == "ATAQUE":
                    processed, bbox, lmk = swapper.swap_and_locate(frame)
                    if bbox is not None:
                        n_swap_ok += 1
                else:
                    processed = frame

                # entrega pelo canal (pyvirtualcam espera RGB)
                cam.send(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
                cam.sleep_until_next_frame()

                # preview opcional: desenha só numa CÓPIA de `processed` — nunca no próprio
                # `processed`, que já foi enviado acima e ainda alimenta a medida emulada
                # abaixo. Mostra, no máximo, o tempo decorrido; nada de similaridade/veredito.
                if preview:
                    disp = processed.copy()
                    cv2.putText(disp, f"t={time.time() - t0:5.1f}s", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(preview_win, disp)
                    cv2.waitKey(1)

                sim_txt = ""
                if verifier is not None:
                    if a.modo == "ATAQUE":
                        emb = verifier.get_embedding(processed, bbox=bbox, landmarks=lmk) if bbox is not None else None
                    else:
                        b2, l2 = verifier.detect(processed)
                        emb = verifier.get_embedding(processed, bbox=b2, landmarks=l2) if b2 is not None else None
                    if emb is not None:
                        sim_txt = f"{get_similarity(emb.cpu().numpy(), ref_emb):.6f}"

                fps_c += 1
                if fps_c >= 15:
                    fps_val = fps_c / (time.time() - fps_t)
                    fps_c, fps_t = 0, time.time()

                elapsed = time.time() - t0
                h = hashlib.blake2b(np.ascontiguousarray(processed).tobytes(), digest_size=8).hexdigest()
                w.writerow([n, f"{time.time():.3f}", a.modo, f"{elapsed:.2f}", f"{fps_val:.1f}",
                            int(bbox is not None) if a.modo == "ATAQUE" else "",
                            sim_txt, h])

                if n % 30 == 0:
                    taxa = 100.0 * n_swap_ok / n if a.modo == "ATAQUE" else 100.0
                    msg = f"[A] {elapsed:5.1f}s frames={n} fps={fps_val:4.1f} swap_ok={taxa:5.1f}%"
                    if not a.ocultar_similaridade:
                        msg += f" sim={sim_txt or '--'}"
                    print(msg)

                if a.max_frames and n >= a.max_frames:
                    break
                if a.duration and elapsed >= a.duration:
                    break
    except KeyboardInterrupt:
        print("\n[A] interrompido.")
    finally:
        cap.release()
        fh.close()
        if preview:
            cv2.destroyAllWindows()
        print(f"[A] fim | frames={n} | swap ok={n_swap_ok} | log={log_path}")


if __name__ == "__main__":
    main()
