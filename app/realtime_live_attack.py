"""
================================================================================
REAL-TIME DEEPFAKE ATTACK — VERIFICAÇÃO AO VIVO (versão refatorada p/ 6GB)
================================================================================
Objetivo: medir o ASR (Attack Success Rate) AO VIVO, dentro do loop — e não
offline como na versão antiga (crítica dos revisores).

Pipeline por frame (modo ATAQUE):
  captura 480p
    → FaceFusion detecta a face UMA vez e faz o swap (inswapper_128)
    → reaproveita a MESMA bbox para a verificação AdaFace (sem re-detectar)
    → similaridade cosseno vs. atacante → veredito nos thresholds 0.45 / 0.60
    → HUD + ASR acumulado + servidor MJPEG

Por que é mais rápido que a versão antiga:
  A versão antiga rodava 3 detectores por frame (YOLO avulso + detecção interna
  do FaceFusion + YOLO dentro do AdaFace). Aqui roda só a detecção interna do
  FaceFusion e ela é reaproveitada. A verificação AdaFace custa ~4 ms.

Uso (headless, via servidor MJPEG — abra http://<ip>:8080/ no navegador):
  python app/realtime_live_attack.py --source 0 \\
      --source-image data/dataset_pessoal/originais/victorsorrindoclaro.jpeg

  Teclas (se houver display): q=sair  m=alterna ATAQUE/PASSTHROUGH
================================================================================
"""

import os
import sys

# Headless-safe (precisa vir antes de importar cv2)
_HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
if not _HAS_DISPLAY:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import csv
import time
import queue
import argparse
import threading
import numpy as np

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(APP_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "yolo8face_adaface"))
sys.path.insert(0, APP_DIR)

from src import AdaFaceVerifier
from src.utils import get_similarity

CONFIG_PATH = os.path.join(REPO_DIR, "yolo8face_adaface", "configs", "config.yaml")
RESULTS_DIR = os.path.join(APP_DIR, "results")
THRESHOLD_LOW = 0.45
THRESHOLD_HIGH = 0.60

GREEN, RED, YELLOW, WHITE, DARK = (0, 220, 80), (0, 60, 220), (0, 200, 220), (255, 255, 255), (20, 20, 20)


# ============================================================================ #
# Servidor MJPEG (visualização ao vivo no navegador, sem display local)
# ============================================================================ #
import time as _t
from http.server import BaseHTTPRequestHandler, HTTPServer


class MJPEGServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._jpg = b""
        self._server = None

    def push(self, frame_bgr, quality=70):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            with self._lock:
                self._jpg = buf.tobytes()

    def _get(self):
        with self._lock:
            return self._jpg

    def start(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path not in ("/", "/stream"):
                    self.send_error(404); return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpg = srv._get()
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                        _t.sleep(0.033)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = HTTPServer((self.host, self.port), H)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"📡 MJPEG: http://{self.host}:{self.port}/  (abra no navegador)")

    def stop(self):
        if self._server:
            self._server.shutdown()


# ============================================================================ #
# HUD
# ============================================================================ #
def draw_hud(frame, sim, match, fps, bbox, mode, asr_low, asr_high, n):
    out = frame.copy()
    h, w = out.shape[:2]
    if bbox is not None:
        color = RED if match else GREEN
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    cv2.rectangle(out, (0, 0), (w, 48), DARK, -1)
    cv2.putText(out, f"MODE: {mode}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    cv2.putText(out, f"FPS: {fps:.1f}   frames: {n}", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(out, f"ASR  t0.45:{asr_low:5.1f}%  t0.60:{asr_high:5.1f}%", (w - 290, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, h - 62), (w, h), DARK, -1)
    col = RED if match else GREEN
    tl = "PASS" if sim >= THRESHOLD_LOW else "FAIL"
    th = "PASS" if sim >= THRESHOLD_HIGH else "FAIL"
    verdict = "FRAUDE ACEITA (ASR +1)" if match else "BLOQUEADO"
    cv2.putText(out, f"Sim: {sim:.4f}  |  t=0.45: {tl}  |  t=0.60: {th}", (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    cv2.putText(out, verdict, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
    return out


# ============================================================================ #
# Loop principal
# ============================================================================ #
def run(args):
    source = int(args.source) if str(args.source).isdigit() else args.source

    print("🚀 Carregando AdaFace + YOLO...")
    verifier = AdaFaceVerifier(CONFIG_PATH)
    print(f"   device: {verifier.device}")

    atk = cv2.imread(args.source_image)
    if atk is None:
        raise FileNotFoundError(f"Imagem do atacante não encontrada: {args.source_image}")

    # Referência de verificação: por padrão é a MESMA imagem do swap (self), mas
    # o correto/blindado é uma foto DIFERENTE da mesma vítima (--ref-image) — evita
    # o self-match que infla a similaridade. O swap continua usando --source-image.
    ref_path = args.ref_image or args.source_image
    ref_src = cv2.imread(ref_path)
    if ref_src is None:
        raise FileNotFoundError(f"Imagem de referência não encontrada: {ref_path}")
    ref = verifier.get_embedding(ref_src)
    if ref is None:
        raise RuntimeError("Nenhuma face detectada na imagem de referência.")
    ref_emb = ref.cpu().numpy()
    sep = "DIFERENTE do swap (blindado)" if args.ref_image and args.ref_image != args.source_image else "= fonte do swap (self)"
    print(f"   referência: {os.path.basename(ref_path)} [{sep}]  emb={ref_emb.shape}")

    from neural_face_swap import NeuralFaceSwapper
    swapper = NeuralFaceSwapper(atk)

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if isinstance(source, int) else cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {source}")

    mjpeg = MJPEGServer(port=args.port)
    mjpeg.start()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "realtime_live_log.csv")
    csv_f = open(csv_path, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame", "mode", "similarity", "match_045", "match_060", "elapsed_s", "fps"])

    writer = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.record, fourcc, args.record_fps, (args.width, args.height))
        print(f"🎥 Gravando em: {args.record}")

    mode = args.mode
    headless = not _HAS_DISPLAY
    n = 0
    attack_frames = 0
    match_low = 0
    match_high = 0
    last_swap, last_bbox, last_lmk, last_sim, last_match = None, None, None, 0.0, False
    fps_val, fps_c, fps_t = 0.0, 0, time.time()
    t0 = time.time()
    print(f"🔴 Rodando | modo={mode} | swap a cada {args.swap_every} frame(s)\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if isinstance(source, str):     # fim de arquivo de vídeo → loop
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                time.sleep(0.01); continue
            frame = cv2.resize(frame, (args.width, args.height))
            n += 1

            if mode == "ATAQUE":
                do_swap = (n % args.swap_every == 0) or (last_swap is None)
                if do_swap:
                    processed, bbox, lmk = swapper.swap_and_locate(frame)
                    last_swap, last_bbox, last_lmk = processed, bbox, lmk
                else:
                    processed, bbox, lmk = last_swap, last_bbox, last_lmk  # reaproveita último swap
                # Verificação AO VIVO na face swapada, alinhando pelos landmarks do FaceFusion
                emb = verifier.get_embedding(processed, bbox=bbox, landmarks=lmk) if bbox is not None else None
                attack_frames += 1
            else:  # PASSTHROUGH: frame real, verifica a face real vs atacante (deve rejeitar)
                processed = frame
                bbox, lmk = verifier.detect(frame)
                emb = verifier.get_embedding(frame, bbox=bbox, landmarks=lmk) if bbox is not None else None

            if emb is not None:
                sim = get_similarity(emb.cpu().numpy(), ref_emb)
                last_sim = sim
            else:
                sim = last_sim
            match = sim >= THRESHOLD_LOW
            last_match = match

            if mode == "ATAQUE" and emb is not None:
                if sim >= THRESHOLD_LOW:  match_low += 1
                if sim >= THRESHOLD_HIGH: match_high += 1

            asr_low = 100.0 * match_low / attack_frames if attack_frames else 0.0
            asr_high = 100.0 * match_high / attack_frames if attack_frames else 0.0

            fps_c += 1
            if fps_c >= 15:
                fps_val = fps_c / (time.time() - fps_t); fps_c, fps_t = 0, time.time()

            out = draw_hud(processed, sim, match, fps_val, bbox, mode, asr_low, asr_high, n)
            mjpeg.push(out)
            if writer is not None:
                writer.write(out)

            elapsed = time.time() - t0
            csv_w.writerow([n, mode, f"{sim:.6f}", int(sim >= THRESHOLD_LOW),
                            int(sim >= THRESHOLD_HIGH), f"{elapsed:.2f}", f"{fps_val:.1f}"])

            if n % 30 == 0:
                print(f"   [{elapsed:6.1f}s] frames={n} fps={fps_val:4.1f} sim={sim:.3f} "
                      f"ASR(0.45)={asr_low:.1f}% ASR(0.60)={asr_high:.1f}%")

            if not headless:
                cv2.imshow("Real-Time Live Attack", out)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('m'):
                    mode = "PASSTHROUGH" if mode == "ATAQUE" else "ATAQUE"
                    print(f"🔄 modo: {mode}")

            if args.max_frames and n >= args.max_frames:
                break
            if args.duration and elapsed >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n⛔ Interrompido.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        mjpeg.stop()
        csv_f.close()
        if not headless:
            cv2.destroyAllWindows()
        elapsed = time.time() - t0
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO — ATAQUE AO VIVO")
        print("=" * 60)
        print(f"   Frames totais     : {n}")
        print(f"   Frames de ataque  : {attack_frames}")
        print(f"   FPS médio         : {n / elapsed:.1f}" if elapsed else "   FPS médio: N/A")
        print(f"   ASR (τ=0.45)      : {100.0*match_low/attack_frames:.2f}%" if attack_frames else "   ASR: N/A")
        print(f"   ASR (τ=0.60)      : {100.0*match_high/attack_frames:.2f}%" if attack_frames else "")
        print(f"   Log               : {csv_path}")
        if args.record:
            print(f"   Vídeo             : {args.record}")
        print("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="Real-Time Deepfake Attack — verificação AO VIVO")
    p.add_argument("--source", type=str, default="0", help="0 = /dev/video0 (webcam/OBS), ou caminho de vídeo")
    p.add_argument("--source-image", type=str,
                   default="data/dataset_pessoal/originais/victorsorrindoclaro.jpeg",
                   help="Imagem do atacante (rosto a injetar no swap)")
    p.add_argument("--ref-image", type=str, default=None,
                   help="Imagem de referência da vítima (DIFERENTE da fonte do swap) p/ blindar o self-match")
    p.add_argument("--mode", choices=["ATAQUE", "PASSTHROUGH"], default="ATAQUE")
    p.add_argument("--swap-every", type=int, default=1, help="Faz swap a cada N frames (1=todo frame)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--record", type=str, default=None, help="Salva o vídeo com HUD (ex: demo.mp4)")
    p.add_argument("--record-fps", type=float, default=12.0)
    p.add_argument("--max-frames", type=int, default=None, help="Para após N frames (headless)")
    p.add_argument("--duration", type=float, default=None, help="Para após N segundos (headless)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
