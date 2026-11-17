#!/usr/bin/env python3
"""
Executor das sessões do experimento de câmera virtual (PROTOCOLO_VCAM.md, 4).
==============================================================================
Para cada identidade de app/results/sessoes_plano_vcam.json, sobe DOIS processos:

  A (app/vcam_atacante.py)     webcam -> FaceFusion -> /dev/video10   + log emulado
  B (app/vcam_verificador.py)  /dev/video10 -> YOLOv8-Face + AdaFace  + log do canal

Espera o ENTER entre as sessões e pula as que já têm log de B.

Uso:
  .venv/bin/python app/rodar_sessoes_vcam.py --modo ATAQUE      --prefixo atq --duration 30
  .venv/bin/python app/rodar_sessoes_vcam.py --modo PASSTHROUGH --prefixo imp --duration 30
"""
import os
import sys
import json
import time
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
SESS = os.path.join(RES, "sessoes_vcam")
PLANO_DEFAULT = os.path.join(RES, "sessoes_plano_vcam.json")
ATACANTE = os.path.join(REPO, "app", "vcam_atacante.py")
VERIFICADOR = os.path.join(REPO, "app", "vcam_verificador.py")
os.chdir(REPO)


def linhas_ok(path, n):
    """True se o arquivo existe e já tem pelo menos n linhas."""
    try:
        with open(path) as fh:
            return sum(1 for _ in fh) >= n
    except OSError:
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Sessões de injeção por câmera virtual")
    p.add_argument("--modo", choices=["ATAQUE", "PASSTHROUGH"], default="ATAQUE")
    p.add_argument("--prefixo", default="atq", help="atq p/ ataque, imp p/ controle impostor")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--plano", default=PLANO_DEFAULT)
    p.add_argument("--source", default="0", help="webcam de entrada do atacante")
    p.add_argument("--vcam", default="/dev/video10", help="dispositivo v4l2loopback")
    p.add_argument("--fps", type=float, default=15.0, help="fps nominal da câmera virtual")
    p.add_argument("--atraso-b", type=float, default=1.0, help="segundos entre A estar no ar e subir B")
    p.add_argument("--espera-a", type=float, default=90.0, help="segundos aguardando A começar a escrever")
    p.add_argument("--preview", dest="preview", action="store_true", default=True,
                   help="mostra no atacante a janela com o quadro enviado à câmera virtual (padrão: ligado)")
    p.add_argument("--sem-preview", dest="preview", action="store_false",
                   help="desliga a janela de preview do atacante")
    p.add_argument("--ocultar-similaridade", dest="ocultar_sim", action="store_true", default=True,
                   help="oculta a similaridade no console de A e B durante a coleta (padrão: ligado; §3.5)")
    p.add_argument("--mostrar-similaridade", dest="ocultar_sim", action="store_false",
                   help="mostra a similaridade no console (só p/ depuração; não usar na coleta)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if not os.path.exists(a.plano):
        sys.exit(f"ERRO: plano não encontrado: {a.plano}\nRode antes: .venv/bin/python app/gerar_plano_vcam.py")
    if not os.path.exists(a.vcam):
        print(f"AVISO: {a.vcam} não existe. Rode antes: bash setup_vcam.sh")

    plano = sorted(json.load(open(a.plano, encoding="utf-8")), key=lambda x: x["n"])
    os.makedirs(SESS, exist_ok=True)
    print(f"plano: {a.plano} ({len(plano)} identidades)")
    print(f"modo={a.modo} | prefixo={a.prefixo} | {a.duration:.0f}s/sessão | "
          f"webcam={a.source} | câmera virtual={a.vcam}")
    print(f"logs em: app/results/sessoes_vcam/{a.prefixo}_<n>_<id>_{{A,B}}.csv")
    print("=" * 72)

    feitas = puladas = faltando = 0
    for x in plano:
        n, idn = x["n"], x["id"]
        fotoA, fotoB = x["fotoA"], x["fotoB"]
        log_a = os.path.join(SESS, f"{a.prefixo}_{n:02d}_{idn}_A.csv")
        log_b = os.path.join(SESS, f"{a.prefixo}_{n:02d}_{idn}_B.csv")

        if os.path.exists(log_b):
            print(f"[{n:02d}/{len(plano)}] {idn}: log de B já existe -> PULANDO")
            puladas += 1
            continue
        if not (os.path.exists(fotoA) and os.path.exists(fotoB)):
            print(f"[{n:02d}/{len(plano)}] {idn}: AVISO imagem faltando -> PULANDO")
            faltando += 1
            continue

        cmd_a = [sys.executable, ATACANTE, "--modo", a.modo, "--source", str(a.source),
                 "--source-image", fotoA, "--ref-image", fotoB, "--vcam", a.vcam,
                 "--fps", str(a.fps), "--duration", "0", "--log", log_a]   # A roda até B terminar
        if a.preview:
            cmd_a.append("--preview")      # a atacante se vê enquanto grava (PROTOCOLO_VCAM.md, 3.5)
        cmd_b = [sys.executable, VERIFICADOR, "--device", a.vcam, "--ref-image", fotoB,
                 "--modo", a.modo, "--duration", str(a.duration), "--log", log_b]
        if a.ocultar_sim:                       # §3.5: a atacante não vê similaridade no console
            cmd_a.append("--ocultar-similaridade")
            cmd_b.append("--ocultar-similaridade")

        print("\n" + "-" * 72)
        print(f">> PRÓXIMA SESSÃO [{n:02d}/{len(plano)}]  identidade: {idn}  modo: {a.modo}")
        print(f"   fonte do swap : {os.path.basename(fotoA)}")
        print(f"   referência    : {os.path.basename(fotoB)}  (blindada)")
        print(f"   movimentos    : cabeça para os lados, inclinação vertical, abrir a boca")
        if a.dry_run:
            print("   (dry-run) A: " + " ".join(cmd_a))
            print("   (dry-run) B: " + " ".join(cmd_b))
            feitas += 1
            continue
        try:
            input("   Posicione-se na câmera e pressione ENTER (Ctrl+C aborta tudo)... ")
        except KeyboardInterrupt:
            print("\nabortado pelo usuário.")
            break

        # A sobe primeiro e só é encerrado quando B termina: assim a janela de
        # `duration` segundos de B (contada do 1º quadro que B recebe) fica
        # inteira dentro da execução de A, mesmo com o tempo de carga dos modelos.
        pa = subprocess.Popen(cmd_a, cwd=REPO)
        t_esp = time.time()
        while not linhas_ok(log_a, 2):          # header + 1º quadro escrito
            if pa.poll() is not None or time.time() - t_esp > a.espera_a:
                break
            time.sleep(0.2)
        if pa.poll() is not None or not linhas_ok(log_a, 2):
            print(f"   ERRO: o atacante não começou a escrever na câmera virtual (código {pa.poll()}).")
            pa.terminate(); pa.wait()
            continue
        time.sleep(a.atraso_b)
        print("   atacante no ar; subindo o verificador (aguarde '[B] recebendo quadros' para começar os movimentos)")
        pb = subprocess.Popen(cmd_b, cwd=REPO)

        try:
            rb = pb.wait(timeout=a.duration + 120)
        except KeyboardInterrupt:
            pa.terminate(); pb.terminate()
            print("\nabortado pelo usuário.")
            break
        except subprocess.TimeoutExpired:
            print("   AVISO: o verificador não encerrou sozinho; finalizando.")
            pb.terminate()
            rb = pb.wait()
        pa.terminate()
        try:
            ra = pa.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pa.kill(); ra = pa.wait()

        if ra != 0:
            print(f"   AVISO: atacante terminou com código {ra}.")
        if rb != 0:
            print(f"   AVISO: verificador terminou com código {rb}.")
        if not os.path.exists(log_b):
            print("   AVISO: o log de B não foi criado — a sessão NÃO conta. Verifique a câmera virtual.")
        feitas += 1

    print("\n" + "=" * 72)
    print(f"resumo: rodadas={feitas} | puladas(já existiam)={puladas} | puladas(img faltando)={faltando}")
    print("análise: .venv/bin/python app/agg_sessoes_vcam.py")


if __name__ == "__main__":
    main()
