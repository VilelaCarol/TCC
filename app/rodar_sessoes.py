#!/usr/bin/env python3
"""
Executor do protocolo multi-sessão do real-time (revisão do protocolo item 3).
==============================================================================
Lê o plano congelado (app/results/sessoes_plano.json, gerado com semente fixa)
e roda as sessões EM SEQUÊNCIA, uma identidade por vez, chamando
app/realtime_live_attack.py por subprocess. Antes de cada sessão espera o ENTER
para o atacante se posicionar na câmera.

Para cada entrada do plano:
  --source-image <fotoA>   (rosto injetado no swap, no modo ATAQUE)
  --ref-image    <fotoB>   (outra foto da MESMA identidade = referência blindada)
  --log sessoes/<prefixo>_<n:02d>_<id>.csv

Dois usos típicos (o modo e o prefixo são independentes; use pares coerentes):
  # ataque: deepfake da vítima injetado no rosto do atacante
  .venv/bin/python app/rodar_sessoes.py --modo ATAQUE      --prefixo atq --duration 30
  # controle impostor: rosto REAL do atacante (sem swap) vs referência da vítima
  .venv/bin/python app/rodar_sessoes.py --modo PASSTHROUGH --prefixo imp --duration 30

Depois:
  ATAQUE      -> .venv/bin/python app/agg_sessoes_realtime.py "app/results/sessoes/atq_*.csv"
  PASSTHROUGH -> .venv/bin/python app/analise_controles_realtime.py

NÃO calcula ASR aqui — só orquestra a coleta. A análise fica nos scripts acima.
==============================================================================
"""
import os
import sys
import json
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "app", "results")
PLANO_DEFAULT = os.path.join(RES, "sessoes_plano.json")
REALTIME = os.path.join(REPO, "app", "realtime_live_attack.py")
os.chdir(REPO)  # fotoA/fotoB no plano são relativos à raiz do repo


def parse_args():
    p = argparse.ArgumentParser(description="Executa as sessões do protocolo multi-sessão do real-time")
    p.add_argument("--modo", choices=["ATAQUE", "PASSTHROUGH"], default="ATAQUE",
                   help="ATAQUE = deepfake injetado; PASSTHROUGH = rosto real (controle impostor)")
    p.add_argument("--prefixo", default="atq",
                   help="Prefixo do nome do log (ex.: atq p/ ataque, imp p/ impostor)")
    p.add_argument("--duration", type=float, default=30.0, help="Segundos por sessão (padrão 30)")
    # extras práticos (têm padrão; não precisam ser informados)
    p.add_argument("--plano", default=PLANO_DEFAULT, help="JSON do plano congelado")
    p.add_argument("--source", default="0", help="Fonte de vídeo repassada ao realtime (0 = /dev/video0)")
    p.add_argument("--port", type=int, default=8080, help="Porta do servidor MJPEG (http://localhost:PORT/)")
    p.add_argument("--dry-run", action="store_true", help="Só mostra os comandos, não roda nada")
    return p.parse_args()


def main():
    a = parse_args()

    if not os.path.exists(a.plano):
        print(f"ERRO: plano não encontrado: {a.plano}")
        print("Gere-o primeiro (Passo 0) com semente fixa. LFW real: data/lfw/lfw_home/lfw_funneled")
        sys.exit(1)

    plano = json.load(open(a.plano))
    plano = sorted(plano, key=lambda x: x["n"])
    print(f"plano: {a.plano}  ({len(plano)} identidades)")
    print(f"modo={a.modo} | prefixo={a.prefixo} | duração={a.duration:.0f}s/sessão | fonte={a.source} | porta={a.port}")
    print(f"logs em: app/results/sessoes/{a.prefixo}_<n>_<id>.csv")
    print("=" * 72)

    feitas, puladas, faltando = 0, 0, 0
    for x in plano:
        n, idn = x["n"], x["id"]
        fotoA, fotoB = x["fotoA"], x["fotoB"]
        log_rel = os.path.join("sessoes", f"{a.prefixo}_{n:02d}_{idn}.csv")
        log_abs = os.path.join(RES, log_rel)

        if os.path.exists(log_abs):
            print(f"[{n:02d}/{len(plano)}] {idn}: log já existe -> PULANDO ({log_rel})")
            puladas += 1
            continue
        if not (os.path.exists(fotoA) and os.path.exists(fotoB)):
            print(f"[{n:02d}/{len(plano)}] {idn}: AVISO imagem faltando (A={os.path.exists(fotoA)} B={os.path.exists(fotoB)}) -> PULANDO")
            faltando += 1
            continue

        cmd = [sys.executable, REALTIME,
               "--source", str(a.source),
               "--mode", a.modo,
               "--source-image", fotoA,
               "--ref-image", fotoB,
               "--duration", str(a.duration),
               "--log", log_rel,
               "--port", str(a.port)]

        print("\n" + "-" * 72)
        print(f">> PRÓXIMA SESSÃO [{n:02d}/{len(plano)}]  identidade: {idn}  modo: {a.modo}")
        print(f"   swap/fonte : {os.path.basename(fotoA)}")
        print(f"   referência : {os.path.basename(fotoB)}  (blindada, mesma identidade)")
        print(f"   vídeo ao vivo: http://localhost:{a.port}/   | log: {log_rel}")
        if a.dry_run:
            print("   (dry-run) " + " ".join(cmd))
            feitas += 1
            continue
        try:
            input("   Posicione-se na câmera e pressione ENTER (Ctrl+C aborta tudo)... ")
        except KeyboardInterrupt:
            print("\nabortado pelo usuário.")
            break

        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"   AVISO: a sessão terminou com código {r.returncode}.")
        feitas += 1

    print("\n" + "=" * 72)
    print(f"resumo: rodadas={feitas} | puladas(já existiam)={puladas} | puladas(img faltando)={faltando}")
    if a.modo == "ATAQUE":
        print('análise: .venv/bin/python app/agg_sessoes_realtime.py "app/results/sessoes/%s_*.csv"' % a.prefixo)
    else:
        print("análise: .venv/bin/python app/analise_controles_realtime.py")


if __name__ == "__main__":
    main()
