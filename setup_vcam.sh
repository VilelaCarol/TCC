#!/bin/bash
# =============================================================================
# setup_vcam.sh — cria UMA câmera virtual (v4l2loopback) para o experimento de
# injeção por câmera virtual (PROTOCOLO_VCAM.md).
#
# Usa o número 10 de propósito: a webcam do notebook costuma ocupar /dev/video0
# e /dev/video1 (às vezes também 2 e 3, se houver câmera IR). O setup_realtime.sh
# antigo pedia video_nr=0,2 para o OBS e pode colidir com a webcam.
# =============================================================================
VNR=${1:-10}
sudo -v || exit 1
sudo modprobe -r v4l2loopback 2>/dev/null
sudo modprobe v4l2loopback devices=1 video_nr="$VNR" card_label="DeepfakeVirtual" exclusive_caps=1 \
  || { echo "Falha ao carregar v4l2loopback (sudo apt install v4l2loopback-dkms v4l2loopback-utils)"; exit 1; }
echo
v4l2-ctl --list-devices
echo
echo "Câmera virtual: /dev/video$VNR"
echo "Confira acima qual /dev/videoN é a SUA WEBCAM e passe esse N em --source"
echo "(ex.: rodar_sessoes_vcam.py --source 0 se a webcam for /dev/video0)."
