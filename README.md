# Avaliação de Vulnerabilidade de Verificação Facial a Deepfakes Injetados

Red teaming de um sistema de **verificação facial** contra ataques de **deepfake por face swap injetado no laço de captura**, em três cenários — foto (LFW), vídeo (YTF) e tempo real (webcam, agora também com **injeção por câmera virtual v4l2loopback**) — unificados na mesma régua metodológica: alinhamento ArcFace de 5 pontos, **referência blindada** (separada da fonte do swap) e **limiar calibrado por FAR** (ponto de operação do sistema).

> **Cenário ao vivo — duas condições.** (1) *Emulada no laço*: o quadro manipulado é entregue diretamente à verificação. (2) *Por câmera virtual (v4l2loopback)*: um processo **independente** do sistema-alvo lê `/dev/video10` com o próprio YOLOv8-Face, então o alvo **vê o deepfake** e o quadro **atravessa o canal**. A condição por câmera virtual seguiu protocolo **pré-registrado** (commit `f45f69a`; ver [`PROTOCOLO_VCAM.md`](PROTOCOLO_VCAM.md) e `GUIA_DO_PROJETO.md` §8.7) e **foi executada**: **15/15 ataques aceitos** (IC95% Wilson 79,6–100), **0/15 controles**, **detecção 100%**, e diferença emulado × canal de **no máximo 0,005 por sessão** (Wilcoxon pareado p=0,76) — neste canal local, sem compressão de vídeo, a medida **não muda**.

> **Documentação completa** (mapa de arquivos, decisões, como rodar cada parte, resultados e limitações): **[`GUIA_DO_PROJETO.md`](GUIA_DO_PROJETO.md)**

## Sistema avaliado

- **Vítima:** YOLOv8-Face (detecção + 5 landmarks) → **AdaFace IR-50** (embedding 512-d, treinado no **WebFace4M**) → similaridade cosseno.
- **Ataque:** FaceFusion / **inswapper_128** (face swap).
- **Métrica:** ASR (Attack Success Rate) = fração de ataques com similaridade ≥ limiar.

## Ambiente

- Python **3.11** (venv `uv`), **torch 2.6.0+cu124**, **FaceFusion 3.6.1**, onnxruntime-gpu 1.24.4; GPU RTX 4050 Laptop (6 GB).
- Modelo: **AdaFace IR-50 WebFace4M**, checkpoint **local** (`data/models/adaface_ir50_webface4m.ckpt`).
- Datasets (`data/`), pesos, `facefusion/` e a `.venv` ficam **fora do Git** (ver `.gitignore`).

## Resultados finais (blindado + calibrado por FAR)

| Cenário | Verificador | ASR @ FAR 1% | ASR @ FAR 0.1% | Obs. |
|---|---|---|---|---|
| **LFW** (foto) | AdaFace | **91.8%** | 91.3% | 3000 ataques |
| **YTF** (vídeo) | AdaFace | **87.7%** | 84.3% | 616 ataques (ver descartes no GUIA) |
| **LFW** (cross-check) | **FaceNet** (VGGFace2) | **87.9%** | 83.7% | ataque transfere p/ outra família |
| **Tempo real** (webcam, emulada) | AdaFace | **100%** | 100% | 15 sessões (unidade = sessão); 14 válidas, 1 excluída; controle impostor 0/15 |
| **Tempo real** (câmera virtual) | AdaFace | **100%** | 100% | 15 sessões por câmera virtual (v4l2loopback); **15/15** ataques, **0/15** controle; detecção 100%; canal ≈ emulado (ver GUIA §8.7) |

Leitura detalhada e tabela self × blindado: `app/results/tabela_final_blindada.txt` e GUIA §5. Limiares fixos 0.45/0.60 **não** são o ponto de operação (rejeitariam 29–55% dos usuários legítimos); por isso o número reportado é calibrado por FAR.

## Como rodar (resumo — detalhes e tempos/VRAM no GUIA §4)

```bash
# LFW (jeito 1 vs 2 + ROC/EER/FAR)
.venv/bin/python app/reverify_lfw_jeito12.py
# YTF (precisa de data/YouTubeFaces/frame_images_DB/ extraído)
.venv/bin/python app/reverify_ytf_jeito12.py
# Ablação (alinhamento vs cor)
N_BASE_EACH=500 N_ATTACK=300 .venv/bin/python app/ablation_ytf.py
# Blindagem (referência separada) + figura
.venv/bin/python app/blindar_refs.py && .venv/bin/python app/figura_distribuicoes.py
# Figuras da monografia (ROC, ASR×limiar, e fig3 = tempo real por câmera virtual c/ emulada sobreposta) — só CPU, lê os CSVs
.venv/bin/python app/figuras_monografia.py
# Segundo verificador (FaceNet)
.venv/bin/python app/facenet_crosscheck.py
# Tempo real (webcam) — abre http://localhost:8080/
.venv/bin/python app/realtime_live_attack.py --source 0 \
  --source-image data/dataset_pessoal/originais/victorsorrindoclaro.jpeg \
  --ref-image    data/dataset_pessoal/originais/victorserioclaro.jpeg \
  --duration 60 --record app/results/demo.mp4
# Tempo real — NÚMERO OFICIAL: protocolo multi-sessão (15 identidades LFW, semente 42) + controles — ver GUIA §8.6
.venv/bin/python app/rodar_sessoes.py --modo ATAQUE      --prefixo atq --duration 30
.venv/bin/python app/rodar_sessoes.py --modo PASSTHROUGH --prefixo imp --duration 30
.venv/bin/python app/agg_sessoes_realtime.py "app/results/sessoes/atq_*.csv"  # ASR/sessão + IC Wilson
.venv/bin/python app/analise_controles_realtime.py                            # controle impostor (0/15)
.venv/bin/python app/controle_intraclasse.py                                  # validade (A-B genuína)
# Tempo real — INJEÇÃO POR CÂMERA VIRTUAL (v4l2loopback) — ver GUIA §8.7
bash setup_vcam.sh                                                            # cria /dev/video10 (precisa de sudo)
.venv/bin/python app/gerar_plano_vcam.py                                      # plano + validade (sessoes_plano_vcam.json, validade_vcam.txt)
.venv/bin/python app/rodar_sessoes_vcam.py --modo ATAQUE      --prefixo atq --duration 30 --source 0
.venv/bin/python app/rodar_sessoes_vcam.py --modo PASSTHROUGH --prefixo imp --duration 30 --source 0
.venv/bin/python app/agg_sessoes_vcam.py                                      # ASR/sessão + Wilson + pareado emulado×canal
```

## Estrutura

- **`app/`** — scripts do pipeline (ativos + dependências de geração de pares). Mapa completo e papel de cada um no **GUIA §3**.
- **`yolo8face_adaface/`** — wrapper do verificador (`AdaFaceVerifier`: detecção, alinhamento, embedding, verificação).
- **`app/results/`** — resultados finais (scores `*_jeito12`, blindado, ablação, FaceNet, figuras, sessões do tempo real).
- **`app/utils/`** — utilitários avulsos mantidos: `download_lfw.py` (obtenção do LFW) e `visualize_alignment.py` (figura dos 5 landmarks).
- **`GUIA_DO_PROJETO.md`** — documentação detalhada e autoritativa.

## Referências

- **AdaFace**: Kim, M. et al. "AdaFace: Quality Adaptive Margin for Face Recognition." CVPR, 2022.
- **FaceNet**: Schroff, F. et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering." CVPR, 2015.
- **LFW**: Huang, G. B. et al. "Labeled Faces in the Wild." UMass, 2007.
- **YouTube Faces (YTF)**: Wolf, L. et al. "Face Recognition in Unconstrained Videos with Matched Background Similarity." CVPR, 2011.
- **FaceFusion**: https://github.com/facefusion/facefusion
- **YOLOv8-Face**: https://github.com/akanametov/yolo-face
