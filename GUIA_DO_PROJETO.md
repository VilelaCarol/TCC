# Guia do Projeto — Red-team de Verificação Facial contra Deepfake

> Documento de referência para retomar o projeto. **Verificado contra o estado real do repositório em 15/11/2026** (imports conferidos por `grep`, versões lidas do `.venv`, números lidos dos arquivos em `app/results/`); **experimento por câmera virtual acrescentado e conferido em 16/11/2026** (números de `vcam_resultados.txt`, `vcam_sessoes.csv`, `validade_vcam.txt`). Onde algo é incerto, está marcado como **(incerto)**.

---

## 1. Visão geral

O projeto é um **red teaming** (avaliação de vulnerabilidade) de um sistema de **verificação facial** contra ataques de **deepfake por face swap injetado no laço de captura** (duas condições ao vivo: **emulada** no laço e por **câmera virtual** v4l2loopback — esta **executada**; ver §8.7, §7 e `PROTOCOLO_VCAM.md`).

- **Sistema-vítima:** `YOLOv8-Face` (detecção + 5 landmarks) → `AdaFace IR-50` (embedding facial de 512 dimensões, treinado no WebFace4M) → **similaridade cosseno** vs. um limiar.
- **Ataque:** `FaceFusion / inswapper_128` troca o rosto da vítima no quadro do atacante (deepfake).
- **Métrica:** **ASR (Attack Success Rate)** = fração de ataques cuja similaridade ≥ limiar (o sistema "aceita" o deepfake como sendo a vítima).

Três cenários, hoje unificados na mesma metodologia:

| Cenário | O que é | Arquivo que roda |
|---|---|---|
| **LFW** | foto × foto (benchmark acadêmico) | `app/reverify_lfw_jeito12.py` |
| **YTF** | vídeo × vídeo (YouTube Faces, protocolo de média de frames) | `app/reverify_ytf_jeito12.py` |
| **Real-time** (emulada) | webcam ao vivo, ASR medido dentro do loop | `app/realtime_live_attack.py` |
| **Real-time** (câmera virtual) | webcam → FaceFusion → `/dev/video10`, lido por processo independente (YOLOv8-Face+AdaFace) | `app/rodar_sessoes_vcam.py` (ver §8.7) |

---

## 2. O que mudou e por quê (narrativa das decisões)

### 2.1. Reconstrução do ambiente (máquina nova RTX 4050 Laptop, 6 GB)
A máquina antiga (RTX A5500, 24 GB) e duas pastas (`yolo8face_adaface/`, `facefusion/`) foram perdidas. O ambiente foi reconstruído do zero, **confirmado no `.venv`**:
- Python **3.11.16**, `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124` (CUDA 12.4, cuDNN 9.1.0), `onnxruntime-gpu 1.24.4`.
- **FaceFusion fixado na tag 3.6.1** (clonado em `facefusion/`; a API muda em 3.7+).
- Wrapper `yolo8face_adaface/` reconstruído do zero (backbone AdaFace oficial + YOLOv8-Face), sem depender de HuggingFace/transformers.
- `onnxruntime.preload_dlls()` no import acha as libs CUDA/cuDNN do torch; `pre_check()` baixa os modelos do FaceFusion.

### 2.2. Conserto do gargalo do real-time (detecção tripla → única)
A versão antiga rodava **3 detectores por frame** (YOLO avulso + detecção interna do FaceFusion + YOLO dentro do AdaFace), estourando os 6 GB. A versão nova reaproveita a **detecção única do FaceFusion** (bbox + 5 landmarks) tanto para o swap quanto para a verificação. Com isso, a verificação AdaFace custa ~4 ms e o **ASR passou a ser medido AO VIVO, dentro do loop** — não mais offline (salvar frames e calcular depois), que foi um ponto da revisão do protocolo.

### 2.3. "Jeito 1" vs "Jeito 2" (a mesma régua)
- **Jeito 1** = recorte cru da bounding box + canais **RGB**. É o pré-processamento legado do pipeline original.
- **Jeito 2** = **alinhamento ArcFace de 5 pontos** (transformação de similaridade para o template canônico 112×112) + canais **BGR**. É o pré-processamento que o AdaFace espera.

LFW, YTF e real-time usavam pré-processamentos diferentes, então os ASRs **não eram comparáveis**. A unificação no Jeito 2 colocou os três na mesma régua. Efeito medido (`*_thresholds_jeito12.txt`): a genuína sobe (ex.: LFW 0.429→0.619), a do ataque sobe (LFW 0.459→0.750) e o impostor cai — ou seja, **separa melhor as classes**, não é truque para inflar o ASR.

### 2.4. Ablação do YTF (alinhamento vs. cor)
Para saber qual variável causou o salto do ASR, uma ablação 2×2 nas mesmas pares (`app/results/ablation_ytf.txt`):

| Condição | EER | ASR@0.60 |
|---|---|---|
| A — cru+RGB (jeito 1) | 9.20% | 23.3% |
| B — cru+BGR | 8.50% | 28.0% |
| C — alinhado+RGB | 5.40% | 88.3% |
| **D — alinhado+BGR (jeito 2)** | **4.90%** | **92.0%** |

- **Alinhamento é o motor:** +64 a +65 pontos de ASR@0.60.
- **Cor é pequena, mas BGR é a correta:** menor EER que RGB em todas as condições → confirma empiricamente a convenção do AdaFace. O Jeito 2 **não** está com a cor trocada.
- **A baseline YTF original era subótima:** seu EER era **23.56%** vs. **4.2%** agora. Até o nosso recorte *sem* alinhamento (EER 9.2%) supera o `aligned_images_DB` original (23.56%) — ele usa um template de alinhamento diferente do que o AdaFace espera. O salto reflete **consertar uma baseline fraca**, não inflar o ataque.

### 2.5. Controles negativos do real-time
Dúvida legítima: "deu tudo certo, não é bug?" Controle em imagens estáticas (`scratchpad`, reproduzível) mostrou, em média:
- deepfake vs. **outra foto** da vítima = **0.736** (perto da genuína real dela, 0.78) → reconstrói a **identidade**, não é self-match;
- deepfake vs. **pessoa real** da câmera = 0.078; vs. **impostor** = 0.062; **passthrough** (rosto real vs. vítima) = 0.019.
→ O AdaFace **rejeita** corretamente impostor e rosto real, e só é enganado pelo deepfake. O ~93% ao vivo é **ataque real**.

### 2.6. Blindagem da referência + calibração por FAR
- **Blindagem:** a referência de verificação passou a ser uma **imagem/vídeo DIFERENTE** da usada como fonte do swap (antes era a mesma → inflava a similaridade). Resultado em `app/results/blindado_lfw_ytf.txt`.
- **Por que abandonar 0.45/0.60 fixos:** nesses limiares a aceitação de usuários **genuínos** (TPR) seria **LFW 70.7% @0.60** e **YTF 45.5% @0.60** — ou seja, rejeitariam 29–55% dos legítimos. Não são ponto de operação válido.
- **Número headline (blindado + calibrado por FAR@1%):** **LFW ≈ 91.8%**, **YTF ≈ 87.7%** (e 91.3% / 84.3% em FAR@0.1%). É o que sobrevive às críticas de self-match e de limiar arbitrário.

---

## 3. Mapa de arquivos (confirmado por `grep` de imports)

### 3.1. FUNDAMENTAIS (o coração hoje)
- **`yolo8face_adaface/src/__init__.py`** — classe `AdaFaceVerifier`. Carrega o AdaFace; métodos `detect()` (bbox+5 landmarks), `get_embedding(img, bbox=, landmarks=, align=)` e `verify()`. O flag `align` escolhe Jeito 2 (padrão, `True`) ou Jeito 1 (`False`).
- **`yolo8face_adaface/src/utils.py`** — detector YOLOv8-Face (singleton), `align_face()` (5 pontos → `ARCFACE_REF_112`), `preprocess_aligned()` (BGR, Jeito 2), `preprocess_image()` (crop+RGB, Jeito 1), `get_similarity()` (cosseno).
- **`yolo8face_adaface/src/net.py`** — backbone IR-50 do AdaFace (`build_model('ir_50', mode='ir')`).
- **`yolo8face_adaface/configs/config.yaml`** — config do verificador (device, checkpoint, pesos YOLO, limiares 0.45/0.60). Não tem chave `align`, então o padrão é Jeito 2.
- **`app/neural_face_swap.py`** — motor de swap via FaceFusion inswapper_128. Expõe `NeuralFaceSwapper` (`swap_and_locate()` devolve frame+bbox+landmarks) e `swap_pair(src, tgt)` (swap in-process, modelos ficam na VRAM → ~0.12 s/swap no LFW).

### 3.2. ATIVOS (rodam o pipeline atual)
- **`app/reverify_lfw_jeito12.py`** — LFW nos dois jeitos: regenera 3000 swaps (in-process) + baseline 6000 pares + ROC/EER/FAR + figura.
- **`app/reverify_ytf_jeito12.py`** — YTF nos dois jeitos (protocolo vídeo-a-vídeo, média de 5 frames, detecção+alinhamento por frame nos frames crus).
- **`app/ablation_ytf.py`** — ablação 2×2 (alinhamento × cor), reusa swaps do disco.
- **`app/blindar_refs.py`** — re-verifica LFW e YTF com referência **separada** (blindada).
- **`app/figura_distribuicoes.py`** — gera a figura das distribuições self vs. blindado.
- **`app/figuras_monografia.py`** — gera as 3 figuras da monografia (`fig1_roc`, `fig2_asr`, `fig3_tempo_real`) a partir dos CSVs já versionados; só CPU, reprodutível (limiares recalculados dos scores).
- **`app/realtime_live_attack.py`** — ataque real-time na webcam, ASR ao vivo, HUD, servidor MJPEG, gravação, log CSV.
- **`app/rodar_sessoes.py`** — executor do protocolo multi-sessão: lê `sessoes_plano.json` e roda as sessões (ATAQUE/PASSTHROUGH) chamando `realtime_live_attack.py` por subprocess, uma identidade por vez.
- **`app/agg_sessoes_realtime.py`** — agrega as sessões de ataque: ASR por **sessão** (mediana ≥ limiar LFW) + IC de Wilson.
- **`app/analise_controles_realtime.py`** — agrega os controles impostor (frames PASSTHROUGH): % de sessões aceitas nos limiares do LFW (esperado ~0%).
- **`app/controle_intraclasse.py`** — valida cada sessão: similaridade genuína A-B da vítima (fotoA×fotoB) + nº de rostos por foto (diagnostica referências corrompidas).

### 3.3. DEPENDÊNCIAS (não rodo direto, mas os ativos IMPORTAM — não são lixo)
Fornecem geração de pares reprodutível (seed=42) e funções legadas. Também são, em tese, entry points do pipeline **original** (swaps por `subprocess`, Jeito 1), mas hoje servem como biblioteca.
- **`app/baseline_lfw.py`** → `parse_pairs`, `ensure_lfw_images`, `download_file`, `LFW_DIR`, `PAIRS_FILE`, `FACE_SIZE` e funções legadas (`preprocess_face`, `get_embedding`, `detect_and_crop_face`, `_get_verifier`). Importado por `reverify_lfw_jeito12`, `blindar_refs`, `run_swaps_lfw`, `baseline_ytf`.
- **`app/baseline_ytf.py`** → `get_ytf_identities`, `generate_pairs` (genuíno/impostor, seed), `YTF_DATA_DIR`, `NUM_GENUINE_PAIRS=2500`, `NUM_IMPOSTOR_PAIRS=2500`, `FRAMES_TO_SAMPLE=5`. Importado por `reverify_ytf_jeito12`, `ablation_ytf`, `blindar_refs`, `run_swaps_ytf`. **Nota:** `YTF_DATA_DIR` foi repontado para `data/YouTubeFaces/frame_images_DB` (frames crus).
- **`app/run_swaps_lfw.py`** → `get_lfw_identities`, `generate_swap_pairs` (seed), `NUM_SWAPS=3000`, `RANDOM_SEED=42`. Importado por `reverify_lfw_jeito12`, `blindar_refs`.
- **`app/run_swaps_ytf.py`** → `generate_attack_pairs` (seed), `NUM_ATTACKS=1000`. Importado por `reverify_ytf_jeito12`, `ablation_ytf`, `blindar_refs`.

### 3.4. REMOVIDOS nesta versão limpa (eram obsoletos/superados; nada ativo os importava)
> Estes arquivos foram **removidos do repositório** para não confundir o leitor (continuam existindo no repositório original `DeepFake-as-a-Spoofing-Attack` e no histórico). Lista do que era e por quê:
- **`app/realtime_attack_pipeline.py`** — real-time antigo auto-contido (detecção tripla → OOM). Superado por `realtime_live_attack.py`.
- **`app/realtime_api_server.py`** — servidor WebSocket/MJPEG antigo; tinha a verificação removida e salvava frames (ASR offline). Superado.
- **`app/local_virtual_cam.py`** — cliente WebSocket + `pyvirtualcam` da arquitetura distribuída (par do `realtime_api_server`). Superado, mas o `pyvirtualcam` pode ser reaproveitado se um dia for feita a **injeção em Zoom/Meet** (câmera virtual).
- **`app/compute_asr_offline.py`** — cálculo de ASR **offline** (ponto da revisão do protocolo). Superado pela medição ao vivo + `reverify_*`.
- **`app/api.py`** — servidor FastAPI `/verify` usando funções Jeito 1. Não faz parte do pipeline atual.
- **`app/download_ytf.py`** — baixa `aligned_images_DB` (alinhamento do próprio YTF). Superado: usamos `frame_images_DB` (frames crus) baixado manualmente.
- **`app/net.py`** — cópia do backbone; **não é importado por ninguém** (o ativo usa `yolo8face_adaface/src/net.py`). Duplicata.
- **`app/analyze_pose_similarity.py`** — análise de pose antiga; não importada. (aux/antiga)
- **`app/generate_paper_tables.py`** — gera tabelas do paper a partir dos resultados **antigos** (Jeito 1); superado pelos outputs `*_jeito12` / blindado.
- **`app/test_video_similarity.py`** — teste ad-hoc antigo; não importado.

**Limpeza adicional (após `292ed0c`, análise de dependências):** removidos também os restos do pipeline Docker/notebook, sem referência ativa — `pipeline_lfw_adaface.ipynb`, `pipeline_deepfake.ipynb`, `update_notebooks.py`, `test_ff.py`, `Dockerfile`, `docker-compose.gpu.yml`, `docker-compose.cpu.yml`, `passo_a_passo_projeto.txt` (continha os números ANTIGOS 92.8%/34.3%, substituídos pelos blindados/calibrados) e 7 scripts órfãos de `app/utils/` (`benchmark_fps`, `compare_faces`, `generate_paper_plots`, `swap_faces`, `run_experiment`, `run_swaps_experiment`, `test_gpu`). Em `app/utils/` ficaram só `download_lfw.py` (obtenção do LFW) e `visualize_alignment.py` (figura dos 5 landmarks). O `requirements.txt` foi curado para os imports reais (saíram `transformers`, `fastapi`, `uvicorn`, `gdown`; `numpy<2.0` corrigido p/ 2.2.1).

### 3.5. EXPERIMENTO POR CÂMERA VIRTUAL (v4l2loopback) — ver §8.7
Registro prévio e scripts da injeção por câmera virtual (dois processos independentes):
- **`PROTOCOLO_VCAM.md`** — registro prévio do protocolo (commit `f45f69a`, ANTES da coleta); **não editar depois da coleta** (seria adulteração).
- **`setup_vcam.sh`** — carrega o módulo `v4l2loopback` e cria `/dev/video10` (câmera virtual de saída do atacante). Substitui o antigo `setup_realtime.sh` (OBS), movido para `legado/`.
- **`app/gerar_plano_vcam.py`** — gera `app/results/sessoes_plano_vcam.json` + `validade_vcam.txt`: aplica o critério A-B ≥ 0,40 e **continua** o sorteio da semente 42 para a substituta.
- **`app/vcam_atacante.py`** (Processo A) — webcam → FaceFusion inswapper_128 → `/dev/video10`; mede a similaridade **emulada** no laço (log `*_A.csv`). Flags `--preview` e `--ocultar-similaridade`.
- **`app/vcam_verificador.py`** (Processo B) — lê `/dev/video10` com o próprio **YOLOv8-Face + AdaFace**; log `*_B.csv` (**resultado oficial**); colunas `detectado`, `n_rostos`, `duplicado` (regras §3.3/§3.4 do protocolo).
- **`app/rodar_sessoes_vcam.py`** — executor: sobe A e B por sessão; passa `--preview` e `--ocultar-similaridade` por padrão.
- **`app/agg_sessoes_vcam.py`** — agrega por sessão (mediana ≥ limiar; Wilson; sensibilidade; pareado emulado×canal com Wilcoxon) → `vcam_resultados.txt`, `vcam_sessoes.csv`.
- **`app/results/sessoes_vcam/`** — logs: **oficiais na raiz** (`atq_*`, `imp_*`), **testes em `testes/`** (fora da análise).

---

## 4. Como rodar (comandos exatos)

Sempre a partir da raiz do repositório. O `.venv` já tem tudo; use `.venv/bin/python` direto (não precisa ativar), ou `source .venv/bin/activate`.

| Comando | Gera | Tempo aprox. | Pico VRAM |
|---|---|---|---|
| `.venv/bin/python app/reverify_lfw_jeito12.py` | `lfw_scores_jeito12.csv`, `lfw_swap_attack_jeito12.csv`, `lfw_thresholds_jeito12.txt`, `lfw_jeito1_vs_jeito2.png`; swaps em `data/lfw/swaps/` | ~7–15 min (baixa LFW ~233 MB e regenera 3000 swaps na 1ª vez; reusa depois) | ~2.7 GB |
| `.venv/bin/python app/reverify_ytf_jeito12.py` | `ytf_*_jeito12.*`, `ytf_jeito1_vs_jeito2.png`; swaps em `data/YouTubeFaces/swaps/` | ~12 min | ~2.7 GB |
| `N_BASE_EACH=500 N_ATTACK=300 .venv/bin/python app/ablation_ytf.py` | `ablation_ytf.txt`, `ablation_spotcheck/` | ~8 min (reusa swaps, sem FaceFusion) | ~0.5 GB |
| `.venv/bin/python app/blindar_refs.py` | `blindado_lfw_ytf.txt`, `lfw_attack_blindado_scores.csv`, `ytf_attack_blindado_scores.csv` | ~5–8 min | ~0.5 GB |
| `.venv/bin/python app/figura_distribuicoes.py` | `distribuicoes_blindado.png` | ~seg | — (só CPU, lê os CSVs) |
| `.venv/bin/python app/figuras_monografia.py` | `fig1_roc.png`, `fig2_asr.png`, `fig3_tempo_real.png` (fig3 = câmera virtual c/ emulada sobreposta) | ~seg | — (só CPU, lê os CSVs) |
| `bash setup_vcam.sh` | `/dev/video10` (v4l2loopback) | ~seg (precisa de `sudo`) | — |
| `.venv/bin/python app/gerar_plano_vcam.py` | `sessoes_plano_vcam.json`, `validade_vcam.txt` | ~1–3 min | ~2.7 GB |
| `.venv/bin/python app/rodar_sessoes_vcam.py --modo ATAQUE\|PASSTHROUGH --prefixo atq\|imp --duration 30 --source 0` | `app/results/sessoes_vcam/{atq,imp}_*_{A,B}.csv` | ~30 s/sessão | ~3.0 GB (ataque) / ~0.8 GB (controle) |
| `.venv/bin/python app/agg_sessoes_vcam.py` | `vcam_resultados.txt`, `vcam_sessoes.csv` | ~seg | — (só CPU) |

**Pré-requisito do YTF:** `data/YouTubeFaces/frame_images_DB/<Pessoa>/<vídeo>/<frame>.jpg` extraído (o `frame_images_DB.tar.gz` ~11.9 GB, frames CRUS — **não** o `aligned_images_DB`). Os scripts YTF **não** baixam isso sozinhos.

**Real-time (webcam):**
```bash
.venv/bin/python app/realtime_live_attack.py \
  --source 0 \
  --source-image data/dataset_pessoal/originais/victorsorrindoclaro.jpeg \
  --ref-image    data/dataset_pessoal/originais/victorserioclaro.jpeg \
  --duration 60 \
  --record app/results/demo_realtime_blindado.mp4
```
Gera `app/results/realtime_live_log.csv` + o `.mp4`. **8–11 FPS, pico VRAM ~2.7 GB.** Abra `http://localhost:8080/` no navegador para ver o HUD ao vivo.

Flags:
- `--source` — `0` = webcam `/dev/video0`, ou caminho de um vídeo.
- `--source-image` — rosto que será **injetado** (a vítima que o atacante quer imitar); é a fonte do swap.
- `--ref-image` — foto de referência da vítima **DIFERENTE** da fonte do swap (blindagem; sem isso, usa a própria `--source-image` = self, inflado).
- `--record` — salva o vídeo com HUD; `--duration N` para após N s; `--max-frames N`; `--swap-every N` (swap a cada N frames); `--width/--height` (padrão 640×480); `--port` (padrão 8080); `--mode ATAQUE|PASSTHROUGH`.
- Teclas (só com display gráfico): `m` alterna ATAQUE/PASSTHROUGH, `q` sai.

**Protocolo multi-sessão** (15 sessões + controle impostor + validação intraclasse): ver **§8.6**.

---

## 5. Resultados: onde estão e o que significam

Todos em `app/results/`. **Distinção crucial:**
- **`self`** = referência era a mesma imagem do swap → **inflado, NÃO reportar.**
- **`blindado`** = referência separada → **válido, reportar.**

### Arquivos novos (desta fase)
- `lfw_scores_jeito12.csv`, `ytf_scores_jeito12.csv` — baseline (genuíno/impostor) com score nos 2 jeitos.
- `lfw_swap_attack_jeito12.csv`, `ytf_swap_attack_jeito12.csv` — ataque, similaridade nos 2 jeitos (coluna `*_j2` = **self**).
- `lfw_attack_blindado_scores.csv`, `ytf_attack_blindado_scores.csv` — similaridade do ataque com referência **blindada**.
- `lfw_thresholds_jeito12.txt`, `ytf_thresholds_jeito12.txt` — EER/FAR + ASR por jeito (self).
- `blindado_lfw_ytf.txt` — ASR **blindado** em todos os limiares.
- `ablation_ytf.txt` — tabela 2×2 da ablação.
- `tabela_final_blindada.txt` — tabela consolidada dos três (self vs. blindado) + leitura.
- `distribuicoes_blindado.png`, `lfw_jeito1_vs_jeito2.png`, `ytf_jeito1_vs_jeito2.png` — figuras auxiliares.
- `fig1_roc.png`, `fig2_asr.png`, `fig3_tempo_real.png` — figuras da monografia (geradas por `figuras_monografia.py`, reprodutíveis dos CSVs).
- Os vídeos `*.mp4` do real-time **não** vão para o repositório (privacidade: mostram pessoas identificáveis) — ficam só locais. A sessão única antiga (`realtime_live_log.csv`) foi movida para `legado/`; o tempo real oficial são as `sessoes/` abaixo (ver §8.6).
- `sessoes_plano.json` — as 15 identidades do LFW sorteadas (semente 42) para o protocolo multi-sessão (registro da amostra).
- `sessoes/atq_*.csv` (ataque) e `sessoes/imp_*.csv` (controle impostor) — log por frame de cada sessão.
- `controle_intraclasse.txt` — similaridade genuína A-B de cada vítima + nº de rostos por foto (diagnóstico de validade das sessões).
- `controles_realtime.txt` — agregado dos controles impostor (0/15 aceitas).
- `sessoes_plano_vcam.json`, `validade_vcam.txt` — plano e log de validade do experimento por câmera virtual (§8.7): Steve_Waugh invalidado (A-B 0,100), substituto Amanda_Beard (17ª extração, 0,743).
- `sessoes_vcam/atq_*`, `sessoes_vcam/imp_*` (oficiais) + `sessoes_vcam/testes/` (testes) — log por quadro de A (emulado) e B (canal) da injeção por câmera virtual.
- `vcam_resultados.txt`, `vcam_sessoes.csv` — agregação do experimento por câmera virtual (ASR/sessão, Wilson, sensibilidade, pareado emulado×canal).

### Arquivos ANTIGOS — REMOVIDOS nesta versão limpa
Os resultados do pipeline original (Jeito 1 / offline) foram **removidos** para não confundir: `lfw_scores.csv`, `ytf_scores.csv`, `lfw_swap_attack_results.csv`, `ytf_swap_attack_results.csv`, `realtime_asr_offline.csv`, `realtime_asr_summary.txt`, `realtime_attack_log.csv`, `*_conclusion.txt`, `experimental_config.txt`, plots antigos e as pastas `pose_analysis/`, `paper_plots/`, `realtime_frames/`. Ainda estão no repositório original e no histórico Git, se precisar.

### Números-chave finais (ler assim)
| Cenário | ASR self (não reportar) | **ASR blindado @FAR1%** | ASR blindado @FAR0.1% |
|---|---|---|---|
| **LFW** | 94.9% | **91.8%** | 91.3% |
| **YTF** | 98.1% | **87.7%** | 84.3% |
| **Real-time** (emulada) | — | **100%** (14/14 sessões válidas) @ limiar herdado do LFW | 100% |
| **Real-time** (câmera virtual) | — | **100%** (15/15) @ limiar LFW; controle 0/15; detecção 100% | 100% |

Leitura do real-time (protocolo multi-sessão EXECUTADO — ver §8.6): a unidade estatística é a **sessão**, não o frame (frames são autocorrelacionados, ρ=0.866, n_eff≈26). Rodamos **15 sessões** contra 15 identidades do LFW sorteadas com semente fixa (42, congeladas em `sessoes_plano.json`). Regra de decisão fixada **antes** da coleta: sessão aceita se a **mediana** das similaridades ≥ limiar de operação herdado do LFW (0.1562 @FAR1%, 0.2289 @FAR0.1%).
- **Ataque:** 14/15 = **93.3%** (IC95% Wilson 70.2–98.8) bruto. A única "falha" (Steve_Waugh) é **inválida**: a foto de referência tem 2 rostos e a detecção pegou a pessoa errada (similaridade genuína A-B = 0.10), comprovado por `controle_intraclasse.py`. Excluindo-a: **14/14 = 100%** (IC95% Wilson 78.5–100).
- **Controle impostor** (PASSTHROUGH: rosto real do atacante vs. vítima, 15 sessões): **0/15 aceitas** → o verificador rejeita corretamente o rosto real (especificidade; `controles_realtime.txt`).

O antigo "44% @ 0.698" era artefato de um limiar *overfit* do conjunto pessoal de 8 pessoas (separação perfeita → 0.698 não é ponto de operação).

---

## 6. Real-time passo a passo (fluxo de UM frame)

Tudo em `app/realtime_live_attack.py`, função `run()`, salvo indicação.

1. **Captura 480p** — `cap.read()` + `cv2.resize(frame, (args.width, args.height))`.
2. **Swap + detecção única** — a cada `args.swap_every` frames chama `swapper.swap_and_locate(frame)` → **`neural_face_swap.py` → `NeuralFaceSwapper.swap_and_locate()`**: `get_many_faces([frame])` (detecção interna do FaceFusion, 1×), `swap_face(source_face, target_face, frame)` (inswapper_128), e devolve `(frame_processado, bbox, landmarks5)` reaproveitando `target_face.bounding_box` e `target_face.landmark_set['5']`. Entre swaps, reusa o último (`last_swap/last_bbox/last_lmk`).
3. **Alinhamento + embedding** — `verifier.get_embedding(processed, bbox=bbox, landmarks=lmk)` → **`src/__init__.py → AdaFaceVerifier.get_embedding()`**: com `align=True` (padrão), usa os landmarks → `align_face()` (`src/utils.py`, 5 pontos → 112×112) → `preprocess_aligned()` (BGR, [-1,1]) → `self.model` → embedding 512-d normalizado.
4. **Similaridade** — `get_similarity(emb, ref_emb)` (cosseno), onde `ref_emb` vem de `verifier.get_embedding()` da `--ref-image` (blindada) ou `--source-image` (self).
5. **Veredito** — `sim >= 0.45` e `sim >= 0.60`; acumula `match_low/match_high` sobre `attack_frames` → ASR ao vivo.
6. **Saídas** — `draw_hud()` desenha MODE/FPS/ASR/Sim/veredito/bbox; `MJPEGServer.push()` publica em `http://localhost:8080/`; `writer.write()` grava o `.mp4`; `csv_w.writerow()` registra o frame em `realtime_live_log.csv`. No fim, `run()` imprime o relatório com ASR(τ=0.45) e ASR(τ=0.60).

---

## 7. Limitações declaradas (colocar no texto)

- **Cadastro pessoal pequeno:** 8 pessoas, 277 pares de impostor → FAR@0.1% não é confiável nesse conjunto e o limiar 0.698 é sensível à amostra. O ASR robusto vem de LFW/YTF.
- **YTF blindagem:** dos 1000 ataques, **384 foram descartados** porque a pessoa-fonte tinha só um vídeo (sem como separar a referência); restaram **616**.
- **Ambiente antigo perdido:** datasets e swaps foram regenerados numa única pilha consistente → os números offline antigos **não são diretamente comparáveis** (motivo da unificação).
- **Real-time = protocolo multi-sessão** (15 sessões, unidade = sessão, IC de Wilson; ver §8.6). Responde à crítica do ASR offline e já tem IC. Amostra ainda modesta (n=15) → ampliar para IC mais estreito.
- **Detecção multi-face:** pega-se a face de maior confiança. Isso **corrompeu 1 referência** no protocolo multi-sessão (Steve_Waugh: `fotoB` com 2 rostos → referência de outra pessoa, A-B=0.10), **detectada e excluída com justificativa** por `controle_intraclasse.py`. Nas 14 válidas as referências têm A-B 0.54–0.79. Fotos com >1 rosto exigem seleção da face correta.
- **Injeção por câmera virtual — EXECUTADA (ver §8.7).** O cenário ao vivo tem duas condições: a **emulada** no laço (quadro manipulado entregue direto à verificação, alinhamento reaproveitando os pontos do FaceFusion) e a **por câmera virtual (v4l2loopback)**, em que um processo **independente** do sistema-alvo lê `/dev/video10` com o próprio **YOLOv8-Face** — o alvo **vê o deepfake** e o quadro **atravessa o canal**. Resultado: **15/15 ataques, 0/15 controle, detecção 100%**, e a medida pelo canal é indistinguível da emulada (diferença ≤ 0,005/sessão, Wilcoxon p=0,76). **Limitação remanescente:** o canal é **local** (v4l2loopback), **sem a recompressão de vídeo com perdas** que um Zoom/Meet/WhatsApp aplicaria; a robustez do ataque sob compressão de canal fica em aberto.

---

## 8. Adendo pós-revisão (terminologia, descartes, controles, 2º verificador)

### 8.1. Convenção de terminologia (#6)
Para a monografia, padronizar: a **vítima** (identidade personificada) é a **fonte do rosto** no swap; o **atacante** é quem aparece no quadro/câmera de destino. **Atenção:** no código, `run_swaps_lfw.py` chama de `source` a identidade personificada (vítima), mas `run_swaps_ytf.py` chama `source` de "atacante" e `target` de "vítima" — invertido nos comentários. **Não renomeei o código** (quebraria a geração de pares por seed e a reprodutibilidade); a convenção acima vale para o texto. Fixe uma frase no método e use sempre.

### 8.2. Descartes e distribuição de identidades (#7)
- **YTF (blindagem):** dos 1000 ataques sorteados, **616 usados / 384 descartados** (fonte com um único vídeo → sem 2º vídeo para referência separada). **Viés a declarar:** os mantidos têm mais vídeos por fonte (**2.80 vs 2.11** na média); 37.1% das pessoas do YTF têm só um vídeo. Ou seja, a blindagem favorece identidades com mais gravações.
- **LFW:** `MIN_IMAGES_PER_PERSON = 2` garante a 2ª foto para a referência separada → **0 descartes**.

### 8.3. Controles do lado do ataque (#8) — `app/controles_ataque.py` → `controles_ataque.txt`
Mesmo deepfake (LFW, jeito 2), três referências; limiar FAR@1% = 0.1562:

| Comparação | sim média | % ≥ FAR@1% |
|---|---|---|
| (1) deepfake vs **vítima/SRC** (ataque) | 0.516 | **91.8%** |
| (2) deepfake vs **alvo/TGT** (sobra da identidade do quadro) | 0.052 | 6.9% |
| (3) deepfake vs **identidade aleatória** (impostor) | 0.005 | 1.1% |

→ O swap **transfere** a identidade da vítima, **remove** a do alvo e **não casa** com identidades quaisquer (≈ taxa de impostor). O ASR alto é identidade-específico.

### 8.4. Segundo verificador — FaceNet (#9) — `app/facenet_crosscheck.py` → `facenet_crosscheck.txt`
O inswapper condiciona o swap num embedding ArcFace e o AdaFace tem espaço de identidade próximo — ameaça à validade. Repetindo a verificação do LFW com **FaceNet (InceptionResnet/VGGFace2, outra família)**, no **pipeline nativo do FaceNet** (MTCNN → 160×160 RGB + prewhiten; 0 faces não detectadas), ataque blindado:

| Verificador | genuína | impostor | EER | **ASR@FAR1%** | ASR@FAR0.1% |
|---|---|---|---|---|---|
| AdaFace (ArcFace-family) | 0.619 | 0.003 | 6.5% | **91.8%** | 91.3% |
| **FaceNet (VGGFace2)** | 0.728 | 0.023 | 5.17% | **87.9%** | 83.7% |

→ O ataque **transfere** para outra família (87.9% @FAR1%). A pequena queda vs AdaFace é o "bônus de mesma família" plausível (gerador ArcFace ↔ verificador ArcFace), mas a vulnerabilidade **não é artefato** disso. **Correção pós-revisão:** a 1ª tentativa alimentava o FaceNet com o recorte ArcFace 112→160 (fora da spec), inflando o EER para 6.73%; agora usa o MTCNN nativo (EER 5.17%). `facenet-pytorch==2.6.0` instalado com `--no-deps` (não mexe no torch 2.6).

### 8.5. Correções factuais aplicadas (pós-revisão)
- **Checkpoint:** o modelo final é **AdaFace IR-50 WebFace4M** (`data/models/adaface_ir50_webface4m.ckpt`, local). Textos antigos diziam "CVLFace/MS1MV2 via HuggingFace" — **errado**, corrigido no README e nos prints de `baseline_lfw.py`/`baseline_ytf.py`.
- **Real-time:** o número oficial é o **protocolo multi-sessão** (15 sessões LFW, semente 42; `agg_sessoes_realtime.py` + `analise_controles_realtime.py`), no limiar **herdado do LFW** (não o 0.698 overfit) — ver §8.6. A sessão única antiga (`realtime_asr_oficial.py`, 357 quadros do conjunto pessoal) foi movida para `legado/`.
- **`setup_realtime.sh` → `legado/`:** o setup do pipeline antigo (câmera virtual via **OBS**, pedia `video_nr=0,2`) não é referenciado por nenhum script ativo (só um comentário histórico em `setup_vcam.sh`); movido para `legado/` com `git mv`. O experimento por câmera virtual usa `setup_vcam.sh` + v4l2loopback direto (§3.5, §8.7).

### 8.6. Protocolo multi-sessão do real-time (#3) — EXECUTADO
Como os frames são autocorrelacionados (n_eff≈26), a unidade estatística é a **sessão**.

**Passo 0 — amostra congelada:** sorteio de **15 identidades do LFW** com semente fixa (42) entre as que têm ≥2 fotos, salvo em `app/results/sessoes_plano.json`. Para cada uma: `fotoA` (`_0001`) = fonte do swap, `fotoB` (`_0002`) = referência blindada. A semente fixa torna a amostra reprodutível e afasta a suspeita de escolha a dedo.

**Passo 1 — regra de decisão (fixada ANTES da coleta):** sessão aceita se a **mediana** das similaridades dos frames ≥ limiar de operação herdado do LFW (FAR1% = 0.1562).

**Execução** (uma identidade por vez, ENTER entre elas):
```bash
# ataque: deepfake da vítima injetado no rosto do atacante
.venv/bin/python app/rodar_sessoes.py --modo ATAQUE --prefixo atq --duration 30
# controle impostor: rosto REAL do atacante (sem swap) vs. referência da vítima
.venv/bin/python app/rodar_sessoes.py --modo PASSTHROUGH --prefixo imp --duration 30
```
**Análise:**
```bash
.venv/bin/python app/agg_sessoes_realtime.py "app/results/sessoes/atq_*.csv"  # ASR por sessão + IC Wilson
.venv/bin/python app/analise_controles_realtime.py                            # controle impostor
.venv/bin/python app/controle_intraclasse.py                                  # validade (A-B genuína + nº rostos)
```

**Resultados (15 sessões):**
- **Ataque:** 14/15 = **93.3%** (Wilson 70.2–98.8) bruto; 1 sessão inválida (Steve_Waugh, referência multi-rosto, A-B=0.10); **14/14 = 100%** (Wilson 78.5–100) excluindo-a.
- **Impostor:** **0/15 aceitas** — especificidade confirmada (o ASR alto não é verificador frouxo).
- **Intraclasse:** vítimas válidas com A-B genuína 0.54–0.79; o deepfake recupera **66–92%** dela. As duas mais fracas (Greg_Rusedski 0.377, Stephen_Friedman 0.399) são **transferência parcial estável** (std ~0.02; foto-fonte não-frontal), ainda **acima** do ponto de operação.

Scripts: `rodar_sessoes.py` (executor), `agg_sessoes_realtime.py` (ASR/Wilson), `analise_controles_realtime.py` (impostor), `controle_intraclasse.py` (validade). Flag `--log` do `realtime_live_attack.py`.

### 8.7. Experimento por câmera virtual (v4l2loopback) — EXECUTADO
Fecha as duas limitações da condição emulada (o alvo não via o deepfake; o fluxo não atravessava um dispositivo de câmera). Protocolo **pré-registrado** em `PROTOCOLO_VCAM.md` (commit `f45f69a`, antes de qualquer gravação; o horário do commit é a evidência de que as regras foram fixadas sem conhecer os resultados).

**a) Arquitetura (dois processos independentes).**
- **Processo A (atacante, `app/vcam_atacante.py`):** webcam `/dev/video0` → FaceFusion inswapper_128 → escreve o quadro manipulado em `/dev/video10` (v4l2loopback). Mede, no laço, a similaridade contra a referência blindada (medida **emulada**, log `*_A.csv`).
- **Processo B (sistema-alvo, `app/vcam_verificador.py`):** lê `/dev/video10` como leria uma webcam, roda o pipeline completo (**YOLOv8-Face** detecta e alinha → **AdaFace** → cosseno), sem saber do atacante. Log `*_B.csv` = **resultado oficial** (medida **pelo canal**). As duas medidas, por sessão, formam a comparação pareada que mede o efeito conjunto do canal e da troca de detector (FaceFusion → YOLOv8-Face).

**b) Plano (semente 42 continuada; §3.1 do protocolo).** Reusa o `sessoes_plano.json`, mas aplica o critério de validade (similaridade genuína fotoA×fotoB ≥ 0,40) **antes** da coleta. `Steve_Waugh` é invalidado (A-B = **0,100**); a substituta sai **continuando** o mesmo sorteio: 16ª extração `Kamal_Kharrazi` **reprovada** (A-B = **−0,023**), 17ª `Amanda_Beard` **aceita** (A-B = **0,743**). Conjunto final: 15 identidades válidas (`sessoes_plano_vcam.json`, `validade_vcam.txt`).

**c) Resultados (15 ataque + 15 controle; `vcam_resultados.txt`, `vcam_sessoes.csv`).** Unidade = sessão; aceita se a mediana dos quadros detectados ≥ limiar LFW (FAR1% = 0,1562).
- **Ataque:** **15/15 = 100%** (IC95% Wilson **79,6–100**); @FAR0,1% também 15/15. Medianas por sessão (canal) **0,379–0,637** (a menor, `atq_05 Greg_Rusedski` = 0,379, acima do limiar).
- **Controle impostor (PASSTHROUGH):** **0/15 = 0%** (IC95% Wilson **0–20,4**). Medianas **−0,105 a 0,076**, todas abaixo do limiar.
- **Detecção:** **100%** nas 30 sessões; **0 quadros duplicados**; **11.803 quadros** com exatamente 1 rosto.
- **Separação ataque × controle:** Cohen's **d = 8,19**; Mann-Whitney exato **unilateral U = 225, p = 6,4e-9** (separação completa — mesma convenção unilateral da condição emulada; o Wilcoxon do pareado é bilateral porque ali não há direção esperada).
- **Regra de sensibilidade (§3.3, sem detecção = rejeitado):** idêntica à principal, pois a detecção foi 100%.
- **Emulado × canal (§3.7, pareado, descritivo):** mediana emulado 0,588 vs canal 0,590; diferença **≤ 0,005 por sessão**; Wilcoxon pareado bilateral **p = 0,76** → a medida pelo canal fica **sem diferença detectável** em relação à emulada (no máximo 0,005 por sessão). As duas limitações declaradas **não alteraram o resultado**.

**d) Desvios e ocorrências (transparência).**
- Rodei os **blocos de teste completos** (15 de cada) em vez de uma sessão: `teste_imp` **com a similaridade visível no console** (antes do `dfd2baa`) e `teste_atq` já com o console oculto. São testes (§4/§5 do protocolo), movidos para `sessoes_vcam/testes/` e **fora da análise**.
- Commit **`dfd2baa`** (`--ocultar-similaridade`) feito **depois** do registro prévio e **antes** da coleta oficial: oculta a similaridade no **console** de A e B; **nenhuma regra alterada** e **nenhuma coluna do CSV muda** (cumpre o §3.5).
- `pyvirtualcam` instalado com **`uv pip`** (o `.venv` é `uv`, sem `pip`); versão fixada no `requirements.txt`.
- **Nenhuma sessão regravada**; nenhuma falha técnica (§3.8) nas 30 oficiais.

---

## Índice (1 linha por seção)
1. **Visão geral** — o que o projeto faz e os três cenários (LFW/YTF/real-time).
2. **O que mudou e por quê** — ambiente novo, conserto do real-time, Jeito 1 vs 2, ablação, controles, blindagem + FAR.
3. **Mapa de arquivos** — FUNDAMENTAIS, ATIVOS, DEPENDÊNCIAS, OBSOLETOS (confirmado por imports).
4. **Como rodar** — comandos exatos, o que geram, tempo e VRAM.
5. **Resultados** — arquivos em `app/results/`, self (não reportar) vs blindado (reportar), números finais.
6. **Real-time passo a passo** — fluxo de um frame com arquivo/função de cada etapa.
7. **Limitações** — cadastro pequeno, 384 YTF pulados, ambiente perdido, PoC, multi-face.
