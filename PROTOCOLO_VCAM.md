# Protocolo do experimento de injeção por câmera virtual

> **Este arquivo é o registro prévio do protocolo.** Ele deve ser commitado ANTES da
> primeira sessão ser gravada. O horário do commit é a evidência de que as regras
> abaixo foram fixadas sem conhecimento dos resultados.

## 1. O que o experimento acrescenta

O protocolo atual (15 sessões em `app/results/sessoes/`) mede a aceitação **dentro do
laço de execução**: o quadro manipulado é entregue diretamente à verificação, e o
alinhamento reaproveita os pontos fiduciais que o detector do FaceFusion obteve no
quadro original. Duas consequências ficam declaradas como limitações na monografia:

1. o fluxo não atravessa um dispositivo de câmera virtual, então os efeitos do canal
   (conversão de formato, reamostragem, perda de quadros) não são exercidos;
2. nas sessões de ataque, o YOLOv8-Face do sistema-alvo não chega a ver o deepfake.

Este experimento remove as duas. O ataque passa a ser entregue por `/dev/video10`
(v4l2loopback) e consumido por um processo independente, que executa o pipeline
completo do sistema-alvo: YOLOv8-Face detecta e alinha, AdaFace extrai, a similaridade
é comparada ao limiar.

## 2. Arquitetura

    webcam /dev/video0
        -> Processo A (atacante): FaceFusion inswapper_128  -> /dev/video10
        -> Processo B (sistema-alvo): YOLOv8-Face + AdaFace -> log por quadro

O processo B não recebe nenhuma informação do processo A: lê o dispositivo de vídeo
como leria uma webcam. O campo `mode` gravado no log de B é apenas rótulo de
bookkeeping, não entra em nenhum cálculo.

O processo A continua medindo a similaridade **no laço**, como no protocolo atual.
Cada sessão produz, portanto, duas medidas do mesmo ataque no mesmo instante:
**emulada** (A) e **pelo canal** (B). A comparação entre elas é pareada por sessão e
isola o efeito do canal, sem depender das sessões gravadas em outro dia.

## 3. Decisões fixadas antes da coleta

### 3.1 Identidades

- A amostra continua sendo a sorteada com semente 42 e congelada em
  `app/results/sessoes_plano.json`, que **não é alterado**: é ele que documenta as
  sessões já reportadas na monografia.
- O critério de validade é aplicado **antes** da gravação, e não depois:
  uma identidade é válida quando a similaridade genuína entre suas duas fotografias
  (fotoA x fotoB) é **>= 0,40**. O número de rostos em cada fotografia é registrado
  como diagnóstico, mas não invalida por si só (há identidades com fotografia
  multi-rosto em que a detecção selecionou a pessoa correta).
- Pelo critério acima, `Steve_Waugh` (A-B = 0,10) é declarado inválido **antes** da
  coleta. Ele não é gravado e não entra em nenhuma estatística.
- **Substituta:** o sorteio original é *continuado*, e não refeito. Como
  `random.Random(42).sample(nomes, k)` produz os mesmos primeiros 15 elementos para
  qualquer k >= 15, a substituta é a **16ª extração do mesmo sorteio**; se ela
  reprovar no critério de validade, segue-se para a 17ª, e assim por diante.
  O script `app/gerar_plano_vcam.py` verifica primeiro que as 15 primeiras extrações
  reproduzem exatamente o `sessoes_plano.json`; se não reproduzirem, ele aborta e o
  conjunto fica com 14 identidades.
- O resultado é gravado em `app/results/sessoes_plano_vcam.json`.

### 3.2 Regra de decisão da sessão

Igual à do protocolo atual: a sessão é **aceita** quando a **mediana** das
similaridades de seus quadros atinge o limiar de operação herdado da LFW,
FAR 1% = 0,1562 (também se reporta FAR 0,1% = 0,2289). A unidade estatística é a
sessão; o intervalo de confiança é o de Wilson.

### 3.3 Quadros sem rosto detectado

Esta é a única regra nova, porque agora o detector do sistema-alvo pode falhar sobre o
quadro manipulado. Ambas as contas são registradas aqui, antes da coleta:

- **Regra principal:** a mediana é calculada **apenas sobre os quadros em que o
  YOLOv8-Face detectou rosto**, e a **taxa de detecção por sessão** é reportada junto.
  É o comportamento de um sistema real: sem rosto, não há decisão, apenas o próximo
  quadro.
- **Análise de sensibilidade:** a mesma conta tratando quadro sem detecção como
  **rejeitado** (similaridade abaixo de qualquer limiar). Reporta-se se a conclusão
  muda.

O log de B traz, por quadro, as colunas `detectado` e `n_rostos`, de modo que as duas
contas saem do mesmo CSV, sem nova execução.

### 3.4 Quadros repetidos

A câmera virtual entrega sempre o último quadro escrito. Se B ler mais rápido do que A
escreve, o mesmo quadro aparece duas vezes. B compara o *hash* dos bytes de cada quadro
com o do anterior e marca `duplicado = 1`; quadros duplicados são registrados no CSV e
descartados na agregação.

### 3.5 Demais parâmetros (idênticos ao protocolo atual)

- `fotoA` = fonte do *swap*; `fotoB` = referência de verificação (referência blindada).
- 30 segundos por sessão, 640x480.
- Mesma sequência de movimentos: rotação da cabeça para os lados, inclinação vertical e
  abertura da boca.
- 15 sessões de ataque e 15 de controle (PASSTHROUGH: rosto real da atacante, sem
  manipulação, entregue pelo mesmo canal). Primeiro o bloco de ataque inteiro, depois o
  de controle, como no protocolo atual.
- Durante a gravação, a atacante vê numa janela o vídeo **enviado** à câmera virtual (o
  deepfake no ataque, o rosto cru no controle), **sem indicação de similaridade, de
  aceito/rejeitado ou de qualquer resultado** — como nas sessões antigas. É só
  visualização: lê uma cópia do quadro, não altera o que é enviado nem entra em nenhum
  cálculo.

### 3.6 Janela da sessão

O processo A sobe primeiro e só é encerrado quando B termina. Os 30 segundos são
contados por B, a partir do primeiro quadro que ele recebe; a sequência de movimentos
começa quando B anuncia `recebendo quadros`. Assim a janela observada pelo sistema-alvo
fica inteira dentro da execução do atacante, independentemente do tempo de carga dos
modelos.

### 3.7 Comparação pareada emulado x canal

Análise **secundária e descritiva**. Para cada sessão, a mediana emulada (A) é calculada
apenas sobre os quadros de A escritos dentro da janela de B (coluna `t_abs`, tempo
absoluto, presente nos dois logs). Reporta-se a diferença por sessão e o teste de
Wilcoxon pareado **bilateral**. A conclusão sobre a vulnerabilidade vem da regra 3.2
aplicada ao log de B, não desta comparação.

### 3.8 Falhas técnicas e regravação

Uma sessão só pode ser regravada por **falha técnica**, definida sem olhar a
similaridade: (a) o log de B não foi criado, (b) B recebeu menos de 100 quadros únicos,
ou (c) um dos processos terminou com erro. Nesse caso o log é movido para
`app/results/sessoes_vcam/descartadas/` (não apagado), o motivo é anotado em
`app/results/sessoes_vcam/descartadas/MOTIVOS.txt`, e a sessão é gravada de novo.
Nenhuma sessão é regravada por ter similaridade baixa ou alta.

## 4. Execução

    bash setup_vcam.sh                          # cria /dev/video10 e lista os dispositivos
    .venv/bin/pip install pyvirtualcam          # e acrescentar ao requirements.txt
    .venv/bin/python app/gerar_plano_vcam.py    # gera sessoes_plano_vcam.json
    # commitar o plano e este arquivo ANTES de gravar
    # --source = número da webcam mostrado pelo setup_vcam.sh
    .venv/bin/python app/rodar_sessoes_vcam.py --modo ATAQUE      --prefixo atq --duration 30 --source 0
    .venv/bin/python app/rodar_sessoes_vcam.py --modo PASSTHROUGH --prefixo imp --duration 30 --source 0
    .venv/bin/python app/agg_sessoes_vcam.py

Antes da coleta completa, dois testes rápidos, gravados em outra pasta para não se
misturarem com a coleta (`--prefixo teste_imp` e `--prefixo teste_atq`, com a
identidade que for): uma sessão de controle (a similaridade deve ficar próxima de zero,
como nas sessões atuais) e uma de ataque. Os testes servem só para checar o
funcionamento e não entram em nenhuma estatística; depois deles, mova os logs de teste
para `app/results/sessoes_vcam/testes/`.

## 5. Saídas

- `app/results/sessoes_vcam/<prefixo>_NN_<Identidade>_A.csv` — log do atacante
  (medida emulada, no laço).
- `app/results/sessoes_vcam/<prefixo>_NN_<Identidade>_B.csv` — log do sistema-alvo
  (medida pelo canal; é o **resultado oficial** deste experimento).
- `app/results/vcam_sessoes.csv` e `app/results/vcam_resultados.txt` — agregação.

## 6. O que não muda

`app/realtime_live_attack.py`, `app/rodar_sessoes.py`, `app/agg_sessoes_realtime.py`,
`app/results/sessoes/` e `app/results/sessoes_plano.json` permanecem intactos: são o
registro do protocolo já reportado na monografia.
