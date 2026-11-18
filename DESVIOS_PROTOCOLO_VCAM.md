# Desvios em relação ao protocolo registrado (câmera virtual)

Este arquivo foi escrito **depois** da coleta. O protocolo pré-registrado,
`PROTOCOLO_VCAM.md` (commit `f45f69a`, 16/11/2026 18:42), não foi alterado: ele é a
evidência de que as regras foram fixadas antes dos resultados. Os desvios abaixo
estão declarados também na monografia.

## 1. Testes de funcionamento rodados como blocos inteiros (§4)

O protocolo previa dois testes rápidos antes da coleta: uma sessão de controle e uma
de ataque. Por engano de execução, os dois blocos foram rodados por inteiro (15
sessões de controle e 15 de ataque). Todos os registros estão preservados em
`app/results/sessoes_vcam/testes/` e foram excluídos de toda estatística, como o
protocolo determina para os testes de funcionamento.

Linha do tempo (16/11/2026, horário local):

| Horário | Evento |
|---|---|
| 18:42 | registro do protocolo (`f45f69a`) |
| 20:29–20:42 | bloco de teste de **controle**: similaridade ainda impressa no console |
| 20:55 | ocultação da similaridade no console (`dfd2baa`), sem alterar nenhuma regra |
| 21:04–21:21 | bloco de teste de **ataque**, já com a similaridade oculta |
| a partir de 21:28 | coleta oficial (ataque e depois controle), com a similaridade oculta |

Consequência: na condição por câmera virtual, a atacante nunca viu a similaridade de
uma sessão de ataque. No teste de controle ela viu a similaridade do próprio rosto
real, sem manipulação.

## 2. Análises não previstas no protocolo (§3.2 e §3.7)

O protocolo define o intervalo de confiança de Wilson (§3.2) e o teste de Wilcoxon
pareado bilateral como análise secundária e descritiva (§3.7). O teste exato de
Mann-Whitney unilateral e o d de Cohen não constam do protocolo; foram aplicados da
mesma forma que na condição emulada, já reportada antes desta coleta, para manter as
duas condições comparáveis. A comparação entre as coletas emulada e por câmera
virtual (correlação e diferença das medianas) também não consta do protocolo e é
reportada apenas como descritiva.

## 3. Erro factual no §3.5

O §3.5 diz que, durante a gravação, a atacante não vê indicação de similaridade
"como nas sessões antigas". A segunda parte está errada: na condição emulada
(`app/results/sessoes/`), a interface exibia a similaridade e um veredito durante as
sessões. A regra do §3.5 vale para a condição por câmera virtual e foi cumprida na
coleta oficial (ver item 1). A monografia descreve corretamente as duas condições e
trata o retorno à atacante na condição emulada como ameaça à validade.
