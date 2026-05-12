# Lovelace: LLM arquivistico para classificacao documental

Documentacao geral de uso, instalacao, organizacao e reproducao dos experimentos do projeto Lovelace.

Este repositorio acompanha a construcao experimental de um LLM arquivistico voltado para classificacao documental. A motivação para construção do Modelo foi discutada no artigo **"Urdiduras para a preservacao digital: o uso de abordagens computacionais para construcao de LLM arquivista"**, publicado na revista *Acervo* em 2026. Argumenta-se que a classificacao arquivistica, especialmente em ambientes digitais, pode se beneficiar de modelos de linguagem especializados, desde que o uso computacional respeite cadeia de custodia, autenticidade, confiabilidade, rastreabilidade e validacao profissional.

Palavras-chave do projeto: preservação digital, classificação arquivística, cadeia de custódia, LoRA, QLoRA, PREMIS, RVL-CDIP e JurisTCU.





## Sumario rapido

- **RVL-CDIP OCR -> LLM**: pipeline de inferencia textual em documentos digitalizados, usando OCR e LLM local.
- **RVL-CDIP QLoRA**: pipeline de treinamento supervisionado com QLoRA sobre texto extraido por OCR.
- **JurisTCU TIPOPROCESSO**: pipeline juridica em portugues, sem OCR, usando texto nativo do `doc.csv`.
- **Lovelace_old_Sem_Supervisão_Humana**: projeto piloto, ainda em desenvolvimento, voltado a aprendizado sem supervisao humana.
- **Evidencias**: metricas, predicoes, prompts, configuracoes, checkpoints e relatorios ficam em `data/processed` e `docs`.
- **Artigo**: `docs/2771_publicar.pdf` apresenta o problema teorico: preservacao digital, classificacao arquivistica e adaptacao de LLM com LoRA.

## Problema arquivistico

A classificacao documental e uma tarefa estrutural da arquivologia. Ela permite situar o documento em seu contexto de producao, relacionar especies e tipos documentais a funcoes e atividades, subsidiar avaliacao documental e aplicar planos de classificacao e tabelas de temporalidade.

No ambiente digital, esse problema cresce em escala. Sistemas eletronicos produzem documentos em grande volume, muitas vezes classificados por usuarios que nao sao arquivistas. Isso aumenta o risco de erro na captura, na descricao e na atribuicao de categorias. O artigo que fundamenta este projeto argumenta que a preservacao digital nao se resume ao armazenamento: ela exige controle da infraestrutura tecnico-material, fixidez, metadados, trilhas auditaveis e procedimentos que sustentem autenticidade e confiabilidade.

O LLM arquivistico aqui documentado entra nesse ponto: ele explora modelos de linguagem para identificar padroes textuais em especies, tipos ou classes documentais. O foco esta na criacao de um procedimento replicavel, auditavel e ajustavel a dominios especificos, como documentos digitalizados do RVL-CDIP e textos juridicos do JurisTCU.

Neste repositorio, ha uma diferenca importante entre os experimentos principais e o piloto antigo. Os resultados descritos nas pipelines RVL-CDIP QLoRA e JurisTCU TIPOPROCESSO correspondem a **aprendizado supervisionado**, isto e, a adaptacao do modelo a partir de exemplos rotulados e avaliados contra classes conhecidas. Ja a pasta `Lovelace_old_Sem_Supervisão_Humana` registra um caminho paralelo e ainda nao concluido: um projeto piloto de aprendizado sem supervisao humana. Ele deve ser lido como etapa exploratoria em desenvolvimento, nao como resultado final comparavel aos dois LoRAs supervisionados.

## Organizacao do repositorio

Estrutura principal esperada:

```text
.
├── configs/
│   ├── rvl_class_map.json
│   ├── rvl_system_prompt*.txt
│   └── qlora/
├── data/
│   ├── raw/
│   │   └── JurisTCU_repo/
│   ├── rvl-cdip/
│   │   ├── images/
│   │   └── labels/
│   └── processed/
│       ├── saida-ocr-class/
│       ├── ocr-tests/
│       ├── qlora-qwen14b/
│       └── juristcu-tipoprocesso/
├── docs/
│   ├── 2771_publicar.pdf
│   ├── relatório parcial/
│   └── readmes/
├── Lovelace_old_Sem_Supervisão_Humana/
├── scripts/
│   ├── qlora/
│   └── juristcu_tipoprocesso_pipeline/
├── src/
│   ├── rvl_text_pipeline/
│   └── rvl_qlora_train/
├── requirements.txt
├── requirements-paddle.txt
└── README.md
```

### Pastas principais

| Pasta | Funcao |
| --- | --- |
| `src/rvl_text_pipeline` | Inferencia RVL-CDIP com OCR + LLM local via Ollama. |
| `src/rvl_qlora_train` | Construcao de datasets, OCR cacheado, treino QLoRA e avaliacao de LoRA para RVL-CDIP. |
| `scripts/juristcu_tipoprocesso_pipeline` | Preparacao, treino e avaliacao JurisTCU para `TIPOPROCESSO`. |
| `scripts/qlora` | Atalhos operacionais para rodadas RVL QLoRA e avaliacao manual estavel. |
| `configs` | Prompts, mapas de classe e configuracoes de QLoRA. |
| `data/raw` | Bases externas em estado bruto. Hoje inclui `JurisTCU_repo`. |
| `data/rvl-cdip` | Imagens TIFF e labels do RVL-CDIP. |
| `data/processed` | Resultados, datasets derivados, metricas, predicoes, logs, checkpoints e adaptadores. |
| `docs` | Relatorios, readmes historicos, tabelas, prompts, evidencias e artigo publicado. |
| `Lovelace_old_Sem_Supervisão_Humana` | Projeto piloto de aprendizado sem supervisao humana, ainda nao finalizado e separado das pipelines supervisionadas principais. |



## Instalacao

### 1. Ambiente Python basico

Use a raiz do projeto:

```bash
cd /home/arr/Documents/GitHub/Lovelace
```

Crie um ambiente virtual comum para scripts leves:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Esse ambiente e suficiente para partes simples da pipeline textual, mas nao cobre necessariamente QLoRA, PaddleOCR, CUDA, bitsandbytes ou transformers.

### 2. Ambiente QLoRA

Para treino e avaliacao com QLoRA, este projeto usa um ambiente separado:

```bash
cd /home/arr/Documents/GitHub/Lovelace
python3 -m venv .venv-qlora
source .venv-qlora/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-paddle.txt
```

Dependendo da maquina, tambem podem ser necessarias instalacoes especificas de:

- PyTorch com CUDA compativel;
- `transformers`;
- `peft`;
- `bitsandbytes`;
- `accelerate`;
- PaddleOCR e dependencias de OCR;
- drivers NVIDIA e toolkit CUDA compativel.

O projeto foi executado localmente em estacao com GPU. A reproducao em outra maquina pode exigir ajuste fino de versoes, memoria de GPU, batch size, `gradient_accumulation_steps`, `max_seq_len` e configuracoes 4-bit.

### 3. Ollama para inferencia direta RVL

A pipeline RVL-CDIP de inferencia direta usa modelo local via Ollama:

```bash
ollama serve
ollama pull qwen2.5:14b
```

Opcionalmente, tambem foram testados modelos como `qwen2.5:32b`, `qwq:32b` e `deepseek-r1:14b`, conforme os relatorios historicos em `docs/readmes`.

## Dados esperados

### RVL-CDIP

Estrutura esperada:

```text
data/rvl-cdip/
├── images/
└── labels/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

O RVL-CDIP contem imagens documentais em TIFF e 16 classes oficiais:

```text
0=letter
1=form
2=email
3=handwritten
4=advertisement
5=scientific report
6=scientific publication
7=specification
8=file folder
9=news article
10=budget
11=invoice
12=presentation
13=questionnaire
14=resume
15=memo
```

Nas fases QLoRA finais, a classe `handwritten` foi excluida do treino principal, resultando em 15 classes ativas. Essa decisao e metodologica: `handwritten` depende fortemente de sinal visual e legibilidade, o que reduz comparabilidade em uma pipeline textual baseada em OCR.

### JurisTCU

Estrutura esperada:

```text
data/raw/JurisTCU_repo/JurisTCU/doc.csv
```

A pipeline JurisTCU usa texto nativo, sem OCR. O campo de classificacao e `TIPOPROCESSO`. O texto de entrada e composto por:

```text
ENUNCIADO + "\n\n" + EXCERTO
```

O dataset final manteve apenas as 15 classes mais frequentes de `TIPOPROCESSO`, descartando linhas sem label e classes fora do Top 15.

Classes JurisTCU usadas:

```text
0=TOMADA DE CONTAS ESPECIAL
1=REPRESENTAÇÃO
2=APOSENTADORIA
3=RELATÓRIO DE AUDITORIA
4=PENSÃO CIVIL
5=PRESTAÇÃO DE CONTAS
6=CONSULTA
7=RELATÓRIO DE LEVANTAMENTO
8=DENÚNCIA
9=ADMINISTRATIVO
10=ATOS DE ADMISSÃO
11=SOLICITAÇÃO DO CONGRESSO NACIONAL
12=PRESTAÇÃO DE CONTAS SIMPLIFICADA
13=TOMADA DE CONTAS
14=MONITORAMENTO
```

## Pipeline 1: RVL-CDIP OCR -> LLM

Esta pipeline avalia classificacao textual sem treino supervisionado. Ela extrai OCR das imagens do RVL-CDIP e envia o texto para um LLM local via Ollama.

Ponto de entrada:

```bash
python -m src.rvl_text_pipeline.main --help
```

Exemplo basico:

```bash
cd /home/arr/Documents/GitHub/Lovelace
python -m src.rvl_text_pipeline.main \
  --labels-file data/rvl-cdip/labels/test.txt \
  --images-root data/rvl-cdip/images \
  --model qwen2.5:14b \
  --sample-size 100 \
  --seed 42 \
  --run-id run_exemplo_ocr_llm \
  --ocr-engine paddle \
  --paddle-device auto
```

Saidas:

```text
data/processed/saida-ocr-class/<run_id>/
├── manifests/
├── ocr/
├── predictions/
├── evaluation/
├── logs/
└── premis/
```

Arquivos importantes:

- `manifests/sample.jsonl`: amostra selecionada.
- `manifests/state.json`: estado da execucao.
- `ocr/*.txt`: texto OCR por documento.
- `predictions/predictions.jsonl`: predicoes por documento.
- `evaluation/metrics.json`: metricas agregadas.
- `premis/*.jsonl`: eventos e objetos para rastreabilidade.

### Interpretacao desta fase

Os primeiros resultados mostraram que a qualidade do sistema dependia muito do contrato de inferencia: prompt, schema de saida, parser, few-shot e recuperacao de respostas invalidas. A melhoria entre `run_001` e `run_002`, por exemplo, veio mais da estabilizacao do protocolo do que de troca de modelo.

Essa fase serviu para responder uma pergunta inicial: ate onde e possivel ir sem treinar o modelo, apenas com OCR, prompt e inferencia local? A resposta foi: e possivel obter um baseline util, mas o salto estrutural exigiu adaptacao supervisionada.

## Pipeline 2: RVL-CDIP QLoRA

Esta pipeline cria datasets textuais a partir do RVL-CDIP, executa ou reaproveita OCR, treina um adaptador LoRA com QLoRA e avalia o modelo adaptado.

Ponto de entrada:

```bash
python -m src.rvl_qlora_train.main --help
```

Rodada R1:

```bash
cd /home/arr/Documents/GitHub/Lovelace
bash scripts/qlora/run_round1.sh
```

Rodada R2, usada como melhor resultado RVL-CDIP da linha 14B:

```bash
cd /home/arr/Documents/GitHub/Lovelace
bash scripts/qlora/run_round2_qvko.sh
```

Comando expandido de referencia para R2:

```bash
cd /home/arr/Documents/GitHub/Lovelace
python -m src.rvl_qlora_train.main \
  --labels-train data/rvl-cdip/labels/train.txt \
  --labels-val data/rvl-cdip/labels/val.txt \
  --labels-test data/rvl-cdip/labels/test.txt \
  --images-root data/rvl-cdip/images \
  --base-model Qwen/Qwen2.5-14B-Instruct \
  --run-id qlora_r2_15cls_qvko_v1 \
  --sample-train-per-class 2000 \
  --sample-val-per-class 300 \
  --sample-test-per-class 300 \
  --seed 42 \
  --llm-gpu-index 0 \
  --ocr-engine paddle \
  --paddle-variant server_ch \
  --paddle-lang en \
  --paddle-latin-lang en \
  --paddle-device gpu \
  --paddle-angle-cls \
  --system-prompt-file configs/qlora/rvl_train_prompt_v1_minjson.txt \
  --qlora-config configs/qlora/qlora_qwen14b_r2_qvko_lowmem.yaml \
  --legacy-sample-manifest data/processed/saida-ocr-class/run_001/manifests/sample.jsonl \
  --max-ocr-chars 12000 \
  --min-ocr-chars 40 \
  --skip-auto-eval
```

Avaliacao manual estavel do LoRA RVL:

```bash
cd /home/arr/Documents/GitHub/Lovelace
.venv-qlora/bin/python scripts/qlora/eval_lora_infer_stable.py \
  --run-dir data/processed/qlora-qwen14b/qlora_r2_15cls_qvko_v1 \
  --splits eval_legacy_93,val_balanced,test_holdout_balanced \
  --gpu-index 0 \
  --vote-on-invalid-only
```

Saidas:

```text
data/processed/qlora-qwen14b/<run_id>/
├── datasets/
├── checkpoints/
├── evaluation_manual_stable_*/
├── logs/
└── reports/
```

Artefatos importantes:

- `datasets/*.jsonl`: datasets textuais para treino, validacao e teste.
- `datasets/dataset_summary.json`: resumo dos dados usados.
- `checkpoints/adapter_final/`: adaptador LoRA final.
- `checkpoints/training_summary.json`: resumo do treino.
- `evaluation_manual_stable_*/summary.json`: resumo da avaliacao.
- `evaluation_manual_stable_*/*/metrics.json`: metricas por split.
- `evaluation_manual_stable_*/*/per_class.csv`: metricas por classe.
- `evaluation_manual_stable_*/*/confusion_matrix.csv`: matriz de confusao.

## Pipeline 3: JurisTCU TIPOPROCESSO

Esta pipeline foi criada depois da consolidacao RVL-CDIP para testar transferencia metodologica em uma base juridica em portugues. Ao contrario do RVL-CDIP, ela nao usa OCR: o texto vem diretamente de `doc.csv`.

Pasta:

```text
scripts/juristcu_tipoprocesso_pipeline/
```

### 1. Preparar dataset

```bash
cd /home/arr/Documents/GitHub/Lovelace
.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/prepare_juristcu_dataset.py \
  --run-id juristcu_top15_qvko_v1
```

O script usa por padrao:

```text
data/raw/JurisTCU_repo/JurisTCU/doc.csv
```

Saidas:

```text
data/processed/juristcu-tipoprocesso/<run_id>/
├── datasets/train.jsonl
├── datasets/val.jsonl
├── datasets/test.jsonl
├── configs/class_map.json
├── configs/system_prompt.txt
└── reports/
```

Resumo do dataset final:

| Item | Valor |
| --- | ---: |
| Registros Top 15 usados | 15.109 |
| Treino | 7.554 |
| Validacao | 3.777 |
| Teste | 3.778 |
| Linhas sem `TIPOPROCESSO` | 227 |
| Linhas fora do Top 15 | 709 |
| Truncamento textual | 4.000 caracteres |

O split e feito por grupo de acordao, reduzindo risco de vazamento entre treino, validacao e teste.

### 2. Treinar LoRA JurisTCU

```bash
cd /home/arr/Documents/GitHub/Lovelace
.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/train_juristcu_qlora.py \
  --run-id juristcu_top15_qvko_v1 \
  --gpu-index 0
```

Caracteristicas principais:

- modelo base: `Qwen/Qwen2.5-14B-Instruct`;
- metodo: QLoRA 4-bit;
- LoRA em `q_proj`, `v_proj`, `k_proj`, `o_proj`;
- avaliacao durante treino desativada por padrao;
- checkpoints frequentes para retomada;
- adaptador final em `checkpoints/adapter_final`.

### 3. Avaliar LoRA JurisTCU

```bash
cd /home/arr/Documents/GitHub/Lovelace
.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/eval_juristcu_stable.py \
  --run-id juristcu_top15_qvko_v1 \
  --splits val,test \
  --gpu-index 0
```

Saidas:

```text
data/processed/juristcu-tipoprocesso/<run_id>/evaluation_manual_stable_rest_full_<timestamp>/
├── summary.json
├── pipeline.log
├── val/
└── test/
```

### 4. Avaliar modelo base sem LoRA

Este comando usa o mesmo dataset e o mesmo modelo base, mas sem carregar adaptador LoRA. Ele serve como baseline.

```bash
cd /home/arr/Documents/GitHub/Lovelace
.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/eval_juristcu_base_stable.py \
  --run-id juristcu_top15_qvko_v1 \
  --splits val,test \
  --gpu-index 0
```

Saida esperada:

```text
data/processed/juristcu-tipoprocesso/<run_id>/evaluation_manual_stable_base14b_rest_full_<timestamp>/
```

## Projeto piloto: aprendizado sem supervisao humana

A pasta `Lovelace_old_Sem_Supervisão_Humana` corresponde a uma linha anterior e ainda em desenvolvimento do projeto. Ela nao faz parte dos dois estudos supervisionados consolidados no README, mas precisa ser registrada porque representa uma tentativa paralela de construcao de LLM arquivistico sem depender diretamente de exemplos rotulados por supervisao humana.

Essa linha deve ser entendida assim:

- e um piloto;
- nao esta finalizada;
- nao deve ser comparada diretamente com os resultados RVL-CDIP QLoRA e JurisTCU LoRA;
- possui estrutura propria, com `scripts`, `data`, `model`, `offload`, `build` e `dist`;
- serve como registro de uma direcao experimental futura.

Portanto, o projeto tem hoje dois blocos metodologicos:

| Bloco | Estado | Metodo | Onde esta |
| --- | --- | --- | --- |
| RVL-CDIP QLoRA | Consolidado | Aprendizado supervisionado com LoRA/QLoRA | `src/rvl_qlora_train`, `scripts/qlora`, `data/processed/qlora-qwen14b` |
| JurisTCU TIPOPROCESSO | Consolidado | Aprendizado supervisionado com LoRA/QLoRA | `scripts/juristcu_tipoprocesso_pipeline`, `data/processed/juristcu-tipoprocesso` |
| Lovelace sem supervisao humana | Piloto em desenvolvimento | Aprendizado sem supervisao humana | `Lovelace_old_Sem_Supervisão_Humana` |

## Resultados principais

### RVL-CDIP R2 QLoRA

Fonte principal:

```text
data/processed/qlora-qwen14b/qlora_r2_15cls_qvko_v1/
```

Resumo:

| Item | Valor |
| --- | ---: |
| Modelo base | `Qwen/Qwen2.5-14B-Instruct` |
| Adaptador | LoRA `q,v,k,o` |
| Parametros treinaveis | 50.331.648 |
| Razao treinavel | 0,6127% |
| Exemplos de treino tokenizados | 27.477 |
| Exemplos de validacao tokenizados | 4.135 |
| `max_seq_len` | 1.536 |
| Tempo de treino | ~18h38m41s |
| `train_loss` | 0,1438 |
| `val_eval_loss` | 0,1609 |

Avaliacao final:

| Split | N | Accuracy | Macro-F1 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `eval_legacy_93` | 80 | 90,0000% | 83,6042% | 100,0000% |
| `val_balanced` | 4.135 | 88,3676% | 87,4778% | 100,0000% |
| `test_holdout_balanced` | 4.094 | 87,6160% | 86,5998% | 100,0000% |

Interpretacao: o R2 consolidou o melhor resultado textual da linha RVL-CDIP com Qwen 14B. O ganho frente ao R1 apareceu tanto em subset legado quanto nos splits grandes.

### JurisTCU LoRA

Fonte principal:

```text
data/processed/juristcu-tipoprocesso/juristcu_top15_qvko_v1/
```

Resumo:

| Item | Valor |
| --- | ---: |
| Modelo base | `Qwen/Qwen2.5-14B-Instruct` |
| Adaptador | LoRA `q,v,k,o` |
| Parametros treinaveis | 50.331.648 |
| Razao treinavel | 0,6127% |
| Registros Top 15 | 15.109 |
| Split | 7.554 / 3.777 / 3.778 |
| `max_seq_len` | 1.024 |
| Tempo de treino | ~4h38m16s |
| `train_loss` | 0,6613 |
| Epoca | 1,0 |

Avaliacao final com LoRA:

| Split | N | Accuracy | Macro-F1 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `val` | 3.777 | 82,7906% | 63,7151% | 100,0000% |
| `test` | 3.778 | 82,7157% | 62,7094% | 100,0000% |

Baseline sem LoRA:

| Split | N | Accuracy | Macro-F1 | Coverage |
| --- | ---: | ---: | ---: | ---: |
| `val` | 3.777 | 60,8903% | 51,8888% | 99,9206% |
| `test` | 3.778 | 59,3965% | 48,8476% | 100,0000% |

Interpretacao: o LoRA JurisTCU produziu ganho expressivo sobre o modelo base no mesmo protocolo de avaliacao. A accuracy global ficou estavel entre validacao e teste, mas o macro-F1 menor indica que classes raras ainda sao mais dificeis.

## Rastreabilidade e evidencias

O projeto foi documentado com preocupacao de auditoria, em dialogo com preservacao digital. As principais evidencias ficam em:

```text
docs/readmes/
docs/relatório parcial/
docs/prompts_conteudo/
docs/comprovacao_artigo_csv_json_prompts_*/
data/processed/
```

Evidencias recorrentes:

- `metrics.json`: metricas por split.
- `summary.json`: resumo consolidado de avaliacao.
- `training_summary.json`: resumo de treino e adaptador.
- `dataset_summary.json`: volume e filtros do dataset.
- `class_map.json`: mapa de classes.
- `predictions.jsonl` e `predictions.csv`: predicoes por documento.
- `per_class.csv`: metricas por classe.
- `confusion_matrix.csv`: matriz de confusao.
- `pipeline.log`: log operacional.
- `premis/*.jsonl`: eventos, objetos, agentes e direitos no padrao PREMIS.



## Prompts e contratos de inferencia

Um aprendizado central do projeto foi que o desempenho nao depende apenas do tamanho do modelo. A forma do contrato de inferencia importa muito.

No RVL-CDIP, os primeiros prompts pediam JSON mais rico, com justificativas e campos adicionais. Isso aumentou erros de validacao. A evolucao posterior simplificou a resposta, reforcou o mapa de classes, estabilizou o parser e reduziu a saida esperada. No treino QLoRA, o prompt passou a exigir apenas o id numerico da classe.

No JurisTCU, a instrucao tambem foi minimalista:

```text
Voce e um classificador de documentos juridicos do TCU.
Tarefa: classificar o texto em exatamente uma classe TIPOPROCESSO.
Regras:
- Use somente o texto fornecido.
- Retorne apenas o id numerico da classe.
- Nao retorne palavras, JSON ou explicacoes.
```

Esse desenho reduz ambiguidade, facilita avaliacao automatica e aproxima treino e inferencia.

## Interpretacao cientifica

A trajetoria do projeto pode ser lida em tres fases.

Primeiro, a fase de inferencia direta mostrou que OCR, prompt e parser determinam boa parte do comportamento inicial. A melhoria de schema e few-shot reduziu erros tecnicos e aumentou cobertura.

Segundo, a fase QLoRA mostrou que o treinamento supervisionado gera salto estrutural quando a pipeline textual ja esta minimamente estabilizada. No RVL-CDIP, o R2 superou o R1 e os melhores baselines 14B anteriores.

Terceiro, a transferencia para JurisTCU demonstrou que o mesmo padrao tecnico pode ser adaptado a documentos juridicos em portugues. Como os textos sao nativos, sem OCR, o problema fica mais diretamente semantico. O resultado forte em accuracy e a diferenca entre accuracy e macro-F1 mostram ao mesmo tempo potencial e cautela: o modelo reconhece bem classes frequentes, mas classes minoritarias exigem analise especializada.

Em paralelo, a pasta `Lovelace_old_Sem_Supervisão_Humana` indica uma direcao ainda aberta: investigar classificacao e organizacao arquivistica com menor dependencia de rotulos humanos previamente definidos. Essa linha ainda nao foi finalizada e, por isso, aparece nesta documentacao como piloto, nao como evidencia experimental fechada.

Do ponto de vista arquivistico, a conclusao e deliberadamente prudente: o LLM arquivistico apoia a classificacao, acelera triagem, registra evidencias e pode reduzir gargalos, mas a homologacao final deve continuar com profissionais e instituicoes responsaveis.

## Cuidados operacionais

- Nao rode treino completo sem conferir espaco em disco.
- Nao misture runs antigos e novos com o mesmo `run_id`, a menos que esteja retomando um checkpoint de proposito.
- Para JurisTCU, rode `prepare` antes de `train`.
- Para RVL QLoRA, garanta que labels e imagens estejam disponiveis.
- Para inferencia RVL com Ollama, garanta que `ollama serve` esteja ativo.
- Para QLoRA, confira GPU, CUDA, PyTorch, `bitsandbytes` e memoria disponivel.
- Nao apague `adapter_final` se quiser reutilizar o LoRA.
- Nao use metricas de denominadores diferentes como se fossem comparacoes diretas.

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Execute comandos a partir da raiz do projeto:

```bash
cd /home/arr/Documents/GitHub/Lovelace
```

Para RVL QLoRA, prefira:

```bash
python -m src.rvl_qlora_train.main --help
```

### `Run root not found`

O `run_id` informado ainda nao existe em `data/processed`. Prepare ou treine o run antes de avaliar.

### `Adapter not found`

A avaliacao LoRA procura:

```text
checkpoints/adapter_final/
```

Se essa pasta nao existe, o treino nao terminou ou o run_id esta incorreto.

### Erro de OCR vazio

No RVL-CDIP, alguns documentos podem gerar OCR vazio ou curto. A pipeline registra esses casos como erro ou filtra exemplos abaixo de `min_ocr_chars`.

### OOM ou falta de memoria GPU

Reduza `max_seq_len`, batch efetivo, workers de OCR, ou use configuracoes `lowmem`. Tambem e recomendavel separar treino e avaliacao em processos diferentes, como foi feito nas rodadas finais.

## Relatorios internos

Os documentos mais importantes para entender a evolucao experimental sao:

```text
docs/readmes/history.md
docs/readmes/history_addendum.md
docs/readmes/history_addendum_v3.md
docs/readmes/history_addendum_v4.md
docs/readmes/history_addendum_v5.md
docs/readmes/history_addendum_v6.md
docs/relatório parcial/history_scientific_unified_rvl_juristcu_v1.html
docs/2771_publicar.pdf
```

Leitura recomendada:

1. `docs/2771_publicar.pdf`: problema teorico e justificativa arquivistica.
2. `docs/readmes/history.md`: baseline OCR -> LLM.
3. `docs/readmes/history_addendum_v3.md`: recorte sem `handwritten`.
4. `docs/readmes/history_addendum_v5.md`: QLoRA R1.
5. `docs/readmes/history_addendum_v6.md`: QLoRA R2.
6. `docs/relatório parcial/history_scientific_unified_rvl_juristcu_v1.html`: sintese RVL-CDIP + JurisTCU.

## Comandos de verificacao rapida

Sem rodar treino ou avaliacao longa:

```bash
cd /home/arr/Documents/GitHub/Lovelace

python -m src.rvl_text_pipeline.main --help
python -m src.rvl_qlora_train.main --help

.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/prepare_juristcu_dataset.py --help
.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/train_juristcu_qlora.py --help
.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/eval_juristcu_stable.py --help
.venv-qlora/bin/python scripts/juristcu_tipoprocesso_pipeline/eval_juristcu_base_stable.py --help
```

Verificar presenca dos dados principais:

```bash
test -f data/rvl-cdip/labels/test.txt
test -d data/rvl-cdip/images
test -f data/raw/JurisTCU_repo/JurisTCU/doc.csv
```

## Citacao e contexto academico

Este repositorio acompanha o experimento apresentado no artigo:

```text
ROTA, Alesson Ramon. Urdiduras para a preservacao digital:
o uso de abordagens computacionais para construcao de LLM arquivista.
Acervo, Rio de Janeiro, v. 39, n. 1, p. 1-29, jan./abr. 2026.
DOI: https://doi.org/10.64729/an.acervo.v39i1.2771
```


