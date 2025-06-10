# Ada-Lovelace LoRA 🇧🇷 Arquivologia e Língua Portuguesa

LoRA (~80 MB) aprimorada sobre **[Vicuna-7B v1.5]** para melhorar respostas em arquivologia, especialmente para aplicação de Planos de Classificação e Tabelas de Temporalidades produzidos pelo Estado de São Paulo

![banner](https://img.shields.io/badge/LoRA-Vicuna7B-blue)
![license](https://img.shields.io/badge/license-MIT-%2B-NC-green)

---

## Objetivo

Este modelo foi treinado com materiais selecionados do Arquivo Público do Estado de São Paulo, visando aprimorar a precisão na classificação de tipos de processo e fornecer respostas especializadas em arquivologia.

## Uso rápido

```bash
git clone https://github.com/<usuario>/ada-lovelace-lora.git
cd ada-lovelace-lora
pip install -r requirements.txt

# precisa já ter o modelo base:
ollama pull vicuna:7b

# cria o modelo final no Ollama
ollama create ada-lovelace -f Modelfile

# teste:
ollama run ada-lovelace "Explique a diferença entre prontuário funcional e social."
```

## Estrutura dos diretórios

```
Lovelace/
├── data/
│   ├── hf_dataset/               # Dataset final no formato Hugging Face (arrow, state.json, etc)
│   ├── logs/                     # Logs de processamento, extração, treinamento
│   ├── txt_clean/                # Textos extraídos e limpos dos PDFs (um .txt por documento)
│   └── README.md                 # Documentação dos dados
│
├── model/
│   ├── ada-lovelace-lora/        # Checkpoints do modelo LoRA treinado e tokenizer customizado
│   │   └── checkpoint-3/         # Checkpoint mais recente (weights, optimizer, trainer states, etc)
│   ├── Modelfile                 # Arquivo com receita para gerar modelo Ollama customizado
│   └── README.md                 # Instruções/model card do modelo custom
│
├── scripts/
│   ├── books_corpus.py           # (Exemplo) Script de manipulação/conversão de corpus
│   ├── build_dataset.py          # Script para criar dataset Hugging Face a partir dos .txt limpos
│   ├── teste.py                  # Script para testar o modelo customizado (inference)
│   └── train_lora.py             # Script de treinamento LoRA 
│
├── .gitattributes                # Configurações do Git (por exemplo, para tratar LF/CRLF, linguagens)
├── logs_resumidos.json           # Logs resumidos ou estatísticas de treinamento/processamento
└── README.md                     # Documentação principal do repositório
```

## Dados
Ler  `data/README.md`.

## Treinamento (LoRA)

| Parâmetro   | Valor                     |
| ----------- | ------------------------- |
| Base        | vicuna-7b-v1.5            |
| Épocas      | 1                         |
| SeqLen      | 1024                      |
| Rank (LoRA) | 16                        |
| LR          | 1e-4 (cosine)             |
| Hardware    | i9-13900H · RTX 4060 8 GB |
| Tempo       | 22 min GPU / 2 h CPU      |

### Passos:

```bash
python scripts/process_pdfs.py data/pdf_raw data/txt_clean
python scripts/build_dataset.py
python scripts/train_lora.py --dataset_dir hf_dataset  # GPU
# ou python scripts/train_lora.py                  # CPU
```

### Aplicação de tese:
Código produz uma interação de pergunta e resposta
```bash
python scripts/test.py
```

### Modelfile

```
FROM vicuna:7b
ADAPTER ada-lovelace-lora
```

## Resultados

| Prompt                          | Vicuna-7B                   | Ada-Lovelace                           |
| ------------------------------- | --------------------------- | -------------------------------------- |
| "prontuário funcional × social" | respostas genéricas         | resposta com vocabulário especializado |

Taxa de acerto Vicuna-7B: 60%, 70%, 60%  
Taxa de acerto Ada-Lovelace: 86%

| Aplicação / Integração             | Funcionalidade                           | Descrição                                                            |
| ---------------------------------- | ---------------------------------------- | -------------------------------------------------------------------- |
| Definição de Tipo Documental       | Leitura página a página                  | Considera aspectos qualitativos que compõem o tipo de processo       |
| Definição de Processos             | Modelagem de documentos e tipos          | Criação de templates e estruturas para diferentes fluxos documentais |
| Agente Cognitivo                   | Integração com IA                        | Conexão com modelos de NLP para análise e automação de decisões      |
| Gerenciamento de Respostas         | Controle de tokens de saída              | Limitação de tokens para otimização de custos e performance          |
| Otimização de Modelos              | Aprimoramento via Few-Shot Learning      | Adaptação do modelo com poucos exemplos para tarefas específicas     |
| Especialização Temática            | Aceita overfitting controlado            | Redundância em dados especializados para ganho de precisão temática  |
| Integração com Ollama (ou similar) | Facilidade de uso                        | Exportação, manipulação e aplicação direta do modelo                 |
| Supervisão e Reforço Humano        | Preserva camadas arquivísticas complexas | Permite uso de bases de dados heterogêneas com validação contextual  |


## Licença

* Código — NC + CC-BY-NC.  
* Pesos LoRA — Vicuna NC + CC-BY-NC.

## Citação

```bibtex
@misc{rota2025adalovelace,
  title   = {Ada-Lovelace LoRA: Fine-tuning Vicuna-7B on Brazilian archival manuals},
  author  = {Rota, Alesson Ramon.},
  year    = {2025},
  url     = {https://github.com/<usuario>/ada-lovelace-lora}
}
```
