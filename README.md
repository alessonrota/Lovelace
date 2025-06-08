# Ada-Lovelace LoRA 🇧🇷 Arquivologia e lingua portuguesa

LoRA (\~80 MB) finetunada sobre **\[Vicuna-7B v1.5]** para melhorar respostas em arquivologia e classificação de processos SEI-SP.

![banner](https://img.shields.io/badge/LoRA-Vicuna7B-blue)
![license](https://img.shields.io/badge/license-MIT-%2B-NC-green)

---

## Objetivo

Classificacao de documentos lingua portuguesa

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

## Dados

Nove manuais públicos do Arquivo do Estado de SP.
Links, verificações SHA-256 e notas estão em `data/README.md`.

##  Treinamento (LoRA)

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
# ou python scripts/train_lora_cpu.py                  # CPU
```

### Modelfile

```
FROM vicuna:7b
ADAPTER ada-lovelace-lora
```


## Resultados

Capacidade em define documentos e tipos de processo
integração com agente cognitivo
controle de tolkens de saida
aprimoramento com few shot

| Prompt                          | Vicuna-7B                   | Ada-Lovelace                           |Taxa de acerto Vicuna-7B      |Taxa de acerto Ada-Lovelace     |
| ------------------------------- | --------------------------- | -------------------------------------- |------------------------------|
| "prontuário funcional × social" | respostas genéricas         | resposta com vocabulário especializado | Melhor cenário 70%           |Melhor cenário 86%



## Licença

* Código — NC + CC-BY-NC.
* Pesos LoRA — Vicuna NC + CC-BY-NC.

## Citação

```
@misc{rota2025adalovelace,
  title   = {Ada-Lovelace LoRA: Fine-tuning Vicuna-7B on Brazilian archival manuals},
  author  = {Rota, Alesson Ramon.},
  year    = {2025},
  url     = {https://github.com/<usuario>/ada-lovelace-lora}
}
```

---
