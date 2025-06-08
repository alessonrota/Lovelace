# Ada-Lovelace LoRA 🇧🇷

LoRA (~80 MB) finetunada sobre **[Vicuna-7B v1.5]** para melhorar respostas em
arquivologia e classificação de processos SEI-SP.

![banner](https://img.shields.io/badge/LoRA-Vicuna7B-blue)
![license](https://img.shields.io/badge/license-MIT-%2B-NC-green)

---

##  Objetivo  
Ajudar servidores públicos a entender diferenças entre **415** tipos de processo
(SEI São Paulo) e conceitos de gestão documental (ex.: prontuário funcional × social).

##  Uso rápido

git clone https://github.com/<usuario>/ada-lovelace-lora.git
cd ada-lovelace-lora
pip install -r requirements.txt

# precisa já ter o modelo base:
ollama pull vicuna:7b

# cria o modelo final no Ollama
ollama create ada-lovelace -f Modelfile

# teste:
ollama run ada-lovelace "Explique a diferença entre prontuário funcional e social."


## Treinamento (LoRA)
| Param       | Valor                     |
| ----------- | ------------------------- |
| Base        | vicuna-7b-v1.5            |
| Épocas      | **1**                     |
| SeqLen      | 1 024                     |
| Rank (LoRA) | 16                        |
| LR          | 1e-4 (cosine)             |
| HW          | i9-13900H · RTX 4060 8 GB |
| Tempo       | 22 min GPU / 2 h CPU      |

# Passos
python scripts/process_pdfs.py data/pdf_raw data/txt_clean
python scripts/build_dataset.py
python scripts/train_lora.py --dataset_dir hf_dataset  # GPU

# Data
Ler data/README.md

## Licença

- **Código - NC + CC-BY-NC
Pesos LoRA — Vicuna NC + CC-BY-NC

- **Developed by:** [Alesson Ramon Rota]
- **Model type:** [Ada-LoveLace-3]
- **Language(s) (NLP):** [português]
- **License:** [Código - NC + CC-BY-NC, Pesos LoRA — Vicuna NC + CC-BY-NC]

# Citação
@misc{rota2025adalovelace,
  title   = {Ada-Lovelace LoRA: Fine-tuning Vicuna-7B on Brazilian archival manuals},
  author  = {ROTA, Alesson Ramon.},
  year    = {2025},
  url     = {https://github.com/alessonrota/Lovelace}
}


- PEFT 0.15.2