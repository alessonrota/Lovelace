
# Estrutura do Diretório `ada-lovelace-lora`

```txt
ada-lovelace-lora/
│
├── adapter_config.json
├── adapter_model.safetensors
├── Modelfile.txt
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer.model
├── tokenizer_config.json
│
└── checkpoint-3/
     ├── adapter_config.json
     ├── adapter_model.safetensors
     ├── optimizer.pt
     ├── README.md
     ├── rng_state.pth
     ├── scaler.pt
     ├── scheduler.pt
     ├── special_tokens_map.json
     ├── tokenizer.json
     ├── tokenizer.model
     ├── tokenizer_config.json
     ├── trainer_state.json
     └── training_args.bin
```

## O que é cada arquivo/pasta?

### Arquivos na raiz (`ada-lovelace-lora/`):

- **adapter_config.json**  
  Arquivo de configuração dos parâmetros LoRA (Low-Rank Adaptation). Define hiperparâmetros e arquitetura dos adaptadores treinados.

- **adapter_model.safetensors**  
  Arquivo binário contendo os pesos do adaptador treinado (LoRA). É o núcleo do que foi aprendido no fine-tuning.

- **Modelfile.txt**  
  Descreve para o Ollama ou outro loader qual base usar, como montar o modelo, etc. (ex: `FROM vicuna:7b`).

- **special_tokens_map.json**  
  Mapeamento de tokens especiais (ex: tokens de início/fim de texto, separadores, etc).

- **tokenizer.json / tokenizer.model / tokenizer_config.json**  
  Arquivos da tokenização do modelo:
  - **tokenizer.json:** vocabulário e regras em formato JSON
  - **tokenizer.model:** modelo de tokenização (normalmente SentencePiece ou similar)
  - **tokenizer_config.json:** configurações extras da tokenização

---

### Pasta `checkpoint-3/`:

Cada checkpoint é um “salvamento” do treinamento em determinado passo (step/epoch).  
Pode haver vários checkpoints, mas aqui há um, chamado `checkpoint-3`.

- **adapter_config.json, adapter_model.safetensors**  
  Versão do adaptador/LoRA no estado salvo neste checkpoint.

- **optimizer.pt, scheduler.pt, scaler.pt, rng_state.pth, trainer_state.json, training_args.bin**  
  Dados do treinamento para retomar (resume) o treinamento de onde parou:
  - **optimizer.pt:** estado do otimizador (ex: AdamW)
  - **scheduler.pt:** scheduler de learning rate
  - **scaler.pt:** usado em treinos com mixed precision
  - **rng_state.pth:** estado dos geradores aleatórios (reprodutibilidade)
  - **trainer_state.json:** progresso do treinamento (epoch, loss, etc)
  - **training_args.bin:** hiperparâmetros e configs do HuggingFace Trainer

