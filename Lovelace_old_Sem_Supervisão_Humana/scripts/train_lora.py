# train_lora.py  — Continual pre-training LoRA em FP16 (Windows, RTX 4060)
import os, torch, transformers
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model

# ▸ Caminhos ------------------------------------------------------------------
root       = os.path.dirname(__file__)
data_path  = os.path.join(root, "hf_dataset")
model_name = "lmsys/vicuna-7b-v1.5"

# ▸ 1. Dataset ----------------------------------------------------------------
print("→ Carregando dataset …")
ds = load_from_disk(data_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)

# ▸ 2. Modelo base (FP16, safetensors, offload) -------------------------------
print("→ Carregando modelo base …")
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",          # divide GPU / CPU
    low_cpu_mem_usage=True,
    use_safetensors=True
)

# ▸ 3. Configura LoRA ---------------------------------------------------------
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)

# ▸ 4. Tokenização ------------------------------------------------------------
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=2048)

tok_ds = ds.map(tokenize, batched=True, remove_columns=["text"])
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ▸ 5. TrainingArguments ------------------------------------------------------
args = transformers.TrainingArguments(
    output_dir="ada-lovelace-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=2,
    save_total_limit=1
)

# ▸ 6. Trainer que NÃO move o modelo -----------------------------------------
class NoMoveTrainer(transformers.Trainer):
    def _move_model_to_device(self, model, device):
        # Mantém modelo onde já está (GPU + CPU via device_map)
        return model

trainer = NoMoveTrainer(
    model=model,
    args=args,
    train_dataset=tok_ds,
    data_collator=data_collator
)

print("→ Iniciando treino …")
trainer.train()

# ▸ 7. Salva adaptadores LoRA -------------------------------------------------
model.save_pretrained("ada-lovelace-lora")
tokenizer.save_pretrained("ada-lovelace-lora")
print("✅ LoRA concluída e salva em ada-lovelace-lora")
