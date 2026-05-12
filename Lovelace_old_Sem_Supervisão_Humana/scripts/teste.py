from transformers import (AutoModelForCausalLM, AutoTokenizer)
from huggingface_hub import snapshot_download
from peft import PeftModel
from pathlib import Path
import torch, os, sys

# ───── caminhos ─────
if getattr(sys, "frozen", False):
    runtime_root = Path(sys.executable).resolve().parent
    base_root = Path(getattr(sys, "_MEIPASS", runtime_root))
else:
    runtime_root = Path(__file__).resolve().parents[1]
    base_root = runtime_root

model_root = base_root / "model"

base = Path(os.environ.get(
    "BASE_MODEL_PATH",
    str(model_root / "base" / "vicuna-7b-v1.5")
))
lora = Path(os.environ.get(
    "LORA_PATH",
    str(model_root / "ada-lovelace-lora")
))

print(f"Base esperado em: {base}")
print(f"LoRA esperada em: {lora}")

if not (base / "config.json").exists():
    print(
        "Base model nao encontrado localmente. Baixando vicuna-7b-v1.5 para uso futuro..."
    )
    base.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="lmsys/vicuna-7b-v1.5",
        local_dir=str(base),
        local_dir_use_symlinks=False,
    )
    if not (base / "config.json").exists():
        raise FileNotFoundError(
            "Falha ao baixar o base. Verifique conexao ou permissao de escrita em "
            "model/base/vicuna-7b-v1.5."
        )
else:
    print(f"Base local encontrada: {base}")
if not (lora / "adapter_config.json").exists():
    raise FileNotFoundError(
        "LoRA nao encontrada. Defina LORA_PATH ou coloque a LoRA em "
        "model/ada-lovelace-lora."
    )

try:
    from transformers import LlamaTokenizer
    tok = LlamaTokenizer.from_pretrained(str(base), local_files_only=True)
except Exception:
    tok = AutoTokenizer.from_pretrained(str(base), use_fast=True, local_files_only=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

offload_dir = runtime_root / "offload"
offload_dir.mkdir(exist_ok=True)

dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    str(base),
    torch_dtype=dtype,
    device_map="auto",
    use_safetensors=True,
    local_files_only=True,
    offload_folder=str(offload_dir),
    offload_state_dict=True
)
model = PeftModel.from_pretrained(
    model,
    str(lora),
    device_map="auto",
    local_files_only=True,
    offload_folder=str(offload_dir),
    offload_state_dict=True,
    offload_buffers=True
)
model.eval()
print(f"LoRA carregada de: {lora}")

user_prompt = "Explique a diferença entre prontuário funcional e prontuário social."
if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
else:
    prompt = f"USER: {user_prompt}\nASSISTANT:"

inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id
    )

decoded = tok.decode(output_ids[0], skip_special_tokens=True)
print("\n\nResposta completa:\n", decoded)
