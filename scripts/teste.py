from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)
from peft import PeftModel
from threading import Thread
from tqdm import tqdm
import torch, os, itertools

# ───── caminhos ─────
base  = "lmsys/vicuna-7b-v1.5"
lora  = r"C:\Users\aless\Documents\treinamento\ada-lovelace-lora"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)

os.makedirs("offload", exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
    use_safetensors=True,
    offload_folder="offload",
    offload_state_dict=True
)
model = PeftModel.from_pretrained(
    model,
    lora,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True,
    offload_buffers=True
)
model.eval()

prompt = "Explique a diferença entre prontuário funcional e prontuário social."
inputs = tok(prompt, return_tensors="pt").to(model.device)

# ───── streamer + thread ─────
streamer = TextIteratorStreamer(tok, skip_prompt=True)
gen_kwargs = dict(
    **inputs,
    streamer        = streamer,
    max_new_tokens  = 120,
    temperature     = 0.7,
    top_p           = 0.9
)

def generate():
    with torch.no_grad():
        model.generate(**gen_kwargs)

thread = Thread(target=generate)
thread.start()

# barra de progresso (120 tokens máx)
progress = tqdm(itertools.islice(streamer, 120), total=120, desc="Gerando", unit="tok")
decoded  = ""
for token in progress:
    decoded += token
    progress.set_postfix_str(decoded[-40:])  # mostra cauda da frase

print("\n\nResposta completa:\n", decoded)
