import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dirs() -> tuple[Path, Path]:
    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
        runtime_dir = Path(sys.executable).resolve().parent
    else:
        runtime_dir = Path(__file__).resolve().parents[1]
        base_dir = runtime_dir
    return base_dir, runtime_dir


def has_safetensors_weights(path: Path) -> bool:
    if not path:
        return False
    if (path / "model.safetensors").exists():
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    return any(path.glob("model-*.safetensors"))


def has_pytorch_weights(path: Path) -> bool:
    if not path:
        return False
    if (path / "pytorch_model.bin").exists():
        return True
    if (path / "pytorch_model.bin.index.json").exists():
        return True
    return any(path.glob("pytorch_model-*.bin"))


def is_valid_base_model(path: Path) -> bool:
    if not path:
        return False
    if not (path / "config.json").exists():
        return False
    return has_safetensors_weights(path) or has_pytorch_weights(path)


def is_valid_lora(path: Path) -> bool:
    if not path:
        return False
    return (path / "adapter_config.json").exists()


def resolve_existing(env_name: str, candidates: list[Path], validator) -> Path | None:
    env_value = os.environ.get(env_name)
    if env_value:
        p = Path(env_value)
        if validator(p):
            return p
    for p in candidates:
        if validator(p):
            return p
    return None


def find_hf_vicuna_snapshot() -> Path | None:
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots_dir = (
        cache_root
        / "hub"
        / "models--lmsys--vicuna-7b-v1.5"
        / "snapshots"
    )
    if not snapshots_dir.exists():
        return None
    candidates = [
        p
        for p in snapshots_dir.glob("*")
        if (p / "config.json").exists()
        and (has_safetensors_weights(p) or has_pytorch_weights(p))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_prompt(args_prompt: list[str]) -> str:
    if args_prompt:
        return " ".join(args_prompt).strip()
    try:
        return input("Digite o prompt: ").strip()
    except EOFError:
        return ""


def load_tokenizer(model_path: Path):
    try:
        from transformers import LlamaTokenizer

        tok = LlamaTokenizer.from_pretrained(str(model_path))
    except Exception:
        tok = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa o modelo Ada-Lovelace (standalone, sem Ollama)."
    )
    parser.add_argument(
        "prompt", nargs="*", help="Texto da pergunta (se omitido, pede no terminal)."
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--base-model-path", default="", help="Caminho do base (vicuna-7b-v1.5)."
    )
    parser.add_argument(
        "--lora-path", default="", help="Caminho da LoRA ada-lovelace-lora."
    )
    args = parser.parse_args()

    prompt = build_prompt(args.prompt)
    if not prompt:
        print("Prompt vazio. Informe um texto para consulta.")
        return 2

    base_dir, runtime_dir = get_dirs()

    base_candidates = [
        Path(args.base_model_path) if args.base_model_path else Path(),
        runtime_dir / "model" / "base" / "vicuna-7b-v1.5",
        base_dir / "model" / "base" / "vicuna-7b-v1.5",
    ]
    hf_snapshot = find_hf_vicuna_snapshot()
    if hf_snapshot:
        base_candidates.append(hf_snapshot)
    lora_candidates = [
        Path(args.lora_path) if args.lora_path else Path(),
        runtime_dir / "model" / "ada-lovelace-lora",
        base_dir / "model" / "ada-lovelace-lora",
    ]

    base_model = resolve_existing("BASE_MODEL_PATH", base_candidates, is_valid_base_model)
    lora_path = resolve_existing("LORA_PATH", lora_candidates, is_valid_lora)

    if not base_model or not lora_path:
        print("Nao encontrei os pesos localmente.")
        print("Base esperado em: model/base/vicuna-7b-v1.5 (ou BASE_MODEL_PATH)")
        print("LoRA esperada em: model/ada-lovelace-lora (ou LORA_PATH)")
        print("O base precisa conter pesos (safetensors ou pytorch_model.bin).")
        return 3

    offload_dir = runtime_dir / "offload"
    offload_dir.mkdir(exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tok = load_tokenizer(base_model)

    use_safetensors = has_safetensors_weights(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=dtype,
        device_map="auto",
        use_safetensors=use_safetensors,
        local_files_only=True,
        offload_folder=str(offload_dir),
        offload_state_dict=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(lora_path),
        device_map="auto",
        local_files_only=True,
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        offload_buffers=True,
    )
    model.eval()

    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = f"USER: {prompt}\nASSISTANT:"

    inputs = tok(rendered, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(output_ids[0], skip_special_tokens=True)
    print("\n\nResposta completa:\n", decoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
