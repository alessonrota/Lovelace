import argparse
import subprocess
import sys


def build_prompt(args_prompt: list[str]) -> str:
    if args_prompt:
        return " ".join(args_prompt).strip()

    try:
        return input("Digite o prompt: ").strip()
    except EOFError:
        return ""


def run_ollama(model: str, prompt: str) -> int:
    if not prompt:
        print("Prompt vazio. Informe um texto para consulta.")
        return 2

    cmd = ["ollama", "run", model, prompt]
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("Ollama nao encontrado. Instale e coloque no PATH: https://ollama.com")
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Executa o modelo Ada-Lovelace via Ollama."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Texto da pergunta (se omitido, pede no terminal)."
    )
    parser.add_argument(
        "--model",
        default="ada-lovelace",
        help="Nome do modelo no Ollama (default: ada-lovelace)."
    )
    args = parser.parse_args()

    prompt = build_prompt(args.prompt)
    return run_ollama(args.model, prompt)


if __name__ == "__main__":
    raise SystemExit(main())
