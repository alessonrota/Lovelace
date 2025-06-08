# processa_livros.py
import os, re, shutil, pathlib
from tqdm import tqdm
from pdfminer.high_level import extract_text

# --- 0. Configurações ---
root = pathlib.Path(__file__).resolve().parent           # pasta onde o script está
pdf_dir   = root / "pdf_raw"
txt_dir   = root / "txt_clean"
log_dir   = root / "logs"
log_file  = log_dir / "extract_log.txt"

# --- 1. Garante a estrutura de pastas ---
for p in (pdf_dir, txt_dir, log_dir):
    p.mkdir(exist_ok=True)

# --- 2. Move PDFs que estejam soltos na pasta-raiz para pdf_raw ---
for pdf in root.glob("*.pdf"):
    shutil.move(pdf, pdf_dir / pdf.name)

# --- 3. Extrai texto dos PDFs ---
total, ok, fail = 0, 0, 0
with open(log_file, "w", encoding="utf-8") as log:
    for pdf_path in tqdm(list(pdf_dir.glob("*.pdf")), desc="Extraindo PDFs"):
        total += 1
        try:
            text = extract_text(str(pdf_path))
            # limpeza simples: remove múltiplos form feeds
            text = re.sub(r"\f+", "\n", text)
            (txt_dir / f"{pdf_path.stem}.txt").write_text(text, encoding="utf-8")
            ok += 1
        except Exception as e:
            fail += 1
            log.write(f"{pdf_path.name}: {e}\n")

# --- 4. Resumo ---
print(f"\n✅ Concluído! {ok}/{total} PDFs extraídos com sucesso.")
if fail:
    print(f"⚠️  {fail} arquivos falharam — veja detalhes em {log_file}")
