# build_dataset.py  (coloque na pasta treinamento)
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm

root     = Path(__file__).resolve().parent
txt_dir  = root / "txt_clean"
out_dir  = root / "hf_dataset"

assert txt_dir.exists(), "Pasta txt_clean não encontrada"

records = []
for txt_file in tqdm(list(txt_dir.glob("*.txt")), desc="Lendo TXT"):
    text = txt_file.read_text(encoding="utf-8", errors="ignore")
    # Se ainda houver BOM no começo, remove
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    records.append({"text": text})

# Cria Dataset a partir da lista
ds = Dataset.from_list(records)
print("Total documentos:", len(ds))

# Salva em disco
out_dir.mkdir(exist_ok=True)
ds.save_to_disk(out_dir)
print(f"Dataset salvo em {out_dir}")
