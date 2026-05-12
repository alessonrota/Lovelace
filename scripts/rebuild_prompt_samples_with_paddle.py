#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paddleocr import PaddleOCR


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_text_from_paddle_result(result: Any) -> str:
    if not isinstance(result, list) or not result:
        return ""
    lines_blob = result[0]
    if not isinstance(lines_blob, list):
        return ""
    lines: list[str] = []
    for item in lines_blob:
        if not isinstance(item, list) or len(item) < 2:
            continue
        rec = item[1]
        if not isinstance(rec, (list, tuple)) or not rec:
            continue
        txt = rec[0]
        if isinstance(txt, str):
            txt = txt.strip()
            if txt:
                lines.append(txt)
    return "\n".join(lines).strip()


def make_example_blocks(rows: list[dict[str, Any]]) -> str:
    counts: dict[int, int] = defaultdict(int)
    lines: list[str] = []
    lines.append("Static reference examples (two real OCR examples per class, full OCR, PaddleOCR+en):")
    lines.append("Do not assume other classes are absent. Always choose among all 16 classes.")
    lines.append("")
    for row in sorted(rows, key=lambda r: (int(r["class_id"]), int(r.get("example_id", 9999)))):
        cid = int(row["class_id"])
        cname = str(row["class_name"])
        counts[cid] += 1
        ex_id = int(row.get("example_id", counts[cid]))
        txt = str(row["ocr_full_paddle"]).strip()
        lines.append(f"[Example class_id={cid} class_name={cname} example_id={ex_id}]")
        lines.append("ocr_text:")
        lines.append('"""')
        lines.append(txt)
        lines.append('"""')
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def replace_prompt_examples(prompt_text: str, new_examples_block: str) -> str:
    start_anchor = "Static reference examples"
    end_anchor = "\nRules:"
    start_idx = prompt_text.find(start_anchor)
    if start_idx < 0:
        raise RuntimeError("Could not find 'Static reference examples' in prompt file.")
    end_idx = prompt_text.find(end_anchor, start_idx)
    if end_idx < 0:
        raise RuntimeError("Could not find '\\nRules:' in prompt file.")

    prefix = prompt_text[:start_idx]
    suffix = prompt_text[end_idx + 1 :]  # keep "Rules:" line
    return prefix + new_examples_block + "\n" + suffix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild RVL prompt samples using PaddleOCR")
    p.add_argument(
        "--examples-jsonl",
        type=Path,
        default=Path("data/processed/saida-ocr-class/run_002/manifests/fewshot_examples_v5_best_eng_full.jsonl"),
    )
    p.add_argument("--prompt-file", type=Path, default=Path("configs/rvl_system_prompt.txt"))
    p.add_argument(
        "--output-examples-jsonl",
        type=Path,
        default=Path("data/processed/saida-ocr-class/run_002/manifests/fewshot_examples_v6_best_eng_paddle_full.jsonl"),
    )
    p.add_argument("--ocr-lang", default="en")
    p.add_argument("--no-backup", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.examples_jsonl.exists():
        raise RuntimeError(f"Examples file not found: {args.examples_jsonl}")
    if not args.prompt_file.exists():
        raise RuntimeError(f"Prompt file not found: {args.prompt_file}")

    rows = read_jsonl(args.examples_jsonl)
    if not rows:
        raise RuntimeError("Examples JSONL is empty.")

    ocr = PaddleOCR(use_angle_cls=False, lang=args.ocr_lang, show_log=False)
    out_rows: list[dict[str, Any]] = []
    failures = 0

    for i, row in enumerate(rows, start=1):
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise RuntimeError(f"Missing image file: {image_path}")
        result = ocr.ocr(str(image_path), cls=False)
        text = extract_text_from_paddle_result(result)
        used_fallback = False
        if not text:
            used_fallback = True
            failures += 1
            text = str(row.get("ocr_full", "")).strip()
        new_row = dict(row)
        new_row["ocr_full_paddle"] = text
        new_row["ocr_source_paddle"] = f"paddleocr:{args.ocr_lang}"
        new_row["rebuilt_at"] = now_iso()
        out_rows.append(new_row)
        print(f"[{i:02d}/{len(rows):02d}] class={row['class_id']} chars={len(text)} fallback={'yes' if used_fallback else 'no'}")

    write_jsonl(args.output_examples_jsonl, out_rows)

    prompt_text = args.prompt_file.read_text(encoding="utf-8")
    updated_prompt = replace_prompt_examples(prompt_text, make_example_blocks(out_rows))

    if not args.no_backup:
        backup_path = args.prompt_file.with_name(f"{args.prompt_file.stem}_backup_before_paddle_{datetime.now().strftime('%Y%m%d_%H%M%S')}{args.prompt_file.suffix}")
        shutil.copy2(args.prompt_file, backup_path)
        print(f"Backup created: {backup_path}")

    updated_prompt = re.sub(r"^VERSION:.*$", "VERSION: v6_best_32full_eng_paddle", updated_prompt, flags=re.MULTILINE)
    args.prompt_file.write_text(updated_prompt, encoding="utf-8")

    print(f"Prompt updated: {args.prompt_file}")
    print(f"Examples saved: {args.output_examples_jsonl}")
    print(f"Total examples: {len(out_rows)} | OCR fallback count: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
