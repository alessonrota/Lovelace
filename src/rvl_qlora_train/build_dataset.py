from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json, now_iso_utc, read_jsonl, truncate_text


def _load_ocr_index(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        raise RuntimeError(f"OCR index not found: {index_path}")
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"OCR index format invalid: {index_path}")
    return {str(k): dict(v) for k, v in raw.items()}


def build_user_instruction(class_map: dict[int, str]) -> str:
    pairs = [f"{cid}={class_map[cid]}" for cid in sorted(class_map.keys())]
    mapping = "\n".join(pairs)
    return (
        "Classify the OCR text into exactly one RVL-CDIP class id from the active set below.\n"
        "Return only the numeric class id.\n\n"
        f"{mapping}\n"
    )


def _compose_prompt(system_prompt: str, user_instruction: str, ocr_text: str) -> str:
    return (
        f"[SYSTEM]\n{system_prompt.strip()}\n\n"
        f"[USER]\n{user_instruction.strip()}\n"
        "ocr_text:\n\"\"\"\n"
        f"{ocr_text}\n"
        "\"\"\"\n"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_datasets_from_ocr(
    datasets_dir: Path,
    manifests_dir: Path,
    ocr_index_path: Path,
    class_map: dict[int, str],
    system_prompt: str,
    max_ocr_chars: int,
    min_ocr_chars: int,
) -> dict[str, Any]:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ocr_index = _load_ocr_index(ocr_index_path)

    split_to_manifest = {
        "train_balanced_fast": manifests_dir / "train_balanced_fast.jsonl",
        "val_balanced": manifests_dir / "val_balanced.jsonl",
        "test_holdout_balanced": manifests_dir / "test_holdout_balanced.jsonl",
        "eval_legacy_93": manifests_dir / "eval_legacy_93.jsonl",
    }

    user_instruction = build_user_instruction(class_map)
    summary: dict[str, Any] = {
        "generated_at": now_iso_utc(),
        "max_ocr_chars": max_ocr_chars,
        "min_ocr_chars": min_ocr_chars,
        "splits": {},
    }

    for split_name, manifest_path in split_to_manifest.items():
        if not manifest_path.exists():
            raise RuntimeError(f"Manifest not found for split {split_name}: {manifest_path}")

        rows = read_jsonl(manifest_path)
        out_rows: list[dict[str, Any]] = []

        counters = {
            "source": len(rows),
            "ok": 0,
            "dropped_ocr_error": 0,
            "dropped_missing_ocr_file": 0,
            "dropped_short_text": 0,
        }

        for row in rows:
            doc_id = str(row["doc_id"])
            class_id = int(row["class_id"])
            class_name = str(row["class_name"])

            ocr_meta = ocr_index.get(doc_id)
            if not ocr_meta or ocr_meta.get("status") != "ok":
                counters["dropped_ocr_error"] += 1
                continue

            ocr_path = Path(str(ocr_meta.get("ocr_path", "")))
            if not ocr_path.exists():
                counters["dropped_missing_ocr_file"] += 1
                continue

            ocr_text = ocr_path.read_text(encoding="utf-8", errors="ignore")
            ocr_text = truncate_text(ocr_text, max_ocr_chars)
            ocr_text = ocr_text.strip()
            if len(ocr_text) < min_ocr_chars:
                counters["dropped_short_text"] += 1
                continue

            prompt = _compose_prompt(system_prompt, user_instruction, ocr_text)
            target = str(class_id)

            out_rows.append(
                {
                    "doc_id": doc_id,
                    "split": split_name,
                    "rel_path": row["rel_path"],
                    "class_id": class_id,
                    "class_name": class_name,
                    "ocr_path": str(ocr_path),
                    "ocr_chars": len(ocr_text),
                    "ocr_text": ocr_text,
                    "prompt": prompt,
                    "target": target,
                }
            )

        counters["ok"] = len(out_rows)
        out_path = datasets_dir / f"{split_name}.jsonl"
        _write_jsonl(out_path, out_rows)

        prompt_preview_path = datasets_dir / f"{split_name}_prompt_preview.txt"
        if out_rows:
            prompt_preview_path.write_text(out_rows[0]["prompt"], encoding="utf-8")

        summary["splits"][split_name] = {
            **counters,
            "dataset_path": str(out_path),
            "prompt_preview_path": str(prompt_preview_path),
        }

    atomic_write_json(datasets_dir / "dataset_summary.json", summary)
    return summary
