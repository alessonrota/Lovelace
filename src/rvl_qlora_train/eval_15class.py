from __future__ import annotations

import csv
import json
import re
import time
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .common import atomic_write_json, now_iso_utc, progress_line, read_jsonl


def _parse_class_id(text: str, class_ids: set[int]) -> int | None:
    text = text.strip()
    if not text:
        return None

    # direct integer reply
    m = re.search(r"-?\d+", text)
    if m:
        cid = int(m.group(0))
        if cid in class_ids:
            return cid

    # optional JSON fallback
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "class_id" in obj:
            cid = int(obj["class_id"])
            if cid in class_ids:
                return cid
    except Exception:
        pass

    return None


def _extract_class_id_candidates(text: str, class_ids: set[int]) -> list[int]:
    found: list[int] = []
    for m in re.findall(r"-?\d+", text):
        try:
            cid = int(m)
        except Exception:
            continue
        if cid in class_ids:
            found.append(cid)
    return found


def _looks_ambiguous_response(text: str, class_ids: set[int]) -> bool:
    low = text.lower()
    if any(token in low for token in ["maybe", "perhaps", "not sure", "uncertain", "or "]):
        return True
    candidates = _extract_class_id_candidates(text, class_ids)
    unique = set(candidates)
    return len(unique) > 1


def _majority_vote(ids: list[int], tie_breaker: int | None = None) -> int | None:
    if not ids:
        return None
    counts = Counter(ids)
    top = counts.most_common()
    if not top:
        return None
    best_count = top[0][1]
    best = [cid for cid, cnt in top if cnt == best_count]
    if tie_breaker is not None and tie_breaker in best:
        return tie_breaker
    return min(best)


def _compute_metrics(rows: list[dict[str, Any]], class_map: dict[int, str]) -> dict[str, Any]:
    class_ids = sorted(class_map.keys())
    id2idx = {cid: idx for idx, cid in enumerate(class_ids)}
    n = len(class_ids)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    total = len(rows)
    ok_rows = [r for r in rows if r["status"] == "ok" and isinstance(r.get("pred_class_id"), int)]

    for row in ok_rows:
        t = int(row["true_class_id"])
        p = int(row["pred_class_id"])
        matrix[id2idx[t]][id2idx[p]] += 1

    correct_ok = 0
    for cid in class_ids:
        i = id2idx[cid]
        correct_ok += matrix[i][i]

    strict_correct = sum(
        1
        for row in rows
        if row["status"] == "ok" and isinstance(row.get("pred_class_id"), int) and row["pred_class_id"] == row["true_class_id"]
    )

    ok_total = len(ok_rows)
    accuracy = (correct_ok / ok_total) if ok_total else 0.0
    strict_accuracy = (strict_correct / total) if total else 0.0
    coverage = (ok_total / total) if total else 0.0

    per_class: list[dict[str, Any]] = []
    f1_values: list[float] = []

    for cid in class_ids:
        i = id2idx[cid]
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n) if r != i)
        fn = sum(matrix[i][c] for c in range(n) if c != i)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(matrix[i])
        per_class.append(
            {
                "class_id": cid,
                "class_name": class_map[cid],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        f1_values.append(f1)

    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "generated_at": now_iso_utc(),
        "num_total": total,
        "num_ok": ok_total,
        "coverage": coverage,
        "accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
        "macro_f1": macro_f1,
        "status_counts": status_counts,
        "class_ids": class_ids,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _write_predictions(predictions_path: Path, rows: list[dict[str, Any]]) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_predictions_csv(predictions_csv: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "doc_id",
        "split",
        "rel_path",
        "status",
        "true_class_id",
        "true_class_name",
        "pred_class_id",
        "pred_class_name",
        "raw_response",
        "is_correct",
    ]
    with predictions_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_per_class_csv(path: Path, per_class: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class_id", "class_name", "precision", "recall", "f1", "support"])
        w.writeheader()
        w.writerows(per_class)


def _write_confusion_csv(path: Path, metrics: dict[str, Any], class_map: dict[int, str]) -> None:
    class_ids = metrics["class_ids"]
    matrix = metrics["confusion_matrix"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["true_class_id", "true_class_name"] + [f"pred_{cid}" for cid in class_ids]
        w.writerow(header)
        for idx, cid in enumerate(class_ids):
            w.writerow([cid, class_map[cid], *matrix[idx]])


def evaluate_dataset(
    dataset_jsonl: Path,
    base_model: str,
    adapter_dir: Path,
    out_dir: Path,
    class_map: dict[int, str],
    logger: Any,
    max_new_tokens: int = 12,
    gpu_index: int = 0,
    primary_temperature: float = 0.0,
    primary_top_p: float = 0.9,
    primary_top_k: int = 40,
    invalid_json_fallback_retry: int = 1,
    fallback_max_new_tokens: int = 24,
    ambiguous_votes: int = 1,
    vote_temperature: float = 0.2,
    vote_top_p: float = 0.95,
    vote_top_k: int = 50,
    vote_on_invalid_only: bool = True,
) -> dict[str, Any]:
    # Keep evaluation pinned to a single physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "Missing eval dependencies. Install: torch transformers peft bitsandbytes"
        ) from exc

    rows = read_jsonl(dataset_jsonl)
    if not rows:
        raise RuntimeError(f"Evaluation dataset is empty: {dataset_jsonl}")

    class_ids = set(class_map.keys())

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    logger.info(f"Loading base model for evaluation on physical cuda:{gpu_index} (logical cuda:0): {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_dir.mkdir(parents=True, exist_ok=True)

    preds: list[dict[str, Any]] = []
    total = len(rows)
    done = 0
    started = time.perf_counter()

    for row in rows:
        done += 1
        prompt = str(row.get("prompt", ""))

        status = "ok"
        pred_class_id: int | None = None
        pred_class_name: str | None = None
        raw_response = ""

        try:
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(model.device) for k, v in encoded.items()}

            def _generate_once(
                do_sample: bool,
                temperature: float,
                top_p: float,
                top_k: int,
                max_tokens: int,
            ) -> str:
                kwargs = {
                    "do_sample": do_sample,
                    "max_new_tokens": max_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    kwargs["temperature"] = max(1e-5, float(temperature))
                    kwargs["top_p"] = float(top_p)
                    kwargs["top_k"] = int(top_k)
                with torch.no_grad():
                    output = model.generate(**encoded, **kwargs)
                new_tokens = output[0][encoded["input_ids"].shape[1] :]
                return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            raw_response = _generate_once(
                do_sample=(primary_temperature > 0.0),
                temperature=primary_temperature,
                top_p=primary_top_p,
                top_k=primary_top_k,
                max_tokens=max_new_tokens,
            )
            pred_class_id = _parse_class_id(raw_response, class_ids)

            need_vote = False
            if ambiguous_votes > 1:
                if vote_on_invalid_only:
                    need_vote = pred_class_id is None
                else:
                    need_vote = pred_class_id is None or _looks_ambiguous_response(raw_response, class_ids)

            if need_vote:
                vote_ids: list[int] = []
                if pred_class_id is not None:
                    vote_ids.append(pred_class_id)
                for _ in range(int(ambiguous_votes)):
                    vote_raw = _generate_once(
                        do_sample=True,
                        temperature=vote_temperature,
                        top_p=vote_top_p,
                        top_k=vote_top_k,
                        max_tokens=max_new_tokens,
                    )
                    vote_id = _parse_class_id(vote_raw, class_ids)
                    if vote_id is not None:
                        vote_ids.append(vote_id)
                pred_class_id = _majority_vote(vote_ids, tie_breaker=pred_class_id)

            if pred_class_id is None and invalid_json_fallback_retry > 0:
                for _ in range(int(invalid_json_fallback_retry)):
                    fallback_raw = _generate_once(
                        do_sample=True,
                        temperature=vote_temperature,
                        top_p=vote_top_p,
                        top_k=vote_top_k,
                        max_tokens=fallback_max_new_tokens,
                    )
                    fallback_id = _parse_class_id(fallback_raw, class_ids)
                    if fallback_id is not None:
                        pred_class_id = fallback_id
                        raw_response = fallback_raw
                        break

            if pred_class_id is None:
                status = "validation_error"
            else:
                pred_class_name = class_map[pred_class_id]
        except Exception as exc:
            status = "llm_error"
            raw_response = str(exc)

        true_class_id = int(row["class_id"])

        preds.append(
            {
                "doc_id": row["doc_id"],
                "split": row["split"],
                "rel_path": row["rel_path"],
                "status": status,
                "true_class_id": true_class_id,
                "true_class_name": row["class_name"],
                "pred_class_id": pred_class_id,
                "pred_class_name": pred_class_name,
                "raw_response": raw_response,
                "is_correct": bool(status == "ok" and pred_class_id == true_class_id),
            }
        )

        logger.info(progress_line("EVAL", done, total, started))

    metrics = _compute_metrics(preds, class_map=class_map)

    predictions_jsonl = out_dir / "predictions.jsonl"
    predictions_csv = out_dir / "predictions.csv"
    metrics_json = out_dir / "metrics.json"
    per_class_csv = out_dir / "per_class.csv"
    confusion_csv = out_dir / "confusion_matrix.csv"

    _write_predictions(predictions_jsonl, preds)
    _write_predictions_csv(predictions_csv, preds)
    atomic_write_json(metrics_json, metrics)
    _write_per_class_csv(per_class_csv, metrics["per_class"])
    _write_confusion_csv(confusion_csv, metrics, class_map=class_map)

    return {
        "dataset": str(dataset_jsonl),
        "predictions_jsonl": str(predictions_jsonl),
        "predictions_csv": str(predictions_csv),
        "metrics_json": str(metrics_json),
        "per_class_csv": str(per_class_csv),
        "confusion_csv": str(confusion_csv),
        "metrics": metrics,
    }
