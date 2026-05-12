#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "configs").exists():
            return p
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rvl_qlora_train.common import PipelineLogger, load_class_map, progress_line, read_jsonl
from src.rvl_qlora_train.eval_15class import (
    _compute_metrics,
    _looks_ambiguous_response,
    _majority_vote,
    _parse_class_id,
    _write_confusion_csv,
    _write_per_class_csv,
    _write_predictions,
    _write_predictions_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stable final evaluation for JurisTCU TIPOPROCESSO using BASE model only (no LoRA)."
    )
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "juristcu-tipoprocesso",
    )
    p.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--splits", default="val,test")
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--fallback-max-new-tokens", type=int, default=24)
    p.add_argument("--invalid-json-fallback-retry", type=int, default=1)
    p.add_argument("--ambiguous-votes", type=int, default=3)
    p.add_argument("--vote-on-invalid-only", action="store_true")
    p.add_argument("--primary-temperature", type=float, default=0.0)
    p.add_argument("--primary-top-p", type=float, default=0.9)
    p.add_argument("--primary-top-k", type=int, default=40)
    p.add_argument("--vote-temperature", type=float, default=0.2)
    p.add_argument("--vote-top-p", type=float, default=0.95)
    p.add_argument("--vote-top-k", type=int, default=50)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def evaluate_dataset_base(
    dataset_jsonl: Path,
    base_model: str,
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "Missing eval dependencies. Install: torch transformers bitsandbytes"
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
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, use_fast=False)
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
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
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


def main() -> int:
    args = parse_args()
    run_root = args.output_root / args.run_id
    if not run_root.exists():
        raise RuntimeError(f"Run root not found: {run_root}")

    datasets_dir = run_root / "datasets"
    class_map_path = run_root / "configs" / "class_map.json"
    if not datasets_dir.exists():
        raise RuntimeError(f"Datasets dir not found: {datasets_dir}")
    if not class_map_path.exists():
        raise RuntimeError(f"class_map.json not found: {class_map_path}")

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = run_root / f"evaluation_manual_stable_base14b_rest_full_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger(out_dir / "pipeline.log")
    class_map = load_class_map(class_map_path)

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not split_names:
        raise RuntimeError("No split provided.")

    summary: dict[str, dict[str, float | int]] = {}
    for split in split_names:
        dataset_jsonl = datasets_dir / f"{split}.jsonl"
        if not dataset_jsonl.exists():
            raise RuntimeError(f"Dataset not found for split '{split}': {dataset_jsonl}")

        res = evaluate_dataset_base(
            dataset_jsonl=dataset_jsonl,
            base_model=args.base_model,
            out_dir=out_dir / split,
            class_map=class_map,
            logger=logger,
            max_new_tokens=args.max_new_tokens,
            gpu_index=args.gpu_index,
            primary_temperature=args.primary_temperature,
            primary_top_p=args.primary_top_p,
            primary_top_k=args.primary_top_k,
            invalid_json_fallback_retry=args.invalid_json_fallback_retry,
            fallback_max_new_tokens=args.fallback_max_new_tokens,
            ambiguous_votes=args.ambiguous_votes,
            vote_temperature=args.vote_temperature,
            vote_top_p=args.vote_top_p,
            vote_top_k=args.vote_top_k,
            vote_on_invalid_only=args.vote_on_invalid_only,
        )
        m = res["metrics"]
        summary[split] = {
            "accuracy": float(m["accuracy"]),
            "strict_accuracy": float(m["strict_accuracy"]),
            "macro_f1": float(m["macro_f1"]),
            "coverage": float(m["coverage"]),
            "num_total": int(m["num_total"]),
        }
        print(
            f"{split}: accuracy={m['accuracy']:.4f} | strict={m['strict_accuracy']:.4f} "
            f"| macro_f1={m['macro_f1']:.4f} | coverage={m['coverage']:.4f} | n={m['num_total']}"
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_json={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
