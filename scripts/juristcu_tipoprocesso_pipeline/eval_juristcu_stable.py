#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "src").exists() and (p / "configs").exists():
            return p
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rvl_qlora_train.common import PipelineLogger, load_class_map
from src.rvl_qlora_train.eval_15class import evaluate_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stable final evaluation for JurisTCU TIPOPROCESSO LoRA (Top15)."
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


def main() -> int:
    args = parse_args()
    run_root = args.output_root / args.run_id
    if not run_root.exists():
        raise RuntimeError(f"Run root not found: {run_root}")

    adapter_dir = run_root / "checkpoints" / "adapter_final"
    if not adapter_dir.exists():
        raise RuntimeError(f"Adapter not found: {adapter_dir}")

    datasets_dir = run_root / "datasets"
    class_map_path = run_root / "configs" / "class_map.json"
    if not datasets_dir.exists():
        raise RuntimeError(f"Datasets dir not found: {datasets_dir}")
    if not class_map_path.exists():
        raise RuntimeError(f"class_map.json not found: {class_map_path}")

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = run_root / f"evaluation_manual_stable_rest_full_{ts}"
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

        res = evaluate_dataset(
            dataset_jsonl=dataset_jsonl,
            base_model=args.base_model,
            adapter_dir=adapter_dir,
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
