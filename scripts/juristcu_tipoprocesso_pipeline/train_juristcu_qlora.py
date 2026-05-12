#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
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

from src.rvl_qlora_train.common import PipelineLogger, atomic_write_json, now_iso_utc
from src.rvl_qlora_train.train_qlora import load_qlora_config, run_qlora_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train JurisTCU TIPOPROCESSO LoRA (Top15) with QLoRA Qwen14B. "
            "No evaluation during training; frequent checkpoints for resume."
        )
    )
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "juristcu-tipoprocesso",
    )
    p.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument(
        "--qlora-config",
        type=Path,
        default=REPO_ROOT / "configs" / "qlora" / "qlora_qwen14b_r2_qvko_lowmem.yaml",
    )
    p.add_argument("--gpu-index", type=int, default=0)

    # Required behavior from plan
    p.add_argument("--save-steps", type=int, default=25)
    p.add_argument("--save-total-limit", type=int, default=6)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--disable-eval-during-train", dest="disable_eval_during_train", action="store_true")
    p.add_argument("--enable-eval-during-train", dest="disable_eval_during_train", action="store_false")
    p.set_defaults(disable_eval_during_train=True)
    p.add_argument("--max-train-hours", type=float, default=None)

    p.add_argument("--force-resume", action="store_true")
    p.add_argument("--force-no-resume", action="store_true")
    return p.parse_args()


def _has_checkpoint(checkpoints_dir: Path) -> bool:
    return any(p.is_dir() and p.name.startswith("checkpoint-") for p in checkpoints_dir.glob("checkpoint-*"))


def _write_empty_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    datasets_dir = run_dir / "datasets"
    checkpoints_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger(logs_dir / "train.log")
    logger.info(f"Starting training run_id={args.run_id}")

    train_jsonl = datasets_dir / "train.jsonl"
    val_jsonl = datasets_dir / "val.jsonl"
    test_jsonl = datasets_dir / "test.jsonl"
    for p in [train_jsonl, val_jsonl, test_jsonl]:
        if not p.exists():
            raise RuntimeError(f"Required dataset file not found: {p}")

    if not args.qlora_config.exists():
        raise RuntimeError(f"QLoRA config not found: {args.qlora_config}")

    config = load_qlora_config(args.qlora_config, base_model_override=args.base_model)
    config.save_steps = int(args.save_steps)
    config.save_total_limit = int(args.save_total_limit)
    config.max_seq_len = int(args.max_seq_len)
    config.gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    if args.max_train_hours is not None:
        config.max_train_hours = float(args.max_train_hours)

    train_val_path = val_jsonl
    if args.disable_eval_during_train:
        train_val_path = datasets_dir / "_train_no_eval_val.jsonl"
        _write_empty_jsonl(train_val_path)
        logger.info("Evaluation during training disabled (using empty val dataset for trainer).")

    if args.force_resume and args.force_no_resume:
        raise RuntimeError("Use only one of --force-resume or --force-no-resume.")

    if args.force_resume:
        resume = True
    elif args.force_no_resume:
        resume = False
    else:
        resume = _has_checkpoint(checkpoints_dir)

    logger.info(f"Resume mode: {resume}")
    logger.info(
        "Effective training overrides: "
        f"save_steps={config.save_steps}, save_total_limit={config.save_total_limit}, "
        f"max_seq_len={config.max_seq_len}, gradient_accumulation_steps={config.gradient_accumulation_steps}"
    )

    result = run_qlora_training(
        train_jsonl=train_jsonl,
        val_jsonl=train_val_path,
        checkpoints_dir=checkpoints_dir,
        logger=logger,
        config=config,
        resume=resume,
        gpu_index=args.gpu_index,
    )

    wrapper_report: dict[str, Any] = {
        "generated_at": now_iso_utc(),
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "train_dataset": str(train_jsonl),
        "val_dataset_for_final_eval": str(val_jsonl),
        "test_dataset_for_final_eval": str(test_jsonl),
        "val_dataset_used_during_training": str(train_val_path),
        "evaluation_during_training": False,
        "resume": resume,
        "base_model": args.base_model,
        "qlora_config_source": str(args.qlora_config),
        "effective_overrides": {
            "save_steps": config.save_steps,
            "save_total_limit": config.save_total_limit,
            "max_seq_len": config.max_seq_len,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_train_hours": config.max_train_hours,
        },
        "training_summary_file": str(checkpoints_dir / "training_summary.json"),
        "training_result": result,
    }
    atomic_write_json(reports_dir / "train_run_config.json", wrapper_report)

    logger.info(f"Training completed. Adapter: {checkpoints_dir / 'adapter_final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
