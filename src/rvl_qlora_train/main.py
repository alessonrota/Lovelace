from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .build_dataset import build_datasets_from_ocr
from .build_manifests import build_manifests
from .common import (
    PipelineLogger,
    atomic_write_json,
    check_class_ids_are_sparse_aware,
    load_class_map,
    now_iso_utc,
)
from .eval_15class import evaluate_dataset
from .ocr_paddle_cache import PaddleOCRConfig, collect_unique_docs, run_cached_paddle_ocr
from .premis import PremisRecorder
from .train_qlora import load_qlora_config, run_qlora_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA training pipeline for RVL-CDIP textual classification (15 classes)")

    p.add_argument("--labels-train", type=Path, required=True)
    p.add_argument("--labels-val", type=Path, required=True)
    p.add_argument("--labels-test", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)

    p.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--copy-ocr-from-run",
        type=Path,
        default=None,
        help="Optional source run directory (or OCR dir) to bootstrap OCR cache into this run.",
    )

    p.add_argument("--sample-train-per-class", type=int, default=2000)
    p.add_argument("--sample-val-per-class", type=int, default=300)
    p.add_argument("--sample-test-per-class", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-gpu-index", type=int, default=0, help="GPU index for model load/train/eval (default=0, RTX 4000 Ada).")

    p.add_argument("--ocr-engine", choices=["paddle"], default="paddle")
    p.add_argument("--paddle-variant", default="server_ch")
    p.add_argument("--paddle-lang", default="en")
    p.add_argument("--paddle-latin-lang", default="en")
    p.add_argument("--paddle-auto-script", action="store_true")
    p.add_argument("--paddle-device", choices=["auto", "cpu", "gpu"], default="gpu")
    p.add_argument("--paddle-det-limit-side-len", type=int, default=1920)
    p.add_argument("--paddle-det-db-thresh", type=float, default=0.2)
    p.add_argument("--paddle-det-db-box-thresh", type=float, default=0.45)
    p.add_argument("--paddle-det-db-unclip-ratio", type=float, default=2.0)
    p.add_argument("--paddle-rec-score-thresh", type=float, default=0.0)
    p.add_argument("--paddle-drop-score", type=float, default=0.0)
    p.add_argument("--ocr-workers", type=int, default=0, help="OCR workers (0=auto). For GPU, auto keeps 1 for stability.")
    p.add_argument("--ocr-max-cpu-util", type=float, default=0.6, help="Target CPU utilization when ocr-workers=0 in CPU mode.")
    p.add_argument(
        "--paddle-gpu-mem-fraction",
        type=float,
        default=0.6,
        help="Target fraction of GPU memory for PaddleOCR (best effort).",
    )
    p.add_argument(
        "--ocr-worker-threads",
        type=int,
        default=1,
        help="Thread count per OCR worker process (CPU mode).",
    )
    p.add_argument("--paddle-angle-cls", dest="paddle_angle_cls", action="store_true")
    p.add_argument("--no-paddle-angle-cls", dest="paddle_angle_cls", action="store_false")
    p.set_defaults(paddle_angle_cls=True)

    p.add_argument("--class-map-file", type=Path, default=Path("configs/qlora/rvl15_class_map.json"))
    p.add_argument("--system-prompt-file", type=Path, default=Path("configs/qlora/rvl_train_prompt_v1_minjson.txt"))
    p.add_argument("--qlora-config", type=Path, default=Path("configs/qlora/qlora_qwen14b_r1.yaml"))

    p.add_argument(
        "--legacy-sample-manifest",
        type=Path,
        default=Path("data/processed/saida-ocr-class/run_001/manifests/sample.jsonl"),
    )

    p.add_argument("--max-ocr-chars", type=int, default=12000)
    p.add_argument("--min-ocr-chars", type=int, default=40)
    p.add_argument("--max-eval-new-tokens", type=int, default=12)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument(
        "--skip-auto-eval",
        action="store_true",
        help="Skip automatic post-train evaluation. Use manual evaluation command later.",
    )

    p.add_argument("--output-root", type=Path, default=Path("data/processed/qlora-qwen14b"))

    return p.parse_args()


def ensure_run_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "run": run_dir,
        "manifests": run_dir / "manifests",
        "ocr": run_dir / "ocr",
        "datasets": run_dir / "datasets",
        "checkpoints": run_dir / "checkpoints",
        "evaluation": run_dir / "evaluation",
        "logs": run_dir / "logs",
        "premis": run_dir / "premis",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_or_init_state(state_path: Path, config: dict[str, Any], resume: bool) -> dict[str, Any]:
    if resume and state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    state = {
        "created_at": now_iso_utc(),
        "updated_at": now_iso_utc(),
        "config": config,
        "stages": {},
        "artifacts": {},
    }
    atomic_write_json(state_path, state)
    return state


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso_utc()
    atomic_write_json(state_path, state)


def _resolve_source_ocr_dir(source: Path) -> Path:
    source = source.resolve()
    if (source / "ocr").is_dir():
        return source / "ocr"
    if source.is_dir() and source.name == "ocr":
        return source
    raise RuntimeError(
        f"Invalid --copy-ocr-from-run path: {source}. Expected a run directory containing 'ocr/' or the OCR directory itself."
    )


def _bootstrap_ocr_cache_from_run(source_path: Path, target_ocr_dir: Path, logger: PipelineLogger) -> dict[str, Any]:
    source_ocr_dir = _resolve_source_ocr_dir(source_path)
    source_index_path = source_ocr_dir / "ocr_index.json"
    if not source_index_path.exists():
        raise RuntimeError(f"Source OCR index not found: {source_index_path}")

    raw = json.loads(source_index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Source OCR index has invalid format: {source_index_path}")

    target_ocr_dir.mkdir(parents=True, exist_ok=True)
    target_index_path = target_ocr_dir / "ocr_index.json"
    now = now_iso_utc()

    copied_ok_files = 0
    reused_ok_files = 0
    missing_ok_files = 0
    ok_entries = 0
    error_entries = 0

    target_index: dict[str, dict[str, Any]] = {}
    for doc_key, meta_raw in raw.items():
        doc_id = str(doc_key)
        meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}

        status = str(meta.get("status", "ocr_error"))
        src_txt = Path(str(meta.get("ocr_path", ""))) if meta.get("ocr_path") else (source_ocr_dir / f"{doc_id}.txt")
        dst_txt = target_ocr_dir / f"{doc_id}.txt"

        if status == "ok":
            if src_txt.exists():
                if not dst_txt.exists():
                    shutil.copy2(src_txt, dst_txt)
                    copied_ok_files += 1
                else:
                    reused_ok_files += 1
                ok_entries += 1
            else:
                status = "ocr_error"
                missing_ok_files += 1
                error_entries += 1
                meta["last_error"] = f"source_ocr_file_missing: {src_txt}"
        else:
            error_entries += 1

        out_meta = dict(meta)
        out_meta["doc_id"] = doc_id
        out_meta["status"] = status
        out_meta["ocr_path"] = str(dst_txt)
        out_meta["copied_from_ocr_dir"] = str(source_ocr_dir)
        out_meta["updated_at"] = now
        target_index[doc_id] = out_meta

    atomic_write_json(target_index_path, target_index)

    summary = {
        "generated_at": now,
        "source_ocr_dir": str(source_ocr_dir),
        "target_ocr_dir": str(target_ocr_dir),
        "target_index_file": str(target_index_path),
        "total_entries": len(target_index),
        "ok_entries": ok_entries,
        "error_entries": error_entries,
        "copied_ok_files": copied_ok_files,
        "reused_ok_files": reused_ok_files,
        "missing_ok_source_files": missing_ok_files,
    }
    logger.info(
        "OCR bootstrap copy finished: "
        f"entries={summary['total_entries']}, ok={ok_entries}, errors={error_entries}, "
        f"copied={copied_ok_files}, reused={reused_ok_files}, missing_ok_source={missing_ok_files}"
    )
    return summary


def main() -> int:
    args = parse_args()

    run_dir = args.output_root / args.run_id
    if args.resume:
        if not run_dir.exists():
            raise RuntimeError(f"--resume set but run directory not found: {run_dir}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"Run directory already exists and is not empty: {run_dir}")

    run_paths = ensure_run_dirs(run_dir)
    logger = PipelineLogger(run_paths["logs"] / "pipeline.log")

    class_map = load_class_map(args.class_map_file)
    check_class_ids_are_sparse_aware(class_map)

    if not args.system_prompt_file.exists():
        raise RuntimeError(f"System prompt file not found: {args.system_prompt_file}")
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8")

    if not args.images_root.exists():
        raise RuntimeError(f"Images root not found: {args.images_root}")

    for labels_path in [args.labels_train, args.labels_val, args.labels_test]:
        if not labels_path.exists():
            raise RuntimeError(f"Labels file not found: {labels_path}")

    premis = PremisRecorder(run_paths["premis"], args.run_id)
    premis.ensure_defaults(args.base_model)

    run_config = {
        "run_id": args.run_id,
        "labels_train": str(args.labels_train),
        "labels_val": str(args.labels_val),
        "labels_test": str(args.labels_test),
        "images_root": str(args.images_root),
        "base_model": args.base_model,
        "seed": args.seed,
        "sample_train_per_class": args.sample_train_per_class,
        "sample_val_per_class": args.sample_val_per_class,
        "sample_test_per_class": args.sample_test_per_class,
        "llm_gpu_index": args.llm_gpu_index,
        "class_map_file": str(args.class_map_file),
        "system_prompt_file": str(args.system_prompt_file),
        "qlora_config": str(args.qlora_config),
        "legacy_sample_manifest": str(args.legacy_sample_manifest),
        "ocr_engine": args.ocr_engine,
        "paddle_variant": args.paddle_variant,
        "paddle_lang": args.paddle_lang,
        "paddle_latin_lang": args.paddle_latin_lang,
        "paddle_auto_script": args.paddle_auto_script,
        "paddle_device": args.paddle_device,
        "paddle_angle_cls": args.paddle_angle_cls,
        "paddle_det_limit_side_len": args.paddle_det_limit_side_len,
        "paddle_det_db_thresh": args.paddle_det_db_thresh,
        "paddle_det_db_box_thresh": args.paddle_det_db_box_thresh,
        "paddle_det_db_unclip_ratio": args.paddle_det_db_unclip_ratio,
        "paddle_rec_score_thresh": args.paddle_rec_score_thresh,
        "paddle_drop_score": args.paddle_drop_score,
        "ocr_workers": args.ocr_workers,
        "ocr_max_cpu_util": args.ocr_max_cpu_util,
        "paddle_gpu_mem_fraction": args.paddle_gpu_mem_fraction,
        "ocr_worker_threads": args.ocr_worker_threads,
        "max_ocr_chars": args.max_ocr_chars,
        "min_ocr_chars": args.min_ocr_chars,
        "max_eval_new_tokens": args.max_eval_new_tokens,
        "evaluate_test": args.evaluate_test,
        "skip_auto_eval": args.skip_auto_eval,
        "copy_ocr_from_run": str(args.copy_ocr_from_run) if args.copy_ocr_from_run else None,
        "resume": args.resume,
        "started_at": now_iso_utc(),
    }

    state_path = run_paths["manifests"] / "state.json"
    state = load_or_init_state(state_path, run_config, resume=args.resume)

    logger.info(f"Run id: {args.run_id}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output run dir: {run_dir}")

    premis.add_event(
        "ingest",
        "started",
        {
            "labels_train": str(args.labels_train),
            "labels_val": str(args.labels_val),
            "labels_test": str(args.labels_test),
            "images_root": str(args.images_root),
            "legacy_sample_manifest": str(args.legacy_sample_manifest),
        },
        object_id="obj:labels",
        object_path=str(args.labels_train),
        agents=["agent:pipeline"],
    )

    manifest_bundle = build_manifests(
        manifests_dir=run_paths["manifests"],
        labels_train=args.labels_train,
        labels_val=args.labels_val,
        labels_test=args.labels_test,
        images_root=args.images_root,
        class_map=class_map,
        train_per_class=args.sample_train_per_class,
        val_per_class=args.sample_val_per_class,
        test_per_class=args.sample_test_per_class,
        seed=args.seed,
        legacy_sample_manifest=args.legacy_sample_manifest,
        resume=args.resume,
    )

    state["stages"]["sampling"] = {
        "status": "success",
        "at": now_iso_utc(),
        "summary": manifest_bundle.summary,
    }
    state["artifacts"]["sampling_summary"] = str(run_paths["manifests"] / "sampling_summary.json")
    save_state(state_path, state)

    premis.add_object("obj:sampling_summary", state["artifacts"]["sampling_summary"], fmt="application/json")
    premis.add_event(
        "sampling",
        "success",
        manifest_bundle.summary,
        object_id="obj:sampling_summary",
        object_path=state["artifacts"]["sampling_summary"],
        agents=["agent:pipeline"],
    )

    manifest_paths = [
        run_paths["manifests"] / "train_balanced_fast.jsonl",
        run_paths["manifests"] / "val_balanced.jsonl",
        run_paths["manifests"] / "test_holdout_balanced.jsonl",
        run_paths["manifests"] / "eval_legacy_93.jsonl",
    ]
    unique_docs = collect_unique_docs(manifest_paths)

    if args.copy_ocr_from_run:
        target_index = run_paths["ocr"] / "ocr_index.json"
        if args.resume and target_index.exists():
            logger.info(
                f"Skipping OCR bootstrap copy on resume because target index already exists: {target_index}"
            )
        else:
            ocr_copy_summary = _bootstrap_ocr_cache_from_run(
                source_path=args.copy_ocr_from_run,
                target_ocr_dir=run_paths["ocr"],
                logger=logger,
            )
            state["stages"]["ocr_bootstrap_copy"] = {
                "status": "success",
                "at": now_iso_utc(),
                "summary": ocr_copy_summary,
            }
            state["artifacts"]["ocr_bootstrap_index"] = str(run_paths["ocr"] / "ocr_index.json")
            save_state(state_path, state)
            premis.add_event(
                "ocr",
                "bootstrap_copy",
                ocr_copy_summary,
                object_id="obj:ocr_bootstrap_index",
                object_path=str(run_paths["ocr"] / "ocr_index.json"),
                agents=["agent:pipeline"],
            )

    paddle_cfg = PaddleOCRConfig(
        variant=args.paddle_variant,
        lang=args.paddle_lang,
        latin_lang=args.paddle_latin_lang,
        auto_script=args.paddle_auto_script,
        device=args.paddle_device,
        angle_cls=args.paddle_angle_cls,
        det_limit_side_len=args.paddle_det_limit_side_len,
        det_db_thresh=args.paddle_det_db_thresh,
        det_db_box_thresh=args.paddle_det_db_box_thresh,
        det_db_unclip_ratio=args.paddle_det_db_unclip_ratio,
        rec_score_thresh=args.paddle_rec_score_thresh,
        drop_score=args.paddle_drop_score,
        workers=args.ocr_workers,
        max_cpu_utilization=args.ocr_max_cpu_util,
        gpu_mem_fraction=args.paddle_gpu_mem_fraction,
        worker_threads=args.ocr_worker_threads,
    )

    ocr_summary = run_cached_paddle_ocr(
        docs=unique_docs,
        ocr_dir=run_paths["ocr"],
        logger=logger,
        premis=premis,
        cfg=paddle_cfg,
        resume=(args.resume or bool(args.copy_ocr_from_run)),
    )

    state["stages"]["ocr"] = {"status": "success", "at": now_iso_utc(), "summary": ocr_summary}
    state["artifacts"]["ocr_index"] = str(run_paths["ocr"] / "ocr_index.json")
    save_state(state_path, state)

    premis.add_event(
        "dataset_build",
        "started",
        {
            "max_ocr_chars": args.max_ocr_chars,
            "min_ocr_chars": args.min_ocr_chars,
        },
        object_id="batch:dataset_build",
        agents=["agent:pipeline"],
    )

    dataset_summary = build_datasets_from_ocr(
        datasets_dir=run_paths["datasets"],
        manifests_dir=run_paths["manifests"],
        ocr_index_path=run_paths["ocr"] / "ocr_index.json",
        class_map=class_map,
        system_prompt=system_prompt,
        max_ocr_chars=args.max_ocr_chars,
        min_ocr_chars=args.min_ocr_chars,
    )

    state["stages"]["dataset_build"] = {
        "status": "success",
        "at": now_iso_utc(),
        "summary": dataset_summary,
    }
    state["artifacts"]["dataset_summary"] = str(run_paths["datasets"] / "dataset_summary.json")
    save_state(state_path, state)

    premis.add_object("obj:dataset_summary", state["artifacts"]["dataset_summary"], fmt="application/json")
    premis.add_event(
        "dataset_build",
        "success",
        dataset_summary,
        object_id="obj:dataset_summary",
        object_path=state["artifacts"]["dataset_summary"],
        agents=["agent:pipeline"],
    )

    qlora_cfg = load_qlora_config(args.qlora_config, base_model_override=args.base_model)

    premis.add_event(
        "train",
        "started",
        {
            "base_model": qlora_cfg.base_model,
            "target_modules": qlora_cfg.target_modules,
            "lora_r": qlora_cfg.lora_r,
            "lora_alpha": qlora_cfg.lora_alpha,
            "lora_dropout": qlora_cfg.lora_dropout,
            "max_seq_len": qlora_cfg.max_seq_len,
            "max_train_hours": qlora_cfg.max_train_hours,
        },
        object_id="batch:train",
        agents=["agent:pipeline", "agent:trainer", "agent:base_model"],
    )

    train_summary = run_qlora_training(
        train_jsonl=run_paths["datasets"] / "train_balanced_fast.jsonl",
        val_jsonl=run_paths["datasets"] / "val_balanced.jsonl",
        checkpoints_dir=run_paths["checkpoints"],
        logger=logger,
        config=qlora_cfg,
        resume=args.resume,
        gpu_index=args.llm_gpu_index,
    )

    state["stages"]["train"] = {"status": "success", "at": now_iso_utc(), "summary": train_summary}
    state["artifacts"]["adapter_dir"] = str(run_paths["checkpoints"] / "adapter_final")
    state["artifacts"]["training_summary"] = str(run_paths["checkpoints"] / "training_summary.json")
    save_state(state_path, state)

    premis.add_object("obj:training_summary", state["artifacts"]["training_summary"], fmt="application/json")
    premis.add_event(
        "train",
        "success",
        train_summary,
        object_id="obj:training_summary",
        object_path=state["artifacts"]["training_summary"],
        agents=["agent:pipeline", "agent:trainer"],
    )

    evaluation_summary: dict[str, Any] = {"generated_at": now_iso_utc(), "splits": {}}

    if args.skip_auto_eval:
        evaluation_summary = {
            "generated_at": now_iso_utc(),
            "skipped": True,
            "reason": "--skip-auto-eval",
            "splits": {},
        }
        logger.info("Auto-evaluation skipped (--skip-auto-eval).")
        premis.add_event(
            "evaluation",
            "skipped",
            {"reason": "--skip-auto-eval"},
            object_id="batch:evaluation",
            agents=["agent:pipeline"],
        )
    else:
        eval_targets = {
            "eval_legacy_93": run_paths["datasets"] / "eval_legacy_93.jsonl",
            "val_balanced": run_paths["datasets"] / "val_balanced.jsonl",
        }
        if args.evaluate_test:
            eval_targets["test_holdout_balanced"] = run_paths["datasets"] / "test_holdout_balanced.jsonl"

        for split_name, dataset_path in eval_targets.items():
            premis.add_event(
                "evaluation",
                "started",
                {"split": split_name, "dataset": str(dataset_path)},
                object_id=f"batch:evaluation:{split_name}",
                agents=["agent:pipeline", "agent:base_model"],
            )

            split_out_dir = run_paths["evaluation"] / split_name
            split_result = evaluate_dataset(
                dataset_jsonl=dataset_path,
                base_model=args.base_model,
                adapter_dir=run_paths["checkpoints"] / "adapter_final",
                out_dir=split_out_dir,
                class_map=class_map,
                logger=logger,
                max_new_tokens=args.max_eval_new_tokens,
                gpu_index=args.llm_gpu_index,
            )

            evaluation_summary["splits"][split_name] = split_result["metrics"]

            premis.add_event(
                "evaluation",
                "success",
                {
                    "split": split_name,
                    "metrics_json": split_result["metrics_json"],
                    "accuracy": split_result["metrics"]["accuracy"],
                    "strict_accuracy": split_result["metrics"]["strict_accuracy"],
                    "macro_f1": split_result["metrics"]["macro_f1"],
                    "coverage": split_result["metrics"]["coverage"],
                },
                object_id=f"obj:metrics:{split_name}",
                object_path=split_result["metrics_json"],
                agents=["agent:pipeline"],
            )

    atomic_write_json(run_paths["evaluation"] / "summary.json", evaluation_summary)

    state["stages"]["evaluation"] = {
        "status": "skipped" if args.skip_auto_eval else "success",
        "at": now_iso_utc(),
        "summary": evaluation_summary,
    }
    state["artifacts"]["evaluation_summary"] = str(run_paths["evaluation"] / "summary.json")
    save_state(state_path, state)

    premis.add_object("obj:evaluation_summary", state["artifacts"]["evaluation_summary"], fmt="application/json")
    premis.add_event(
        "export",
        "success",
        {
            "state": str(state_path),
            "evaluation_summary": state["artifacts"]["evaluation_summary"],
            "adapter_dir": state["artifacts"]["adapter_dir"],
        },
        object_id="obj:evaluation_summary",
        object_path=state["artifacts"]["evaluation_summary"],
        agents=["agent:pipeline"],
    )

    logger.info("QLoRA training pipeline finished successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise
