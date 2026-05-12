#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rvl_text_pipeline.main import (
    PipelineLogger,
    PremisRecorder,
    create_paddle_ocr,
    ensure_run_dirs,
    export_results,
    load_class_map,
    load_or_init_state,
    now_iso_utc,
    progress_line,
    read_jsonl,
    resolve_paddle_lang,
    resolve_paddle_use_gpu,
    run_llm_stage,
    save_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PaddleOCR + LLM and compare with an existing Tesseract run")
    p.add_argument("--run-id", required=True, help="New isolated run id for PaddleOCR comparison")
    p.add_argument("--output-root", type=Path, default=Path("data/processed/ocr-tests/paddle-vs-tesseract"))
    p.add_argument(
        "--sample-manifest",
        type=Path,
        default=Path("data/processed/saida-ocr-class/run_001/manifests/sample.jsonl"),
    )
    p.add_argument("--class-map-file", type=Path, default=Path("configs/rvl_class_map.json"))
    p.add_argument("--system-prompt-file", type=Path, default=Path("configs/rvl_system_prompt.txt"))
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434/api/generate")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--request-timeout", type=int, default=180)
    p.add_argument("--max-ocr-chars", type=int, default=12000)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("data/processed/saida-ocr-class/run_009_prompt_v5_32full_best_eng/evaluation/metrics.json"),
    )
    p.add_argument(
        "--baseline-state",
        type=Path,
        default=Path("data/processed/saida-ocr-class/run_009_prompt_v5_32full_best_eng/manifests/state.json"),
    )
    p.add_argument("--paddle-lang", default="latin")
    p.add_argument("--paddle-latin-lang", default="latin")
    p.add_argument("--paddle-auto-script", action="store_true")
    p.add_argument("--paddle-device", choices=["auto", "cpu", "gpu"], default="auto")
    p.add_argument("--paddle-det-limit-side-len", type=int, default=1920)
    p.add_argument("--paddle-det-db-thresh", type=float, default=0.2)
    p.add_argument("--paddle-det-db-box-thresh", type=float, default=0.45)
    p.add_argument("--paddle-det-db-unclip-ratio", type=float, default=2.0)
    p.add_argument("--paddle-rec-score-thresh", type=float, default=0.0)
    p.add_argument("--paddle-drop-score", type=float, default=0.0)
    p.add_argument("--paddle-angle-cls", dest="paddle_angle_cls", action="store_true")
    p.add_argument("--no-paddle-angle-cls", dest="paddle_angle_cls", action="store_false")
    p.set_defaults(paddle_angle_cls=True)
    return p.parse_args()


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
        # Older PaddleOCR returns tuple(text, score); newer variants may return list.
        if not isinstance(rec, (list, tuple)) or not rec:
            continue
        txt = rec[0]
        if isinstance(txt, str):
            txt = txt.strip()
            if txt:
                lines.append(txt)
    return "\n".join(lines).strip()


def run_paddle_ocr_stage(
    state: dict[str, Any],
    run_paths: dict[str, Path],
    logger: PipelineLogger,
    premis: PremisRecorder,
    lang: str,
    paddle_latin_lang: str,
    paddle_auto_script: bool,
    paddle_device: str,
    paddle_angle_cls: bool,
    paddle_det_limit_side_len: int,
    paddle_det_db_thresh: float,
    paddle_det_db_box_thresh: float,
    paddle_det_db_unclip_ratio: float,
    paddle_rec_score_thresh: float,
    paddle_drop_score: float,
) -> None:
    pending = [d for d in state["docs"].values() if d["status"] == "pending"]
    if not pending:
        logger.info("PaddleOCR stage skipped: no pending documents")
        return

    total = len(pending)
    started = time.perf_counter()

    # Keep runtime deterministic and skip model-host checks (already cached after first run).
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    use_gpu = resolve_paddle_use_gpu(paddle_device)
    ocr_cache: dict[str, Any] = {}

    premis.add_event(
        "ocr",
        "started",
        {
            "total_docs": total,
            "engine": "paddleocr",
            "lang": lang,
            "latin_lang": paddle_latin_lang,
            "auto_script": paddle_auto_script,
            "device_mode": paddle_device,
            "use_gpu": use_gpu,
            "angle_cls": paddle_angle_cls,
            "det_limit_side_len": paddle_det_limit_side_len,
            "det_db_thresh": paddle_det_db_thresh,
            "det_db_box_thresh": paddle_det_db_box_thresh,
            "det_db_unclip_ratio": paddle_det_db_unclip_ratio,
            "rec_score_thresh": paddle_rec_score_thresh,
            "drop_score": paddle_drop_score,
        },
        object_id="batch:ocr:paddle",
        agents=["agent:pipeline"],
    )

    done = 0
    for doc in pending:
        done += 1
        doc_id = doc["doc_id"]
        image_path = Path(doc["image_path"])
        ocr_path = run_paths["ocr"] / f"{doc_id}.txt"
        doc["ocr_path"] = str(ocr_path)

        try:
            if not image_path.exists():
                raise RuntimeError(f"image_not_found: {image_path}")

            lang_used, script_used = resolve_paddle_lang(
                image_path=image_path,
                paddle_lang=lang,
                paddle_auto_script=paddle_auto_script,
                paddle_latin_lang=paddle_latin_lang,
            )
            if lang_used not in ocr_cache:
                ocr_cache[lang_used] = create_paddle_ocr(
                    lang=lang_used,
                    use_gpu=use_gpu,
                    angle_cls=paddle_angle_cls,
                    det_limit_side_len=paddle_det_limit_side_len,
                    det_db_thresh=paddle_det_db_thresh,
                    det_db_box_thresh=paddle_det_db_box_thresh,
                    det_db_unclip_ratio=paddle_det_db_unclip_ratio,
                    rec_score_thresh=paddle_rec_score_thresh,
                    drop_score=paddle_drop_score,
                )
            result = ocr_cache[lang_used].ocr(str(image_path), cls=paddle_angle_cls)
            text = extract_text_from_paddle_result(result)

            if not text:
                raise RuntimeError("empty_ocr_text")

            ocr_path.write_text(text, encoding="utf-8")
            doc["status"] = "ocr_done"
            doc["ocr_engine"] = "paddleocr"
            doc["ocr_lang_used"] = lang_used
            doc["ocr_script"] = script_used
            doc["last_error"] = None
            doc["updated_at"] = now_iso_utc()

            premis.add_object(f"obj:ocr:{doc_id}", str(ocr_path), fmt="text/plain")
            premis.add_event(
                "ocr",
                "success",
                {"doc_id": doc_id, "chars": len(text), "engine": "paddleocr"},
                object_id=f"obj:ocr:{doc_id}",
                object_path=str(ocr_path),
                agents=["agent:pipeline"],
            )
        except Exception as exc:
            doc["status"] = "ocr_error"
            doc["ocr_engine"] = "paddleocr"
            doc["last_error"] = str(exc)
            doc["updated_at"] = now_iso_utc()
            premis.add_event(
                "ocr",
                "failure",
                {"doc_id": doc_id, "error": str(exc), "engine": "paddleocr"},
                object_id=f"obj:image:{doc_id}",
                object_path=str(image_path),
                agents=["agent:pipeline"],
            )
            logger.warn(f"PaddleOCR failed for {doc_id}: {exc}")

        save_state(run_paths["manifests"], state)
        logger.info(progress_line("PaddleOCR", done, total, started))


def load_baseline_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Baseline metrics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_baseline_ocr_chars(state_path: Path) -> dict[str, int]:
    if not state_path.exists():
        return {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    docs = state.get("docs", {})
    chars_by_doc: dict[str, int] = {}
    for doc_id, doc in docs.items():
        if doc.get("status") != "ok":
            continue
        p = doc.get("ocr_path")
        if not p:
            continue
        try:
            txt = Path(p).read_text(encoding="utf-8")
        except Exception:
            continue
        chars_by_doc[doc_id] = len(txt)
    return chars_by_doc


def build_comparison(
    baseline_metrics: dict[str, Any],
    paddle_metrics: dict[str, Any],
    baseline_chars: dict[str, int],
    paddle_state: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}

    out["baseline"] = {
        "accuracy": baseline_metrics["accuracy"],
        "strict_accuracy": baseline_metrics["strict_accuracy"],
        "macro_f1": baseline_metrics["macro_f1"],
        "coverage": baseline_metrics["coverage"],
        "status_counts": baseline_metrics.get("status_counts", {}),
    }
    out["paddle"] = {
        "accuracy": paddle_metrics["accuracy"],
        "strict_accuracy": paddle_metrics["strict_accuracy"],
        "macro_f1": paddle_metrics["macro_f1"],
        "coverage": paddle_metrics["coverage"],
        "status_counts": paddle_metrics.get("status_counts", {}),
    }
    out["delta"] = {
        "accuracy": paddle_metrics["accuracy"] - baseline_metrics["accuracy"],
        "strict_accuracy": paddle_metrics["strict_accuracy"] - baseline_metrics["strict_accuracy"],
        "macro_f1": paddle_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "coverage": paddle_metrics["coverage"] - baseline_metrics["coverage"],
    }

    paddle_chars: dict[str, int] = {}
    for doc_id, doc in paddle_state.get("docs", {}).items():
        if doc.get("status") != "ok":
            continue
        p = doc.get("ocr_path")
        if not p:
            continue
        try:
            txt = Path(p).read_text(encoding="utf-8")
        except Exception:
            continue
        paddle_chars[doc_id] = len(txt)

    common = sorted(set(baseline_chars).intersection(paddle_chars))
    if common:
        avg_base = sum(baseline_chars[d] for d in common) / len(common)
        avg_pad = sum(paddle_chars[d] for d in common) / len(common)
    else:
        avg_base = 0.0
        avg_pad = 0.0
    out["ocr_text_stats_common_docs"] = {
        "common_docs": len(common),
        "avg_chars_tesseract": avg_base,
        "avg_chars_paddle": avg_pad,
        "delta_avg_chars": avg_pad - avg_base,
    }
    return out


def main() -> None:
    args = parse_args()
    run_paths = ensure_run_dirs(args.output_root / args.run_id)
    logger = PipelineLogger(run_paths["logs"] / "pipeline.log")
    premis = PremisRecorder(run_paths["premis"], args.run_id, logger)
    premis.ensure_defaults(args.model)

    class_map = load_class_map(args.class_map_file)
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    sample_rows = read_jsonl(args.sample_manifest)
    if not sample_rows:
        raise RuntimeError(f"Sample manifest is empty: {args.sample_manifest}")

    run_config = {
        "run_id": args.run_id,
        "sample_manifest_source": str(args.sample_manifest),
        "model": args.model,
        "system_prompt_file": str(args.system_prompt_file),
        "ocr_engine": "paddleocr",
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
        "max_retries": args.max_retries,
        "request_timeout": args.request_timeout,
        "max_ocr_chars": args.max_ocr_chars,
        "num_predict": args.num_predict,
        "started_at": now_iso_utc(),
    }

    state = load_or_init_state(run_paths["manifests"], sample_rows, run_config, args.resume)

    logger.info(f"Run id: {args.run_id}")
    logger.info(f"Sample manifest: {args.sample_manifest}")
    logger.info(f"Model: {args.model}")
    logger.info(
        "PaddleOCR config: "
        f"lang={args.paddle_lang}, latin_lang={args.paddle_latin_lang}, "
        f"auto_script={args.paddle_auto_script}, device={args.paddle_device}, "
        f"angle_cls={args.paddle_angle_cls}, det_limit_side_len={args.paddle_det_limit_side_len}, "
        f"det_db_thresh={args.paddle_det_db_thresh}, det_db_box_thresh={args.paddle_det_db_box_thresh}, "
        f"det_db_unclip_ratio={args.paddle_det_db_unclip_ratio}, "
        f"rec_score_thresh={args.paddle_rec_score_thresh}, drop_score={args.paddle_drop_score}"
    )

    premis.add_event(
        "ingest",
        "success",
        {"sample_size": len(sample_rows), "sample_manifest": str(args.sample_manifest)},
        object_id="batch:sample",
        object_path=str(args.sample_manifest),
        agents=["agent:pipeline"],
    )

    run_paddle_ocr_stage(
        state=state,
        run_paths=run_paths,
        logger=logger,
        premis=premis,
        lang=args.paddle_lang,
        paddle_latin_lang=args.paddle_latin_lang,
        paddle_auto_script=args.paddle_auto_script,
        paddle_device=args.paddle_device,
        paddle_angle_cls=args.paddle_angle_cls,
        paddle_det_limit_side_len=args.paddle_det_limit_side_len,
        paddle_det_db_thresh=args.paddle_det_db_thresh,
        paddle_det_db_box_thresh=args.paddle_det_db_box_thresh,
        paddle_det_db_unclip_ratio=args.paddle_det_db_unclip_ratio,
        paddle_rec_score_thresh=args.paddle_rec_score_thresh,
        paddle_drop_score=args.paddle_drop_score,
    )

    run_llm_stage(
        state=state,
        run_paths=run_paths,
        logger=logger,
        premis=premis,
        class_map=class_map,
        system_prompt=system_prompt,
        model=args.model,
        endpoint=args.ollama_endpoint,
        max_retries=args.max_retries,
        timeout=args.request_timeout,
        max_ocr_chars=args.max_ocr_chars,
        num_predict=args.num_predict,
        fewshot_block="",
    )

    save_state(run_paths["manifests"], state)

    paddle_metrics = export_results(
        state=state,
        run_paths=run_paths,
        logger=logger,
        premis=premis,
        class_map=class_map,
    )

    baseline_metrics = load_baseline_metrics(args.baseline_metrics)
    baseline_chars = load_baseline_ocr_chars(args.baseline_state)
    comparison = build_comparison(baseline_metrics, paddle_metrics, baseline_chars, state)
    comparison["created_at"] = now_iso_utc()
    comparison["baseline_metrics_file"] = str(args.baseline_metrics)
    comparison["baseline_state_file"] = str(args.baseline_state)

    comparison_path = run_paths["evaluation"] / "comparison_vs_tesseract.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Comparison saved: {comparison_path}")


if __name__ == "__main__":
    main()
