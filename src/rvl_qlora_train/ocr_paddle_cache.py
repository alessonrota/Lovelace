from __future__ import annotations

import inspect
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import atomic_write_json, now_iso_utc, progress_line, read_jsonl
from .premis import PremisRecorder


@dataclass
class PaddleOCRConfig:
    variant: str = "server_ch"
    lang: str = "en"
    latin_lang: str = "en"
    auto_script: bool = False
    device: str = "gpu"
    angle_cls: bool = True
    det_limit_side_len: int = 1920
    det_db_thresh: float = 0.2
    det_db_box_thresh: float = 0.45
    det_db_unclip_ratio: float = 2.0
    rec_score_thresh: float = 0.0
    drop_score: float = 0.0
    workers: int = 0
    max_cpu_utilization: float = 0.6
    gpu_mem_fraction: float = 0.6
    worker_threads: int = 1


def _resolve_use_gpu(device: str) -> bool:
    if device == "gpu":
        return True
    if device == "cpu":
        return False
    try:
        import paddle

        return bool(paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
    except Exception:
        return False


def _resolve_worker_count(use_gpu: bool, requested_workers: int, max_cpu_utilization: float) -> int:
    if requested_workers > 0:
        return requested_workers
    if use_gpu:
        # Keep GPU OCR single-worker by default for stability/resume consistency.
        return 1
    cpu_total = os.cpu_count() or 1
    util = max(0.1, min(1.0, max_cpu_utilization))
    return max(1, int(cpu_total * util))


def _clamp_gpu_mem_fraction(v: float) -> float:
    return max(0.1, min(1.0, float(v)))


def _detect_script_with_tesseract_osd(image_path: Path) -> str | None:
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return None

    try:
        with Image.open(image_path) as img:
            osd_text = pytesseract.image_to_osd(img)
    except Exception:
        return None

    m = re.search(r"Script:\s*([A-Za-z]+)", osd_text)
    if not m:
        return None
    return m.group(1)


def _resolve_lang_for_doc(image_path: Path, cfg: PaddleOCRConfig) -> tuple[str, str | None]:
    if not cfg.auto_script:
        return cfg.lang, None
    script = _detect_script_with_tesseract_osd(image_path)
    if not script:
        return cfg.lang, None
    if script.lower() == "latin":
        return cfg.latin_lang, script
    return cfg.lang, script


def _collect_text_fragments(obj: Any, out: list[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        txt = obj.strip()
        if txt:
            out.append(txt)
        return
    if isinstance(obj, dict):
        for key in ("rec_texts", "texts", "text", "ocr_text", "transcription", "label"):
            if key not in obj:
                continue
            _collect_text_fragments(obj.get(key), out)
        for v in obj.values():
            if isinstance(v, (dict, list, tuple)):
                _collect_text_fragments(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_text_fragments(item, out)


def _extract_text_from_paddle_result(result: Any) -> str:
    # Legacy PaddleOCR 2.x output from .ocr
    if isinstance(result, list) and result:
        blob = result[0]
        if isinstance(blob, list):
            lines: list[str] = []
            for item in blob:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                rec = item[1]
                if not isinstance(rec, (list, tuple)) or not rec:
                    continue
                text = rec[0]
                if isinstance(text, str):
                    text = text.strip()
                    if text:
                        lines.append(text)
            if lines:
                return "\n".join(lines).strip()

    # PaddleOCR 3.x output from .predict (or any nested shape)
    fragments: list[str] = []
    _collect_text_fragments(result, fragments)
    return "\n".join(fragments).strip()


def _create_paddle_ocr(lang: str, cfg: PaddleOCRConfig, use_gpu: bool) -> Any:
    from paddleocr import PaddleOCR

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    params = set(inspect.signature(PaddleOCR.__init__).parameters.keys())

    # Legacy 2.x API
    if "use_gpu" in params:
        kwargs = {
            "lang": lang,
            "show_log": False,
            "use_gpu": use_gpu,
            "use_angle_cls": cfg.angle_cls,
            "det_limit_side_len": cfg.det_limit_side_len,
            "det_db_thresh": cfg.det_db_thresh,
            "det_db_box_thresh": cfg.det_db_box_thresh,
            "det_db_unclip_ratio": cfg.det_db_unclip_ratio,
            "rec_score_thresh": cfg.rec_score_thresh,
            "drop_score": cfg.drop_score,
        }
        if "gpu_mem" in params and use_gpu:
            # PaddleOCR 2.x expects GPU memory in MB.
            try:
                total_mb = 0
                try:
                    import paddle

                    total_mb = int(paddle.device.cuda.get_device_properties(0).total_memory // (1024 * 1024))
                except Exception:
                    total_mb = 0
                if total_mb > 0:
                    kwargs["gpu_mem"] = int(total_mb * _clamp_gpu_mem_fraction(cfg.gpu_mem_fraction))
            except Exception:
                pass
        try:
            return PaddleOCR(**kwargs)
        except TypeError:
            fallback = {
                "lang": lang,
                "show_log": False,
                "use_gpu": use_gpu,
                "use_angle_cls": cfg.angle_cls,
            }
            return PaddleOCR(**fallback)

    # Modern 3.x API
    device = "gpu:0" if use_gpu else "cpu"
    kwargs = {
        "lang": lang,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": cfg.angle_cls,
        "text_det_limit_side_len": cfg.det_limit_side_len,
        "text_det_thresh": cfg.det_db_thresh,
        "text_det_box_thresh": cfg.det_db_box_thresh,
        "text_det_unclip_ratio": cfg.det_db_unclip_ratio,
        "text_rec_score_thresh": cfg.rec_score_thresh,
        "device": device,
    }
    return PaddleOCR(**kwargs)


def _run_paddle(ocr_engine: Any, image_path: Path, angle_cls: bool) -> Any:
    if hasattr(ocr_engine, "ocr"):
        try:
            return ocr_engine.ocr(str(image_path), cls=angle_cls)
        except TypeError:
            return ocr_engine.ocr(str(image_path))
    if hasattr(ocr_engine, "predict"):
        return ocr_engine.predict(str(image_path))
    raise RuntimeError("PaddleOCR instance has neither .ocr nor .predict")


def _prepare_image_for_paddle(image_path: Path, temp_dir: Path) -> Path:
    ext = image_path.suffix.lower()
    if ext not in {".tif", ".tiff"}:
        return image_path

    temp_dir.mkdir(parents=True, exist_ok=True)
    converted = temp_dir / f"{image_path.stem}.png"
    if converted.exists() and converted.stat().st_mtime >= image_path.stat().st_mtime:
        return converted

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "PIL/Pillow is required to convert TIFF images for PaddleOCR. Install: pip install pillow"
        ) from exc

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb.save(converted, format="PNG")
    return converted


def _calc_ocr_quality(text: str) -> dict[str, Any]:
    chars = len(text)
    if chars == 0:
        return {"chars": 0, "words": 0, "alnum_ratio": 0.0, "line_count": 0}
    alnum = sum(ch.isalnum() for ch in text)
    words = len([w for w in re.split(r"\s+", text) if w])
    lines = len([ln for ln in text.splitlines() if ln.strip()])
    return {
        "chars": chars,
        "words": words,
        "alnum_ratio": alnum / chars,
        "line_count": lines,
    }


def _load_index(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        return {}
    try:
        raw = index_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): dict(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _save_index(index_path: Path, index: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(index_path, index)


def _cfg_to_worker_payload(cfg: PaddleOCRConfig) -> dict[str, Any]:
    return {
        "variant": cfg.variant,
        "lang": cfg.lang,
        "latin_lang": cfg.latin_lang,
        "auto_script": cfg.auto_script,
        "device": cfg.device,
        "angle_cls": cfg.angle_cls,
        "det_limit_side_len": cfg.det_limit_side_len,
        "det_db_thresh": cfg.det_db_thresh,
        "det_db_box_thresh": cfg.det_db_box_thresh,
        "det_db_unclip_ratio": cfg.det_db_unclip_ratio,
        "rec_score_thresh": cfg.rec_score_thresh,
        "drop_score": cfg.drop_score,
        "workers": cfg.workers,
        "max_cpu_utilization": cfg.max_cpu_utilization,
        "gpu_mem_fraction": cfg.gpu_mem_fraction,
        "worker_threads": cfg.worker_threads,
    }


_WORKER_OCR_CACHE: dict[str, Any] = {}
_WORKER_CFG: PaddleOCRConfig | None = None
_WORKER_USE_GPU: bool | None = None


def _worker_process_doc(task: dict[str, Any]) -> dict[str, Any]:
    global _WORKER_CFG, _WORKER_USE_GPU

    cfg_raw = task["cfg"]
    if _WORKER_CFG is None:
        _WORKER_CFG = PaddleOCRConfig(**cfg_raw)
    if _WORKER_USE_GPU is None:
        _WORKER_USE_GPU = bool(task["use_gpu"])

    worker_threads = max(1, int(_WORKER_CFG.worker_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(worker_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(worker_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(worker_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(worker_threads))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    if _WORKER_USE_GPU:
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", str(_clamp_gpu_mem_fraction(_WORKER_CFG.gpu_mem_fraction)))

    row = task["row"]
    doc_id = str(row["doc_id"])
    image_path = Path(str(row["image_path"]))
    temp_dir = Path(str(task["temp_dir"]))

    lang_used, script_used = _resolve_lang_for_doc(image_path, _WORKER_CFG)
    if lang_used not in _WORKER_OCR_CACHE:
        _WORKER_OCR_CACHE[lang_used] = _create_paddle_ocr(lang_used, cfg=_WORKER_CFG, use_gpu=_WORKER_USE_GPU)

    paddle_input = _prepare_image_for_paddle(image_path, temp_dir=temp_dir)
    result = _run_paddle(_WORKER_OCR_CACHE[lang_used], paddle_input, angle_cls=_WORKER_CFG.angle_cls)
    text = _extract_text_from_paddle_result(result)
    if not text:
        raise RuntimeError("empty_ocr_text")
    quality = _calc_ocr_quality(text)

    return {
        "doc_id": doc_id,
        "text": text,
        "quality": quality,
        "ocr_lang_used": lang_used,
        "ocr_script": script_used,
    }


def collect_unique_docs(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in manifest_paths:
        rows = read_jsonl(path)
        for row in rows:
            doc_id = str(row["doc_id"])
            if doc_id not in merged:
                merged[doc_id] = row
    docs = list(merged.values())
    docs.sort(key=lambda r: str(r["doc_id"]))
    return docs


def run_cached_paddle_ocr(
    docs: list[dict[str, Any]],
    ocr_dir: Path,
    logger: Any,
    premis: PremisRecorder,
    cfg: PaddleOCRConfig,
    resume: bool,
) -> dict[str, Any]:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        import paddle as _  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "paddlepaddle is not installed in this environment. Install: pip install paddlepaddle"
        ) from exc

    try:
        import paddleocr as _  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PaddleOCR is not installed in this environment. Install: pip install paddleocr paddlepaddle"
        ) from exc

    ocr_dir.mkdir(parents=True, exist_ok=True)
    index_path = ocr_dir / "ocr_index.json"
    temp_img_dir = ocr_dir / "_tmp_images"

    index = _load_index(index_path) if resume else {}
    use_gpu = _resolve_use_gpu(cfg.device)
    workers = _resolve_worker_count(use_gpu=use_gpu, requested_workers=cfg.workers, max_cpu_utilization=cfg.max_cpu_utilization)
    if use_gpu and workers > 1:
        raise RuntimeError(
            "GPU OCR with workers>1 is disabled for safety (avoids duplicated model instances and unstable writes). "
            "Use workers=1 on GPU, or paddle-device=cpu with workers auto for parallel OCR."
        )
    ocr_cache: dict[str, Any] = {}

    total = len(docs)
    done = 0
    started = time.perf_counter()

    premis.add_event(
        "ocr",
        "started",
        {
            "total_docs": total,
            "engine": "paddle",
            "variant": cfg.variant,
            "lang": cfg.lang,
            "latin_lang": cfg.latin_lang,
            "auto_script": cfg.auto_script,
            "device_mode": cfg.device,
            "use_gpu": use_gpu,
            "angle_cls": cfg.angle_cls,
            "det_limit_side_len": cfg.det_limit_side_len,
            "det_db_thresh": cfg.det_db_thresh,
            "det_db_box_thresh": cfg.det_db_box_thresh,
            "det_db_unclip_ratio": cfg.det_db_unclip_ratio,
            "rec_score_thresh": cfg.rec_score_thresh,
            "drop_score": cfg.drop_score,
            "workers": workers,
            "max_cpu_utilization": cfg.max_cpu_utilization,
            "gpu_mem_fraction": _clamp_gpu_mem_fraction(cfg.gpu_mem_fraction),
            "worker_threads": cfg.worker_threads,
        },
        object_id="batch:ocr",
        agents=["agent:pipeline", "agent:paddleocr"],
    )

    ok = 0
    err = 0

    pending: list[dict[str, Any]] = []
    for row in docs:
        doc_id = str(row["doc_id"])
        image_path = Path(str(row["image_path"]))
        text_path = ocr_dir / f"{doc_id}.txt"

        # Resume retries previous OCR errors, skips only successful OCR entries.
        if resume and doc_id in index and index[doc_id].get("status") == "ok":
            done += 1
            logger.info(progress_line("OCR", done, total, started))
            continue

        if not image_path.exists():
            done += 1
            exc = RuntimeError(f"image_not_found: {image_path}")
            index[doc_id] = {
                "doc_id": doc_id,
                "status": "ocr_error",
                "split": row.get("split"),
                "rel_path": row.get("rel_path"),
                "image_path": str(image_path),
                "ocr_path": str(text_path),
                "ocr_engine": "paddle",
                "ocr_variant": cfg.variant,
                "updated_at": now_iso_utc(),
                "last_error": str(exc),
            }
            err += 1
            logger.warn(f"OCR failed for {doc_id}: {exc}")
            logger.info(progress_line("OCR", done, total, started))
            continue

        pending.append(row)

    def _on_success(row: dict[str, Any], text: str, quality: dict[str, Any], lang_used: str, script_used: str | None) -> None:
        nonlocal ok
        doc_id = str(row["doc_id"])
        image_path = Path(str(row["image_path"]))
        text_path = ocr_dir / f"{doc_id}.txt"
        text_path.write_text(text, encoding="utf-8")

        index[doc_id] = {
            "doc_id": doc_id,
            "status": "ok",
            "split": row.get("split"),
            "rel_path": row.get("rel_path"),
            "image_path": str(image_path),
            "ocr_path": str(text_path),
            "ocr_engine": "paddle",
            "ocr_variant": cfg.variant,
            "ocr_lang_used": lang_used,
            "ocr_script": script_used,
            "updated_at": now_iso_utc(),
            "quality": quality,
        }
        ok += 1
        premis.add_object(f"obj:ocr:{doc_id}", str(text_path), fmt="text/plain")
        premis.add_event(
            "ocr",
            "success",
            {
                "doc_id": doc_id,
                "chars": quality["chars"],
                "words": quality["words"],
                "alnum_ratio": quality["alnum_ratio"],
            },
            object_id=f"obj:ocr:{doc_id}",
            object_path=str(text_path),
            agents=["agent:paddleocr"],
        )

    def _on_failure(row: dict[str, Any], error: str) -> None:
        nonlocal err
        doc_id = str(row["doc_id"])
        image_path = Path(str(row["image_path"]))
        text_path = ocr_dir / f"{doc_id}.txt"
        index[doc_id] = {
            "doc_id": doc_id,
            "status": "ocr_error",
            "split": row.get("split"),
            "rel_path": row.get("rel_path"),
            "image_path": str(image_path),
            "ocr_path": str(text_path),
            "ocr_engine": "paddle",
            "ocr_variant": cfg.variant,
            "updated_at": now_iso_utc(),
            "last_error": error,
        }
        err += 1
        logger.warn(f"OCR failed for {doc_id}: {error}")
        premis.add_event(
            "ocr",
            "failure",
            {"doc_id": doc_id, "error": error},
            object_id=f"obj:image:{doc_id}",
            object_path=str(image_path),
            agents=["agent:paddleocr"],
        )

    if workers <= 1:
        for row in pending:
            doc_id = str(row["doc_id"])
            image_path = Path(str(row["image_path"]))
            lang_used, script_used = _resolve_lang_for_doc(image_path, cfg)
            if lang_used not in ocr_cache:
                try:
                    ocr_cache[lang_used] = _create_paddle_ocr(lang_used, cfg=cfg, use_gpu=use_gpu)
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to initialize PaddleOCR. "
                        "Likely API/dependency mismatch (e.g., paddleocr 3.x without paddlepaddle, or old params). "
                        f"lang={lang_used}, use_gpu={use_gpu}, error={exc}"
                    ) from exc
            try:
                paddle_input = _prepare_image_for_paddle(image_path, temp_dir=temp_img_dir)
                result = _run_paddle(ocr_cache[lang_used], paddle_input, angle_cls=cfg.angle_cls)
                text = _extract_text_from_paddle_result(result)
                if not text:
                    raise RuntimeError("empty_ocr_text")
                quality = _calc_ocr_quality(text)
                _on_success(row, text, quality, lang_used=lang_used, script_used=script_used)
            except Exception as exc:
                _on_failure(row, str(exc))

            done += 1
            if done % 20 == 0 or done == total:
                _save_index(index_path, index)
            logger.info(progress_line("OCR", done, total, started))
    else:
        payload_cfg = _cfg_to_worker_payload(cfg)
        tasks: dict[Any, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for row in pending:
                fut = ex.submit(
                    _worker_process_doc,
                    {
                        "row": row,
                        "cfg": payload_cfg,
                        "use_gpu": use_gpu,
                        "temp_dir": str(temp_img_dir),
                    },
                )
                tasks[fut] = row

            for fut in as_completed(tasks):
                row = tasks[fut]
                try:
                    res = fut.result()
                    _on_success(
                        row=row,
                        text=str(res["text"]),
                        quality=dict(res["quality"]),
                        lang_used=str(res["ocr_lang_used"]),
                        script_used=res.get("ocr_script"),
                    )
                except Exception as exc:
                    _on_failure(row, str(exc))
                done += 1
                if done % 20 == 0 or done == total:
                    _save_index(index_path, index)
                logger.info(progress_line("OCR", done, total, started))

    _save_index(index_path, index)

    summary = {
        "generated_at": now_iso_utc(),
        "total_docs": total,
        "ok": ok,
        "ocr_error": err,
        "index_file": str(index_path),
        "ocr_dir": str(ocr_dir),
    }

    premis.add_object("obj:ocr_index", str(index_path), fmt="application/json")
    premis.add_event(
        "ocr",
        "success",
        summary,
        object_id="obj:ocr_index",
        object_path=str(index_path),
        agents=["agent:pipeline", "agent:paddleocr"],
    )

    return summary
