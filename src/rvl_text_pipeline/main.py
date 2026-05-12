from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


DEFAULT_CLASS_MAP = {
    0: "letter",
    1: "form",
    2: "email",
    3: "handwritten",
    4: "advertisement",
    5: "scientific report",
    6: "scientific publication",
    7: "specification",
    8: "file folder",
    9: "news article",
    10: "budget",
    11: "invoice",
    12: "presentation",
    13: "questionnaire",
    14: "resume",
    15: "memo",
}


class ValidationError(Exception):
    """Prediction payload validation error."""


def now_iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


class PipelineLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, msg: str) -> None:
        stamp = now_iso_utc()
        line = f"[{stamp}] {level.upper():5s} {msg}"
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, msg: str) -> None:
        self._write("info", msg)

    def warn(self, msg: str) -> None:
        self._write("warn", msg)

    def error(self, msg: str) -> None:
        self._write("error", msg)


class PremisRecorder:
    def __init__(self, premis_dir: Path, run_id: str, logger: PipelineLogger):
        self.premis_dir = premis_dir
        self.run_id = run_id
        self.logger = logger
        self.events_file = premis_dir / "events.jsonl"
        self.objects_file = premis_dir / "objects.jsonl"
        self.agents_file = premis_dir / "agents.jsonl"
        self.rights_file = premis_dir / "rights.jsonl"
        premis_dir.mkdir(parents=True, exist_ok=True)
        self.event_counter = self._infer_event_counter()

    def _infer_event_counter(self) -> int:
        if not self.events_file.exists():
            return 0
        c = 0
        with self.events_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    c += 1
        return c

    def ensure_defaults(self, model_name: str) -> None:
        if not self.agents_file.exists() or self.agents_file.stat().st_size == 0:
            self.add_agent("agent:pipeline", "rvl_text_pipeline", "software")
            self.add_agent("agent:tesseract", "tesseract-ocr", "software")
            self.add_agent("agent:paddleocr", "paddleocr", "software")
            self.add_agent("agent:ollama", model_name, "software")

        if not self.rights_file.exists() or self.rights_file.stat().st_size == 0:
            append_jsonl(
                self.rights_file,
                {
                    "rightsStatementIdentifier": {
                        "rightsStatementIdentifierType": "local",
                        "rightsStatementIdentifierValue": f"rights:{self.run_id}",
                    },
                    "rightsBasis": "copyright",
                    "rightsGranted": [
                        {
                            "act": "analyze",
                            "restriction": "evaluation-only",
                            "termOfGrant": "run-scope",
                        }
                    ],
                    "createdAt": now_iso_utc(),
                },
            )

    def add_agent(self, agent_id: str, name: str, agent_type: str) -> None:
        append_jsonl(
            self.agents_file,
            {
                "agentIdentifier": {
                    "agentIdentifierType": "local",
                    "agentIdentifierValue": agent_id,
                },
                "agentName": name,
                "agentType": agent_type,
                "createdAt": now_iso_utc(),
            },
        )

    def add_object(self, object_id: str, path: str, category: str = "file", fmt: str = "unknown") -> None:
        append_jsonl(
            self.objects_file,
            {
                "objectIdentifier": {
                    "objectIdentifierType": "local",
                    "objectIdentifierValue": object_id,
                },
                "objectCategory": category,
                "objectPath": path,
                "objectCharacteristics": {
                    "format": fmt,
                },
                "createdAt": now_iso_utc(),
            },
        )

    def add_event(
        self,
        event_type: str,
        outcome: str,
        detail: dict[str, Any],
        object_id: str | None = None,
        object_path: str | None = None,
        agents: list[str] | None = None,
    ) -> None:
        self.event_counter += 1
        evt_id = f"evt-{self.event_counter:06d}"
        payload: dict[str, Any] = {
            "eventIdentifier": {
                "eventIdentifierType": "local",
                "eventIdentifierValue": evt_id,
            },
            "eventType": event_type,
            "eventDateTime": now_iso_utc(),
            "eventOutcomeInformation": {
                "eventOutcome": outcome,
                "eventOutcomeDetail": detail,
            },
        }
        if object_id or object_path:
            payload["linkingObjectIdentifier"] = [
                {
                    "linkingObjectIdentifierType": "local",
                    "linkingObjectIdentifierValue": object_id or object_path or "",
                }
            ]
        if agents:
            payload["linkingAgentIdentifier"] = [
                {
                    "linkingAgentIdentifierType": "local",
                    "linkingAgentIdentifierValue": a,
                }
                for a in agents
            ]
        append_jsonl(self.events_file, payload)


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def progress_line(stage: str, done: int, total: int, started_at: float) -> str:
    elapsed = max(0.001, time.perf_counter() - started_at)
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else 0.0
    return (
        f"{stage}: {done}/{total} | rate={rate:.2f} docs/s "
        f"| elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVL-CDIP textual OCR->LLM evaluation pipeline")
    p.add_argument("--labels-file", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--sample-manifest", type=Path, default=None)
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-root", type=Path, default=Path("data/processed/saida-ocr-class"))
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--request-timeout", type=int, default=180)
    p.add_argument("--max-ocr-chars", type=int, default=12000)
    p.add_argument("--num-predict", type=int, default=512)
    p.add_argument("--fallback-num-predict", type=int, default=64)
    p.add_argument("--invalid-json-fallback-retry", type=int, default=1)
    p.add_argument("--inference-schema", choices=["full_json", "class_id_only"], default="full_json")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--ambiguous-votes", type=int, default=1)
    p.add_argument("--vote-temperature", type=float, default=0.2)
    p.add_argument("--vote-top-p", type=float, default=0.95)
    p.add_argument("--vote-top-k", type=int, default=50)
    p.add_argument("--ambiguity-min-chars", type=int, default=350)
    p.add_argument("--ambiguity-min-unique-words", type=int, default=40)
    p.add_argument("--ambiguity-max-digit-ratio", type=float, default=0.35)
    p.add_argument("--ocr-engine", choices=["paddle", "tesseract"], default="paddle")
    p.add_argument("--ocr-lang", default="eng")
    p.add_argument("--ocr-auto-script", action="store_true")
    p.add_argument("--ocr-latin-langs", default="eng+deu+spa+por+fra+ita+nl")
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
    p.add_argument("--disable-fewshot", action="store_true")
    p.add_argument("--fewshot-labels-file", type=Path, default=Path("data/rvl-cdip/labels/train.txt"))
    p.add_argument("--fewshot-seed", type=int, default=7)
    p.add_argument("--fewshot-max-chars", type=int, default=700)
    p.add_argument("--class-map-file", type=Path, default=Path("configs/rvl_class_map.json"))
    p.add_argument("--system-prompt-file", type=Path, default=Path("configs/rvl_system_prompt.txt"))
    p.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434/api/generate")
    return p.parse_args()


def load_class_map(path: Path) -> dict[int, str]:
    if not path.exists():
        return dict(DEFAULT_CLASS_MAP)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def build_name_to_id(class_map: dict[int, str]) -> dict[str, int]:
    name_to_id: dict[str, int] = {}
    for cid, name in class_map.items():
        name_to_id[name.strip().lower()] = cid
    return name_to_id


def load_system_prompt(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "Classify the document into one of the 16 RVL-CDIP classes and return JSON."


def system_prompt_has_static_fewshot(system_prompt: str) -> bool:
    markers = ("FEWSHOT_SOURCE=system", "STATIC_FEWSHOT=1")
    return any(marker in system_prompt for marker in markers)


def detect_script_from_osd(osd_text: str) -> str | None:
    match = re.search(r"Script:\s*([A-Za-z]+)", osd_text)
    if not match:
        return None
    return match.group(1)


def resolve_ocr_lang(
    img: Any,
    ocr_lang: str,
    ocr_auto_script: bool,
    ocr_latin_langs: str,
    pytesseract_mod: Any,
) -> tuple[str, str | None]:
    if not ocr_auto_script:
        return ocr_lang, None
    try:
        osd_text = pytesseract_mod.image_to_osd(img)
    except Exception:
        return ocr_lang, None
    script = detect_script_from_osd(osd_text)
    if not script:
        return ocr_lang, None
    if script.lower() == "latin":
        return ocr_latin_langs, script
    return ocr_lang, script


def resolve_paddle_use_gpu(device_mode: str) -> bool:
    if device_mode == "gpu":
        return True
    if device_mode == "cpu":
        return False
    try:
        import paddle
    except Exception:
        return False
    try:
        if not paddle.is_compiled_with_cuda():
            return False
        return paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def resolve_paddle_lang(
    image_path: Path,
    paddle_lang: str,
    paddle_auto_script: bool,
    paddle_latin_lang: str,
) -> tuple[str, str | None]:
    if not paddle_auto_script:
        return paddle_lang, None
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return paddle_lang, None
    try:
        with Image.open(image_path) as img:
            osd_text = pytesseract.image_to_osd(img)
    except Exception:
        return paddle_lang, None
    script = detect_script_from_osd(osd_text)
    if not script:
        return paddle_lang, None
    script_l = script.lower()
    if script_l == "latin":
        return paddle_latin_lang, script
    # Keep default language for non-Latin scripts unless caller chooses another model.
    return paddle_lang, script


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


def run_paddle_ocr(ocr_engine: Any, image_path: Path, angle_cls: bool) -> Any:
    """
    PaddleOCR API changed between versions:
    - Older: .ocr(path, cls=...)
    - Newer: .predict(path) without cls
    This wrapper keeps compatibility across both.
    """
    img = str(image_path)

    if hasattr(ocr_engine, "predict"):
        try:
            return ocr_engine.predict(img, use_textline_orientation=angle_cls)
        except TypeError:
            return ocr_engine.predict(img)

    if hasattr(ocr_engine, "ocr"):
        try:
            return ocr_engine.ocr(img, cls=angle_cls)
        except TypeError:
            try:
                return ocr_engine.ocr(img)
            except TypeError:
                pass

    raise RuntimeError("PaddleOCR engine has neither compatible .ocr nor .predict method")


def prepare_image_for_paddle(image_path: Path, temp_dir: Path) -> Path:
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
        raise RuntimeError("Pillow is required to convert TIFF for PaddleOCR") from exc

    try:
        with Image.open(image_path) as img:
            if img.mode not in {"RGB", "L"}:
                img = img.convert("RGB")
            img.save(converted, format="PNG")
    except Exception as exc:
        raise RuntimeError(f"tiff_to_png_failed: {exc}") from exc

    return converted


def create_paddle_ocr(
    lang: str,
    use_gpu: bool,
    angle_cls: bool,
    det_limit_side_len: int,
    det_db_thresh: float,
    det_db_box_thresh: float,
    det_db_unclip_ratio: float,
    rec_score_thresh: float,
    drop_score: float,
) -> Any:
    from paddleocr import PaddleOCR
    import inspect

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    def instantiate_with_pruning(kwargs: dict[str, Any]) -> Any:
        local = dict(kwargs)
        last_exc: Exception | None = None
        for _ in range(12):
            try:
                return PaddleOCR(**local)
            except (TypeError, ValueError) as exc:
                last_exc = exc
                msg = str(exc)
                m = re.search(r"Unknown argument:\s*([A-Za-z_][A-Za-z0-9_]*)", msg)
                if not m:
                    raise
                bad = m.group(1)
                if bad not in local:
                    raise
                local.pop(bad, None)
        raise last_exc or RuntimeError("unable_to_create_paddleocr")

    params = set(inspect.signature(PaddleOCR.__init__).parameters.keys())
    is_modern = "text_det_limit_side_len" in params or "use_doc_orientation_classify" in params

    modern = {
        "lang": lang,
        "device": "gpu:0" if use_gpu else "cpu",
        "use_textline_orientation": angle_cls,
        "text_det_limit_side_len": det_limit_side_len,
        "text_det_thresh": det_db_thresh,
        "text_det_box_thresh": det_db_box_thresh,
        "text_det_unclip_ratio": det_db_unclip_ratio,
        "text_rec_score_thresh": rec_score_thresh,
        "drop_score": drop_score,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
    }

    legacy_full = {
        "lang": lang,
        "use_gpu": use_gpu,
        "use_angle_cls": angle_cls,
        "det_limit_side_len": det_limit_side_len,
        "det_db_thresh": det_db_thresh,
        "det_db_box_thresh": det_db_box_thresh,
        "det_db_unclip_ratio": det_db_unclip_ratio,
        "rec_score_thresh": rec_score_thresh,
        "drop_score": drop_score,
    }
    legacy_minimal = {
        "lang": lang,
        "use_gpu": use_gpu,
        "use_angle_cls": angle_cls,
    }

    candidates = (modern, legacy_full, legacy_minimal) if is_modern else (legacy_full, legacy_minimal, modern)
    for candidate in candidates:
        try:
            return instantiate_with_pruning(candidate)
        except Exception:
            continue
    raise RuntimeError("Failed to initialize PaddleOCR with compatible arguments")


def load_labels(labels_file: Path, class_map: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with labels_file.open("r", encoding="utf-8", errors="ignore") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid labels line at {labels_file}:{i}: {line}")
            rel_path, class_id_s = parts
            class_id = int(class_id_s)
            if class_id not in class_map:
                raise ValueError(f"Class id out of range at {labels_file}:{i}: {class_id}")
            rows.append(
                {
                    "rel_path": rel_path,
                    "class_id": class_id,
                    "class_name": class_map[class_id],
                }
            )
    return rows


def make_doc_id(rel_path: str) -> str:
    stem = Path(rel_path).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    stem = stem[:32] or "doc"
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{h}"


def stratified_sample(records: list[dict[str, Any]], sample_size: int, seed: int, num_classes: int = 16) -> list[dict[str, Any]]:
    if sample_size < num_classes:
        raise ValueError(f"sample_size must be >= {num_classes} to cover all classes")
    if sample_size > len(records):
        raise ValueError(f"sample_size ({sample_size}) > total records ({len(records)})")

    buckets: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_classes)}
    for r in records:
        buckets[r["class_id"]].append(r)

    missing = [cid for cid in range(num_classes) if not buckets[cid]]
    if missing:
        raise ValueError(f"Missing classes in labels for sampling: {missing}")

    rng = __import__("random").Random(seed)

    selected: list[dict[str, Any]] = []
    remaining_pool: dict[int, list[dict[str, Any]]] = {}
    for cid in range(num_classes):
        group = sorted(buckets[cid], key=lambda x: x["rel_path"])
        rng.shuffle(group)
        selected.append(group[0])
        remaining_pool[cid] = group[1:]

    remain_slots = sample_size - num_classes
    if remain_slots > 0:
        total_remaining = sum(len(v) for v in remaining_pool.values())
        if remain_slots > total_remaining:
            raise ValueError("Not enough remaining records to fill sample")

        quotas: dict[int, int] = {cid: 0 for cid in range(num_classes)}
        fracs: list[tuple[float, int]] = []
        allocated = 0

        for cid in range(num_classes):
            cap = len(remaining_pool[cid])
            ideal = (remain_slots * cap / total_remaining) if total_remaining else 0.0
            base = min(cap, int(math.floor(ideal)))
            quotas[cid] = base
            allocated += base
            fracs.append((ideal - base, cid))

        leftover = remain_slots - allocated
        for _, cid in sorted(fracs, key=lambda x: (-x[0], x[1])):
            if leftover <= 0:
                break
            if quotas[cid] < len(remaining_pool[cid]):
                quotas[cid] += 1
                leftover -= 1

        if leftover > 0:
            for cid in range(num_classes):
                while leftover > 0 and quotas[cid] < len(remaining_pool[cid]):
                    quotas[cid] += 1
                    leftover -= 1

        for cid in range(num_classes):
            n = quotas[cid]
            if n:
                selected.extend(remaining_pool[cid][:n])

    rng.shuffle(selected)
    return selected


def ensure_run_dirs(run_dir: Path) -> dict[str, Path]:
    paths = {
        "run": run_dir,
        "manifests": run_dir / "manifests",
        "ocr": run_dir / "ocr",
        "predictions": run_dir / "predictions",
        "evaluation": run_dir / "evaluation",
        "logs": run_dir / "logs",
        "premis": run_dir / "premis",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def build_fewshot_block(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return ""
    parts: list[str] = []
    parts.append("Reference labeled examples (one real OCR example per class):")
    for ex in sorted(examples, key=lambda x: int(x["class_id"])):
        parts.append(
            f"\n[Example class_id={ex['class_id']} class_name={ex['class_name']}]\n"
            "ocr_excerpt:\n\"\"\"\n"
            f"{ex['ocr_excerpt']}\n"
            "\"\"\""
        )
    parts.append(
        "\nUse these examples as semantic guidance. "
        "Do not copy labels blindly; classify only the target document."
    )
    return "\n".join(parts).strip()


def build_user_prompt_class_id_only(doc_id: str, ocr_text: str, fewshot_block: str = "") -> str:
    fewshot_text = f"{fewshot_block}\n\n" if fewshot_block else ""
    return (
        "Classify the target document into RVL-CDIP classes (0..15).\\n\\n"
        f"{fewshot_text}"
        f"doc_id: {doc_id}\\n\\n"
        "ocr_text:\\n\"\"\"\\n"
        f"{ocr_text}\\n"
        "\"\"\"\\n\\n"
        "Return ONLY valid JSON with EXACTLY one key:\\n"
        "{\"class_id\": <integer from 0 to 15>}\\n"
        "No extra keys, no markdown, no explanation."
    )


def build_user_prompt(
    doc_id: str,
    ocr_text: str,
    fewshot_block: str = "",
    inference_schema: str = "full_json",
) -> str:
    if inference_schema == "class_id_only":
        return build_user_prompt_class_id_only(doc_id=doc_id, ocr_text=ocr_text, fewshot_block=fewshot_block)
    fewshot_text = f"{fewshot_block}\n\n" if fewshot_block else ""
    return (
        "Classify the target document below.\\n\\n"
        f"{fewshot_text}"
        f"doc_id: {doc_id}\\n\\n"
        "ocr_text:\\n\"\"\"\\n"
        f"{ocr_text}\\n"
        "\"\"\"\\n\\n"
        "Return ONLY valid JSON and strictly follow the required schema."
    )


def call_ollama(
    endpoint: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    num_predict: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_predict": num_predict,
        },
    }
    req = urlrequest.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urlerror.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    parsed = json.loads(body)
    if parsed.get("error"):
        raise RuntimeError(f"Ollama error: {parsed['error']}")
    text = parsed.get("response", "")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("Ollama returned empty response")
    return text


def recover_payload_from_partial_json(text: str) -> dict[str, Any] | None:
    class_id_m = re.search(r'"class_id"\s*:\s*(-?\d+)', text)
    if not class_id_m:
        return None

    out: dict[str, Any] = {"class_id": int(class_id_m.group(1))}

    class_name_m = re.search(r'"class_name"\s*:\s*"([^"\n\r]+)"', text)
    if class_name_m:
        out["class_name"] = class_name_m.group(1).strip()

    conf_m = re.search(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if conf_m:
        try:
            out["confidence"] = float(conf_m.group(1))
        except Exception:
            pass

    return out


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        recovered = recover_payload_from_partial_json(text)
        if recovered is not None:
            return recovered
        raise ValidationError("No JSON object in model response")

    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        recovered = recover_payload_from_partial_json(text)
        if recovered is not None:
            return recovered
        raise ValidationError(f"Invalid JSON from model: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValidationError("Model JSON is not an object")
    return obj


def clamp01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def normalize_class(class_id: Any, class_name: Any, class_map: dict[int, str], name_to_id: dict[str, int]) -> tuple[int, str]:
    cid: int | None = None
    cname: str | None = None

    try:
        if class_id is not None:
            cid_tmp = int(class_id)
            if cid_tmp in class_map:
                cid = cid_tmp
    except Exception:
        cid = None

    if isinstance(class_name, str):
        cname_clean = class_name.strip().lower()
        if cname_clean in name_to_id:
            cname = class_map[name_to_id[cname_clean]]

    if cid is None and cname is None:
        raise ValidationError("Cannot resolve class from class_id/class_name")

    if cid is None and cname is not None:
        cid = name_to_id[cname.lower()]

    if cid is not None:
        canonical_name = class_map[cid]
        return cid, canonical_name

    raise ValidationError("Unexpected class normalization path")


def normalize_top3(top3_raw: Any, pred_id: int, class_map: dict[int, str], name_to_id: dict[str, int]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []

    if isinstance(top3_raw, list):
        for item in top3_raw:
            if not isinstance(item, dict):
                continue
            try:
                cid, cname = normalize_class(item.get("class_id"), item.get("class_name"), class_map, name_to_id)
            except ValidationError:
                continue
            parsed.append(
                {
                    "class_id": cid,
                    "class_name": cname,
                    "confidence": clamp01(item.get("confidence", 0.0)),
                }
            )

    seen = set()
    out: list[dict[str, Any]] = []

    out.append(
        {
            "class_id": pred_id,
            "class_name": class_map[pred_id],
            "confidence": parsed[0]["confidence"] if parsed and parsed[0]["class_id"] == pred_id else 1.0,
        }
    )
    seen.add(pred_id)

    for item in parsed:
        if item["class_id"] in seen:
            continue
        out.append(item)
        seen.add(item["class_id"])
        if len(out) == 3:
            break

    if len(out) < 3:
        for cid in sorted(class_map.keys()):
            if cid in seen:
                continue
            out.append({"class_id": cid, "class_name": class_map[cid], "confidence": 0.0})
            seen.add(cid)
            if len(out) == 3:
                break

    return out[:3]


def normalize_prediction_payload(
    payload: dict[str, Any],
    doc_id: str,
    class_map: dict[int, str],
    name_to_id: dict[str, int],
    inference_schema: str,
) -> dict[str, Any]:
    if inference_schema == "class_id_only":
        if "class_id" not in payload:
            raise ValidationError("class_id is required in class_id_only schema")
        class_id, class_name = normalize_class(payload.get("class_id"), None, class_map, name_to_id)
        top3 = normalize_top3([], class_id, class_map, name_to_id)
        return {
            "doc_id": doc_id,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": 1.0,
            "rationale": "",
            "top3": top3,
            "evidence": [],
        }

    class_id, class_name = normalize_class(payload.get("class_id"), payload.get("class_name"), class_map, name_to_id)

    confidence = clamp01(payload.get("confidence", 0.0))

    rationale = payload.get("rationale")
    if not isinstance(rationale, str):
        rationale = ""
    rationale = rationale.strip()

    evidence_raw = payload.get("evidence", [])
    evidence: list[str] = []
    if isinstance(evidence_raw, list):
        for v in evidence_raw:
            if isinstance(v, str) and v.strip():
                evidence.append(v.strip())
    evidence = evidence[:5]

    top3 = normalize_top3(payload.get("top3"), class_id, class_map, name_to_id)

    return {
        "doc_id": doc_id,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "rationale": rationale,
        "top3": top3,
        "evidence": evidence,
    }


def build_json_repair_prompt(
    doc_id: str,
    ocr_text: str,
    inference_schema: str,
    previous_response: str,
) -> str:
    snippet = previous_response.strip()
    if len(snippet) > 600:
        snippet = snippet[:600]
    if inference_schema == "class_id_only":
        return (
            "Repair the previous invalid output.\\n\\n"
            f"doc_id: {doc_id}\\n"
            "Return ONLY valid JSON with EXACTLY one key:\\n"
            "{\"class_id\": <integer from 0 to 15>}\\n\\n"
            f"Invalid output was:\\n{snippet}\\n\\n"
            "OCR text:\\n\"\"\"\\n"
            f"{ocr_text}\\n"
            "\"\"\""
        )
    return (
        "Repair the previous invalid output and return valid JSON matching the required schema.\\n\\n"
        f"doc_id: {doc_id}\\n"
        f"Invalid output was:\\n{snippet}\\n\\n"
        "OCR text:\\n\"\"\"\\n"
        f"{ocr_text}\\n"
        "\"\"\""
    )


def is_ambiguous_ocr_text(
    text: str,
    min_chars: int,
    min_unique_words: int,
    max_digit_ratio: float,
) -> bool:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) < max(1, min_chars):
        return True

    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    denom = letters + digits
    if denom > 0 and (digits / denom) > max_digit_ratio:
        return True

    words = {w.lower() for w in re.findall(r"[A-Za-z]{2,}", compact)}
    if len(words) < max(1, min_unique_words):
        return True

    return False


def majority_vote_predictions(preds: list[dict[str, Any]], class_map: dict[int, str]) -> dict[str, Any]:
    if not preds:
        raise ValidationError("majority_vote_predictions received empty list")
    if len(preds) == 1:
        return preds[0]

    counts: dict[int, int] = {}
    first_idx: dict[int, int] = {}
    for i, pred in enumerate(preds):
        cid = int(pred["class_id"])
        counts[cid] = counts.get(cid, 0) + 1
        if cid not in first_idx:
            first_idx[cid] = i

    winner = sorted(counts.keys(), key=lambda cid: (-counts[cid], first_idx[cid]))[0]
    winner_pred = next(p for p in preds if int(p["class_id"]) == winner)
    voted = dict(winner_pred)
    voted["class_id"] = winner
    voted["class_name"] = class_map[winner]
    voted["confidence"] = counts[winner] / len(preds)
    voted["vote_total"] = len(preds)
    voted["vote_winner_count"] = counts[winner]
    return voted


def load_or_create_sample_manifest(
    records: list[dict[str, Any]],
    manifests_dir: Path,
    sample_size: int,
    seed: int,
    images_root: Path,
    class_map: dict[int, str],
    resume: bool,
    sample_manifest_source: Path | None = None,
) -> list[dict[str, Any]]:
    sample_file = manifests_dir / "sample.jsonl"
    summary_file = manifests_dir / "sample_summary.json"

    if resume and sample_file.exists():
        sample = read_jsonl(sample_file)
        if not sample:
            raise RuntimeError("Resume requested but sample manifest is empty")
        return sample

    if sample_manifest_source:
        if not sample_manifest_source.exists():
            raise RuntimeError(f"Sample manifest source not found: {sample_manifest_source}")
        source_rows = read_jsonl(sample_manifest_source)
        if not source_rows:
            raise RuntimeError(f"Sample manifest source is empty: {sample_manifest_source}")
        sample: list[dict[str, Any]] = []
        for row in source_rows:
            rel_path = str(row.get("rel_path", "")).strip()
            if not rel_path:
                raise RuntimeError(f"Invalid row in sample manifest source: missing rel_path ({row})")
            class_id = int(row.get("class_id"))
            if class_id not in class_map:
                raise RuntimeError(f"Invalid class_id in sample manifest source: {class_id}")
            sample.append(
                {
                    "rel_path": rel_path,
                    "class_id": class_id,
                    "class_name": class_map[class_id],
                }
            )
    else:
        sample = stratified_sample(records, sample_size=sample_size, seed=seed)

    manifest_rows: list[dict[str, Any]] = []
    class_counts: dict[int, int] = {i: 0 for i in range(16)}

    for item in sample:
        doc_id = make_doc_id(item["rel_path"])
        class_counts[item["class_id"]] += 1
        manifest_rows.append(
            {
                "doc_id": doc_id,
                "rel_path": item["rel_path"],
                "image_path": str(images_root / item["rel_path"]),
                "class_id": item["class_id"],
                "class_name": item["class_name"],
            }
        )

    with sample_file.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    atomic_write_json(
        summary_file,
        {
            "created_at": now_iso_utc(),
            "sample_size": len(manifest_rows),
            "seed": seed,
            "class_counts": class_counts,
            "sample_manifest_source": str(sample_manifest_source) if sample_manifest_source else None,
        },
    )

    return manifest_rows


def load_or_create_fewshot_examples(
    manifests_dir: Path,
    fewshot_labels_file: Path,
    images_root: Path,
    class_map: dict[int, str],
    exclude_rel_paths: set[str],
    ocr_engine: str,
    ocr_lang: str,
    ocr_auto_script: bool,
    ocr_latin_langs: str,
    paddle_lang: str,
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
    seed: int,
    max_chars: int,
    resume: bool,
    logger: PipelineLogger,
    premis: PremisRecorder,
) -> list[dict[str, Any]]:
    fewshot_file = manifests_dir / "fewshot_examples.jsonl"
    fewshot_summary = manifests_dir / "fewshot_summary.json"

    if resume and fewshot_file.exists():
        rows = read_jsonl(fewshot_file)
        if rows:
            return rows

    if not fewshot_labels_file.exists():
        logger.warn(f"Few-shot labels file not found, skipping few-shot: {fewshot_labels_file}")
        return []

    use_paddle = ocr_engine == "paddle"
    use_tesseract = ocr_engine == "tesseract"
    if not use_paddle and not use_tesseract:
        logger.warn(f"Few-shot skipped: unsupported OCR engine {ocr_engine}")
        return []

    pytesseract_mod = None
    image_mod = None
    paddle_use_gpu = False
    paddle_cache: dict[str, Any] = {}

    if use_tesseract:
        try:
            from PIL import Image
            import pytesseract
        except Exception:
            logger.warn("Few-shot skipped: Tesseract dependencies unavailable")
            return []
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            logger.warn("Few-shot skipped: tesseract binary not found in PATH")
            return []
        image_mod = Image
        pytesseract_mod = pytesseract
    else:
        try:
            _ = __import__("paddleocr")
        except Exception:
            logger.warn("Few-shot skipped: PaddleOCR dependencies unavailable")
            return []
        paddle_use_gpu = resolve_paddle_use_gpu(paddle_device)

    records = load_labels(fewshot_labels_file, class_map)
    buckets: dict[int, list[dict[str, Any]]] = {i: [] for i in range(16)}
    for rec in records:
        if rec["rel_path"] in exclude_rel_paths:
            continue
        buckets[rec["class_id"]].append(rec)

    rng = __import__("random").Random(seed)
    picked: list[dict[str, Any]] = []
    missing: list[int] = []

    for cid in range(16):
        candidates = sorted(buckets[cid], key=lambda x: x["rel_path"])
        rng.shuffle(candidates)
        selected_row: dict[str, Any] | None = None

        for row in candidates:
            image_path = images_root / row["rel_path"]
            if not image_path.exists():
                continue
            try:
                if use_tesseract:
                    assert image_mod is not None and pytesseract_mod is not None
                    with image_mod.open(image_path) as img:
                        lang_used, _ = resolve_ocr_lang(
                            img, ocr_lang, ocr_auto_script, ocr_latin_langs, pytesseract_mod
                        )
                        text = pytesseract_mod.image_to_string(img, lang=lang_used)
                else:
                    lang_used, _ = resolve_paddle_lang(
                        image_path=image_path,
                        paddle_lang=paddle_lang,
                        paddle_auto_script=paddle_auto_script,
                        paddle_latin_lang=paddle_latin_lang,
                    )
                    if lang_used not in paddle_cache:
                        paddle_cache[lang_used] = create_paddle_ocr(
                            lang=lang_used,
                            use_gpu=paddle_use_gpu,
                            angle_cls=paddle_angle_cls,
                            det_limit_side_len=paddle_det_limit_side_len,
                            det_db_thresh=paddle_det_db_thresh,
                            det_db_box_thresh=paddle_det_db_box_thresh,
                            det_db_unclip_ratio=paddle_det_db_unclip_ratio,
                            rec_score_thresh=paddle_rec_score_thresh,
                            drop_score=paddle_drop_score,
                        )
                    result = paddle_cache[lang_used].ocr(str(image_path), cls=paddle_angle_cls)
                    text = extract_text_from_paddle_result(result)
            except Exception:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            excerpt = re.sub(r"\s+", " ", text).strip()
            if max_chars > 0 and len(excerpt) > max_chars:
                excerpt = excerpt[:max_chars]
            selected_row = {
                "class_id": cid,
                "class_name": class_map[cid],
                "rel_path": row["rel_path"],
                "image_path": str(image_path),
                "ocr_excerpt": excerpt,
                "chars": len(excerpt),
            }
            break

        if selected_row is None:
            missing.append(cid)
            continue
        picked.append(selected_row)

    with fewshot_file.open("w", encoding="utf-8") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    atomic_write_json(
        fewshot_summary,
        {
            "created_at": now_iso_utc(),
            "fewshot_labels_file": str(fewshot_labels_file),
            "examples_count": len(picked),
            "missing_class_ids": missing,
            "seed": seed,
            "max_chars": max_chars,
        },
    )

    ocr_agent = "agent:paddleocr" if use_paddle else "agent:tesseract"
    premis.add_object("obj:fewshot", str(fewshot_file), fmt="application/jsonlines")
    premis.add_event(
        "sampling",
        "success",
        {
            "type": "fewshot",
            "labels_file": str(fewshot_labels_file),
            "examples_count": len(picked),
            "missing_class_ids": missing,
        },
        object_id="obj:fewshot",
        object_path=str(fewshot_file),
        agents=["agent:pipeline", ocr_agent],
    )

    if missing:
        logger.warn(f"Few-shot missing classes (no usable OCR example found): {missing}")
    logger.info(f"Few-shot examples prepared: {len(picked)}")
    return picked


def load_or_init_state(
    manifests_dir: Path,
    sample_rows: list[dict[str, Any]],
    run_config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    state_file = manifests_dir / "state.json"

    if resume and state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        return state

    docs: dict[str, Any] = {}
    for row in sample_rows:
        docs[row["doc_id"]] = {
            "doc_id": row["doc_id"],
            "rel_path": row["rel_path"],
            "image_path": row["image_path"],
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "status": "pending",
            "ocr_path": None,
            "prediction_path": None,
            "attempts": 0,
            "last_error": None,
            "updated_at": now_iso_utc(),
        }

    state = {
        "created_at": now_iso_utc(),
        "updated_at": now_iso_utc(),
        "config": run_config,
        "docs": docs,
    }
    atomic_write_json(state_file, state)
    return state


def save_state(manifests_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso_utc()
    atomic_write_json(manifests_dir / "state.json", state)


def run_ocr_stage(
    state: dict[str, Any],
    run_paths: dict[str, Path],
    logger: PipelineLogger,
    premis: PremisRecorder,
    ocr_engine: str,
    ocr_lang: str,
    ocr_auto_script: bool,
    ocr_latin_langs: str,
    paddle_lang: str,
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
    if ocr_engine == "tesseract":
        _run_ocr_stage_tesseract(
            state=state,
            run_paths=run_paths,
            logger=logger,
            premis=premis,
            ocr_lang=ocr_lang,
            ocr_auto_script=ocr_auto_script,
            ocr_latin_langs=ocr_latin_langs,
        )
        return
    if ocr_engine == "paddle":
        _run_ocr_stage_paddle(
            state=state,
            run_paths=run_paths,
            logger=logger,
            premis=premis,
            paddle_lang=paddle_lang,
            paddle_latin_lang=paddle_latin_lang,
            paddle_auto_script=paddle_auto_script,
            paddle_device=paddle_device,
            paddle_angle_cls=paddle_angle_cls,
            paddle_det_limit_side_len=paddle_det_limit_side_len,
            paddle_det_db_thresh=paddle_det_db_thresh,
            paddle_det_db_box_thresh=paddle_det_db_box_thresh,
            paddle_det_db_unclip_ratio=paddle_det_db_unclip_ratio,
            paddle_rec_score_thresh=paddle_rec_score_thresh,
            paddle_drop_score=paddle_drop_score,
        )
        return
    raise RuntimeError(f"Unsupported OCR engine: {ocr_engine}")


def _run_ocr_stage_tesseract(
    state: dict[str, Any],
    run_paths: dict[str, Path],
    logger: PipelineLogger,
    premis: PremisRecorder,
    ocr_lang: str,
    ocr_auto_script: bool,
    ocr_latin_langs: str,
) -> None:
    pending = [d for d in state["docs"].values() if d["status"] == "pending"]
    if not pending:
        logger.info("OCR stage skipped: no pending documents")
        return

    try:
        from PIL import Image
        import pytesseract
    except Exception as exc:
        raise RuntimeError(
            "Missing OCR dependencies. Install: pip install pillow pytesseract and system package tesseract-ocr"
        ) from exc

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "tesseract binary not found in PATH. Install system package tesseract-ocr"
        ) from exc

    total = len(pending)
    started = time.perf_counter()

    premis.add_event(
        "ocr",
        "started",
        {"total_docs": total, "lang": ocr_lang, "auto_script": ocr_auto_script, "latin_langs": ocr_latin_langs},
        object_id="batch:ocr",
        agents=["agent:pipeline", "agent:tesseract"],
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

            with Image.open(image_path) as img:
                lang_used, script_used = resolve_ocr_lang(
                    img, ocr_lang, ocr_auto_script, ocr_latin_langs, pytesseract
                )
                text = pytesseract.image_to_string(img, lang=lang_used)

            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("empty_ocr_text")

            ocr_path.write_text(text, encoding="utf-8")
            doc["status"] = "ocr_done"
            doc["ocr_engine"] = "tesseract"
            doc["ocr_lang_used"] = lang_used
            doc["ocr_script"] = script_used
            doc["last_error"] = None
            doc["updated_at"] = now_iso_utc()

            premis.add_object(f"obj:ocr:{doc_id}", str(ocr_path), fmt="text/plain")
            premis.add_event(
                "ocr",
                "success",
                {"doc_id": doc_id, "chars": len(text)},
                object_id=f"obj:ocr:{doc_id}",
                object_path=str(ocr_path),
                agents=["agent:tesseract"],
            )

        except Exception as exc:
            doc["status"] = "ocr_error"
            doc["ocr_engine"] = "tesseract"
            doc["last_error"] = str(exc)
            doc["updated_at"] = now_iso_utc()
            premis.add_event(
                "ocr",
                "failure",
                {"doc_id": doc_id, "error": str(exc)},
                object_id=f"obj:image:{doc_id}",
                object_path=str(image_path),
                agents=["agent:tesseract"],
            )
            logger.warn(f"OCR failed for {doc_id}: {exc}")

        save_state(run_paths["manifests"], state)
        logger.info(progress_line("OCR", done, total, started))


def _run_ocr_stage_paddle(
    state: dict[str, Any],
    run_paths: dict[str, Path],
    logger: PipelineLogger,
    premis: PremisRecorder,
    paddle_lang: str,
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
        logger.info("OCR stage skipped: no pending documents")
        return

    try:
        _ = __import__("paddleocr")
    except Exception as exc:
        raise RuntimeError(
            "Missing PaddleOCR dependencies. Install compatible packages (for example: paddleocr + paddlepaddle)."
        ) from exc

    use_gpu = resolve_paddle_use_gpu(paddle_device)
    total = len(pending)
    started = time.perf_counter()
    ocr_cache: dict[str, Any] = {}

    premis.add_event(
        "ocr",
        "started",
        {
            "total_docs": total,
            "engine": "paddle",
            "lang": paddle_lang,
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
        object_id="batch:ocr",
        agents=["agent:pipeline", "agent:paddleocr"],
    )

    done = 0
    paddle_temp_dir = run_paths["ocr"] / "_paddle_tmp"
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
                paddle_lang=paddle_lang,
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
            ocr = ocr_cache[lang_used]
            prepared_image = prepare_image_for_paddle(image_path=image_path, temp_dir=paddle_temp_dir)
            result = run_paddle_ocr(ocr_engine=ocr, image_path=prepared_image, angle_cls=paddle_angle_cls)
            text = extract_text_from_paddle_result(result)
            if not text:
                raise RuntimeError("empty_ocr_text")

            ocr_path.write_text(text, encoding="utf-8")
            doc["status"] = "ocr_done"
            doc["ocr_engine"] = "paddle"
            doc["ocr_lang_used"] = lang_used
            doc["ocr_script"] = script_used
            doc["last_error"] = None
            doc["updated_at"] = now_iso_utc()

            premis.add_object(f"obj:ocr:{doc_id}", str(ocr_path), fmt="text/plain")
            premis.add_event(
                "ocr",
                "success",
                {"doc_id": doc_id, "chars": len(text), "engine": "paddle", "lang": lang_used},
                object_id=f"obj:ocr:{doc_id}",
                object_path=str(ocr_path),
                agents=["agent:paddleocr"],
            )

        except Exception as exc:
            doc["status"] = "ocr_error"
            doc["ocr_engine"] = "paddle"
            doc["last_error"] = str(exc)
            doc["updated_at"] = now_iso_utc()
            premis.add_event(
                "ocr",
                "failure",
                {"doc_id": doc_id, "error": str(exc), "engine": "paddle"},
                object_id=f"obj:image:{doc_id}",
                object_path=str(image_path),
                agents=["agent:paddleocr"],
            )
            logger.warn(f"OCR failed for {doc_id}: {exc}")

        save_state(run_paths["manifests"], state)
        logger.info(progress_line("OCR", done, total, started))


def run_llm_stage(
    state: dict[str, Any],
    run_paths: dict[str, Path],
    logger: PipelineLogger,
    premis: PremisRecorder,
    class_map: dict[int, str],
    system_prompt: str,
    model: str,
    endpoint: str,
    max_retries: int,
    timeout: int,
    max_ocr_chars: int,
    num_predict: int,
    fallback_num_predict: int,
    invalid_json_fallback_retry: int,
    inference_schema: str,
    temperature: float,
    top_p: float,
    top_k: int,
    ambiguous_votes: int,
    vote_temperature: float,
    vote_top_p: float,
    vote_top_k: int,
    ambiguity_min_chars: int,
    ambiguity_min_unique_words: int,
    ambiguity_max_digit_ratio: float,
    fewshot_block: str,
) -> None:
    name_to_id = build_name_to_id(class_map)

    queue = [d for d in state["docs"].values() if d["status"] == "ocr_done"]
    if not queue:
        logger.info("LLM stage skipped: no OCR-ready documents")
        return

    total = len(queue)
    started = time.perf_counter()

    premis.add_event(
        "llm_inference",
        "started",
        {
            "total_docs": total,
            "model": model,
            "inference_schema": inference_schema,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "ambiguous_votes": ambiguous_votes,
            "invalid_json_fallback_retry": invalid_json_fallback_retry,
        },
        object_id="batch:llm",
        agents=["agent:pipeline", "agent:ollama"],
    )

    def infer_once(
        *,
        doc_id: str,
        ocr_text: str,
        user_prompt: str,
        vote_temp: float,
        vote_top_p_local: float,
        vote_top_k_local: int,
        vote_num_predict: int,
    ) -> tuple[dict[str, Any], int, int]:
        """
        Returns:
          normalized prediction, api_call_count, repair_call_count
        """
        calls = 0
        repairs = 0
        last_exc: Exception | None = None

        for _ in range(1, max_retries + 2):
            response_text = ""
            try:
                response_text = call_ollama(
                    endpoint=endpoint,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=timeout,
                    num_predict=vote_num_predict,
                    temperature=vote_temp,
                    top_p=vote_top_p_local,
                    top_k=vote_top_k_local,
                )
                calls += 1
                payload = parse_json_object(response_text)
                normalized = normalize_prediction_payload(
                    payload,
                    doc_id,
                    class_map,
                    name_to_id,
                    inference_schema=inference_schema,
                )
                return normalized, calls, repairs
            except ValidationError as exc:
                last_exc = exc
                if invalid_json_fallback_retry <= 0:
                    continue
                for _ in range(invalid_json_fallback_retry):
                    try:
                        repair_prompt = build_json_repair_prompt(
                            doc_id=doc_id,
                            ocr_text=ocr_text,
                            inference_schema=inference_schema,
                            previous_response=response_text,
                        )
                        repaired_text = call_ollama(
                            endpoint=endpoint,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=repair_prompt,
                            timeout=timeout,
                            num_predict=fallback_num_predict,
                            temperature=0.0,
                            top_p=0.9,
                            top_k=20,
                        )
                        calls += 1
                        repairs += 1
                        repaired_payload = parse_json_object(repaired_text)
                        normalized = normalize_prediction_payload(
                            repaired_payload,
                            doc_id,
                            class_map,
                            name_to_id,
                            inference_schema=inference_schema,
                        )
                        return normalized, calls, repairs
                    except Exception as repair_exc:
                        last_exc = repair_exc
                        continue
            except Exception as exc:
                last_exc = exc
                continue

        raise last_exc or RuntimeError("unknown_llm_inference_error")

    done = 0
    for doc in queue:
        done += 1
        doc_id = doc["doc_id"]
        ocr_path = Path(doc["ocr_path"] or "")

        try:
            ocr_text = ocr_path.read_text(encoding="utf-8")
        except Exception as exc:
            doc["status"] = "ocr_error"
            doc["last_error"] = f"cannot_read_ocr: {exc}"
            doc["updated_at"] = now_iso_utc()
            save_state(run_paths["manifests"], state)
            logger.warn(f"Skipping {doc_id}, OCR unreadable: {exc}")
            continue

        if max_ocr_chars > 0 and len(ocr_text) > max_ocr_chars:
            ocr_text = ocr_text[:max_ocr_chars]

        user_prompt = build_user_prompt(
            doc_id=doc_id,
            ocr_text=ocr_text,
            fewshot_block=fewshot_block,
            inference_schema=inference_schema,
        )

        should_vote = ambiguous_votes > 1 and is_ambiguous_ocr_text(
            text=ocr_text,
            min_chars=ambiguity_min_chars,
            min_unique_words=ambiguity_min_unique_words,
            max_digit_ratio=ambiguity_max_digit_ratio,
        )
        votes_requested = max(1, ambiguous_votes if should_vote else 1)
        vote_predictions: list[dict[str, Any]] = []
        vote_errors: list[str] = []
        calls_total = 0
        repairs_total = 0

        for vote_idx in range(votes_requested):
            vote_temp = temperature if vote_idx == 0 else vote_temperature
            vote_top_p_local = top_p if vote_idx == 0 else vote_top_p
            vote_top_k_local = top_k if vote_idx == 0 else vote_top_k
            try:
                prediction, calls, repairs = infer_once(
                    doc_id=doc_id,
                    ocr_text=ocr_text,
                    user_prompt=user_prompt,
                    vote_temp=vote_temp,
                    vote_top_p_local=vote_top_p_local,
                    vote_top_k_local=vote_top_k_local,
                    vote_num_predict=num_predict,
                )
                vote_predictions.append(prediction)
                calls_total += calls
                repairs_total += repairs
            except ValidationError as exc:
                vote_errors.append(f"validation_error: {exc}")
                logger.warn(f"Validation failed for {doc_id} (vote {vote_idx + 1}/{votes_requested}): {exc}")
            except Exception as exc:
                vote_errors.append(f"llm_error: {exc}")
                logger.warn(f"LLM failed for {doc_id} (vote {vote_idx + 1}/{votes_requested}): {exc}")

        if vote_predictions:
            normalized = majority_vote_predictions(vote_predictions, class_map)
            normalized["inference_schema"] = inference_schema
            normalized["votes_mode"] = "ambiguous" if should_vote else "single"
            normalized["votes_requested"] = votes_requested
            normalized["votes_succeeded"] = len(vote_predictions)
            normalized["repair_calls"] = repairs_total

            pred_path = run_paths["predictions"] / f"{doc_id}.json"
            pred_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

            doc["prediction_path"] = str(pred_path)
            doc["status"] = "ok"
            doc["last_error"] = None
            doc["updated_at"] = now_iso_utc()
            doc["prediction"] = normalized
            doc["attempts"] = calls_total

            premis.add_object(f"obj:pred:{doc_id}", str(pred_path), fmt="application/json")
            premis.add_event(
                "llm_inference",
                "success",
                {
                    "doc_id": doc_id,
                    "model": model,
                    "attempts": calls_total,
                    "repairs": repairs_total,
                    "votes_requested": votes_requested,
                    "votes_succeeded": len(vote_predictions),
                    "class_id": normalized["class_id"],
                    "confidence": normalized["confidence"],
                },
                object_id=f"obj:pred:{doc_id}",
                object_path=str(pred_path),
                agents=["agent:ollama"],
            )
        else:
            final_status = "validation_error" if vote_errors and all(e.startswith("validation_error:") for e in vote_errors) else "llm_error"
            doc["status"] = final_status
            doc["last_error"] = vote_errors[-1] if vote_errors else "no_predictions_generated"
            doc["updated_at"] = now_iso_utc()
            doc["attempts"] = calls_total
            premis.add_event(
                "llm_inference",
                "failure",
                {
                    "doc_id": doc_id,
                    "model": model,
                    "attempts": calls_total,
                    "repairs": repairs_total,
                    "votes_requested": votes_requested,
                    "votes_succeeded": 0,
                    "status": final_status,
                    "error": doc["last_error"],
                },
                object_id=f"obj:ocr:{doc_id}",
                object_path=str(ocr_path),
                agents=["agent:ollama"],
            )

        save_state(run_paths["manifests"], state)
        logger.info(progress_line("LLM", done, total, started))


def compute_metrics(rows: list[dict[str, Any]], class_map: dict[int, str]) -> dict[str, Any]:
    n = len(class_map)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    total = len(rows)
    ok_rows = [r for r in rows if r["status"] == "ok" and isinstance(r.get("pred_class_id"), int)]

    for r in ok_rows:
        t = int(r["true_class_id"])
        p = int(r["pred_class_id"])
        matrix[t][p] += 1

    correct_ok = sum(matrix[i][i] for i in range(n))
    ok_total = len(ok_rows)
    strict_correct = sum(1 for r in rows if r["status"] == "ok" and r.get("pred_class_id") == r["true_class_id"])

    accuracy = (correct_ok / ok_total) if ok_total else 0.0
    strict_accuracy = (strict_correct / total) if total else 0.0
    coverage = (ok_total / total) if total else 0.0

    per_class: list[dict[str, Any]] = []
    f1_sum = 0.0
    for cid in range(n):
        tp = matrix[cid][cid]
        fp = sum(matrix[r][cid] for r in range(n) if r != cid)
        fn = sum(matrix[cid][c] for c in range(n) if c != cid)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(matrix[cid])
        f1_sum += f1
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

    macro_f1 = f1_sum / n if n else 0.0

    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    return {
        "generated_at": now_iso_utc(),
        "num_total": total,
        "num_ok": ok_total,
        "coverage": coverage,
        "accuracy": accuracy,
        "strict_accuracy": strict_accuracy,
        "macro_f1": macro_f1,
        "status_counts": status_counts,
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def export_results(
    state: dict[str, Any],
    run_paths: dict[str, Path],
    class_map: dict[int, str],
    logger: PipelineLogger,
    premis: PremisRecorder,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for doc in state["docs"].values():
        pred_obj: dict[str, Any] | None = None
        if doc.get("prediction_path"):
            p = Path(doc["prediction_path"])
            if p.exists():
                try:
                    pred_obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    pred_obj = None

        pred_class_id = pred_obj.get("class_id") if isinstance(pred_obj, dict) else None
        pred_class_name = pred_obj.get("class_name") if isinstance(pred_obj, dict) else None
        confidence = pred_obj.get("confidence") if isinstance(pred_obj, dict) else None
        rationale = pred_obj.get("rationale") if isinstance(pred_obj, dict) else None

        row = {
            "doc_id": doc["doc_id"],
            "rel_path": doc["rel_path"],
            "status": doc["status"],
            "true_class_id": doc["class_id"],
            "true_class_name": doc["class_name"],
            "pred_class_id": pred_class_id,
            "pred_class_name": pred_class_name,
            "confidence": confidence,
            "rationale": rationale,
            "attempts": doc.get("attempts", 0),
            "last_error": doc.get("last_error"),
            "ocr_path": doc.get("ocr_path"),
            "prediction_path": doc.get("prediction_path"),
        }
        row["is_correct"] = bool(row["status"] == "ok" and row["pred_class_id"] == row["true_class_id"])
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["doc_id"])

    predictions_jsonl = run_paths["predictions"] / "predictions.jsonl"
    predictions_csv = run_paths["predictions"] / "predictions.csv"
    errors_jsonl = run_paths["evaluation"] / "errors.jsonl"
    hits_jsonl = run_paths["evaluation"] / "hits.jsonl"
    misses_jsonl = run_paths["evaluation"] / "misses.jsonl"

    for f in [predictions_jsonl, errors_jsonl, hits_jsonl, misses_jsonl]:
        if f.exists():
            f.unlink()

    for r in rows:
        append_jsonl(predictions_jsonl, r)
        if r["status"] != "ok":
            append_jsonl(errors_jsonl, r)
        if r["is_correct"]:
            append_jsonl(hits_jsonl, r)
        else:
            append_jsonl(misses_jsonl, r)

    csv_fields = [
        "doc_id",
        "rel_path",
        "status",
        "true_class_id",
        "true_class_name",
        "pred_class_id",
        "pred_class_name",
        "confidence",
        "rationale",
        "attempts",
        "last_error",
        "is_correct",
        "ocr_path",
        "prediction_path",
    ]
    with predictions_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = compute_metrics(rows, class_map)
    metrics_path = run_paths["evaluation"] / "metrics.json"
    atomic_write_json(metrics_path, metrics)

    confusion_csv = run_paths["evaluation"] / "confusion_matrix.csv"
    with confusion_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["true_class_id", "true_class_name"] + [f"pred_{i}" for i in range(len(class_map))]
        writer.writerow(header)
        for cid, row_vals in enumerate(metrics["confusion_matrix"]):
            writer.writerow([cid, class_map[cid], *row_vals])

    premis.add_object("obj:metrics", str(metrics_path), fmt="application/json")
    premis.add_event(
        "export",
        "success",
        {
            "predictions_jsonl": str(predictions_jsonl),
            "predictions_csv": str(predictions_csv),
            "metrics": str(metrics_path),
            "confusion_matrix": str(confusion_csv),
        },
        object_id="obj:metrics",
        object_path=str(metrics_path),
        agents=["agent:pipeline"],
    )

    logger.info(
        "Evaluation summary: "
        f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
        f"strict_accuracy={metrics['strict_accuracy']:.4f}, coverage={metrics['coverage']:.4f}"
    )

    return metrics


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
    if sorted(class_map.keys()) != list(range(16)):
        raise RuntimeError("Class map must contain IDs 0..15")

    system_prompt = load_system_prompt(args.system_prompt_file)
    static_fewshot = system_prompt_has_static_fewshot(system_prompt)
    disable_fewshot_effective = args.disable_fewshot or static_fewshot

    premis = PremisRecorder(run_paths["premis"], args.run_id, logger)
    premis.ensure_defaults(args.model)

    logger.info(f"Run id: {args.run_id}")
    logger.info(f"Labels file: {args.labels_file}")
    logger.info(f"Images root: {args.images_root}")
    logger.info(f"Model: {args.model}")
    logger.info(
        "LLM inference config: "
        f"schema={args.inference_schema}, temperature={args.temperature}, top_p={args.top_p}, "
        f"top_k={args.top_k}, num_predict={args.num_predict}, fallback_num_predict={args.fallback_num_predict}, "
        f"invalid_json_fallback_retry={args.invalid_json_fallback_retry}, ambiguous_votes={args.ambiguous_votes}, "
        f"vote_temperature={args.vote_temperature}, vote_top_p={args.vote_top_p}, vote_top_k={args.vote_top_k}"
    )
    logger.info(f"OCR engine: {args.ocr_engine}")
    if args.ocr_engine == "tesseract":
        logger.info(f"Tesseract language: {args.ocr_lang}")
        if args.ocr_auto_script:
            logger.info(f"Tesseract auto script enabled; Latin langs: {args.ocr_latin_langs}")
    else:
        logger.info(
            "Paddle config: "
            f"lang={args.paddle_lang}, latin_lang={args.paddle_latin_lang}, "
            f"auto_script={args.paddle_auto_script}, device={args.paddle_device}, "
            f"angle_cls={args.paddle_angle_cls}, det_limit_side_len={args.paddle_det_limit_side_len}, "
            f"det_db_thresh={args.paddle_det_db_thresh}, det_db_box_thresh={args.paddle_det_db_box_thresh}, "
            f"det_db_unclip_ratio={args.paddle_det_db_unclip_ratio}, "
            f"rec_score_thresh={args.paddle_rec_score_thresh}, drop_score={args.paddle_drop_score}"
        )
    if static_fewshot:
        logger.info("Static few-shot detected in system prompt; dynamic few-shot will be disabled.")
    if args.sample_manifest:
        logger.info(f"Sample manifest source: {args.sample_manifest}")

    if not args.labels_file.exists():
        raise RuntimeError(f"Labels file not found: {args.labels_file}")
    if not args.images_root.exists():
        raise RuntimeError(f"Images root not found: {args.images_root}")

    premis.add_event(
        "ingest",
        "started",
        {
            "labels_file": str(args.labels_file),
            "images_root": str(args.images_root),
            "sample_size": args.sample_size,
            "seed": args.seed,
        },
        object_id="obj:labels",
        object_path=str(args.labels_file),
        agents=["agent:pipeline"],
    )

    records = load_labels(args.labels_file, class_map)
    logger.info(f"Loaded labels rows: {len(records)}")

    sample_rows = load_or_create_sample_manifest(
        records=records,
        manifests_dir=run_paths["manifests"],
        sample_size=args.sample_size,
        seed=args.seed,
        images_root=args.images_root,
        class_map=class_map,
        resume=args.resume,
        sample_manifest_source=args.sample_manifest,
    )

    sample_class_coverage = sorted({r["class_id"] for r in sample_rows})
    if sample_class_coverage != list(range(16)):
        raise RuntimeError(f"Sample does not cover all classes: {sample_class_coverage}")

    premis.add_event(
        "sampling",
        "success",
        {
            "sample_size": len(sample_rows),
            "seed": args.seed,
            "class_coverage": sample_class_coverage,
        },
        object_id="obj:sample",
        object_path=str(run_paths["manifests"] / "sample.jsonl"),
        agents=["agent:pipeline"],
    )

    run_config = {
        "run_id": args.run_id,
        "labels_file": str(args.labels_file),
        "images_root": str(args.images_root),
        "model": args.model,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "resume": args.resume,
        "max_retries": args.max_retries,
        "request_timeout": args.request_timeout,
        "max_ocr_chars": args.max_ocr_chars,
        "num_predict": args.num_predict,
        "fallback_num_predict": args.fallback_num_predict,
        "invalid_json_fallback_retry": args.invalid_json_fallback_retry,
        "inference_schema": args.inference_schema,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "ambiguous_votes": args.ambiguous_votes,
        "vote_temperature": args.vote_temperature,
        "vote_top_p": args.vote_top_p,
        "vote_top_k": args.vote_top_k,
        "ambiguity_min_chars": args.ambiguity_min_chars,
        "ambiguity_min_unique_words": args.ambiguity_min_unique_words,
        "ambiguity_max_digit_ratio": args.ambiguity_max_digit_ratio,
        "ocr_engine": args.ocr_engine,
        "ocr_lang": args.ocr_lang,
        "ocr_auto_script": args.ocr_auto_script,
        "ocr_latin_langs": args.ocr_latin_langs,
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
        "disable_fewshot": args.disable_fewshot,
        "disable_fewshot_effective": disable_fewshot_effective,
        "static_fewshot_in_system_prompt": static_fewshot,
        "fewshot_labels_file": str(args.fewshot_labels_file),
        "fewshot_seed": args.fewshot_seed,
        "fewshot_max_chars": args.fewshot_max_chars,
        "ollama_endpoint": args.ollama_endpoint,
        "sample_manifest_source": str(args.sample_manifest) if args.sample_manifest else None,
        "started_at": now_iso_utc(),
    }

    state = load_or_init_state(
        manifests_dir=run_paths["manifests"],
        sample_rows=sample_rows,
        run_config=run_config,
        resume=args.resume,
    )

    run_ocr_stage(
        state=state,
        run_paths=run_paths,
        logger=logger,
        premis=premis,
        ocr_engine=args.ocr_engine,
        ocr_lang=args.ocr_lang,
        ocr_auto_script=args.ocr_auto_script,
        ocr_latin_langs=args.ocr_latin_langs,
        paddle_lang=args.paddle_lang,
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

    fewshot_block = ""
    if disable_fewshot_effective:
        if args.disable_fewshot:
            logger.info("Few-shot disabled via --disable-fewshot")
        elif static_fewshot:
            logger.info("Few-shot disabled: static examples already in system prompt")
    else:
        exclude_rel_paths = {d["rel_path"] for d in state["docs"].values()}
        fewshot_rows = load_or_create_fewshot_examples(
            manifests_dir=run_paths["manifests"],
            fewshot_labels_file=args.fewshot_labels_file,
            images_root=args.images_root,
            class_map=class_map,
            exclude_rel_paths=exclude_rel_paths,
            ocr_engine=args.ocr_engine,
            ocr_lang=args.ocr_lang,
            ocr_auto_script=args.ocr_auto_script,
            ocr_latin_langs=args.ocr_latin_langs,
            paddle_lang=args.paddle_lang,
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
            seed=args.fewshot_seed,
            max_chars=args.fewshot_max_chars,
            resume=args.resume,
            logger=logger,
            premis=premis,
        )
        fewshot_block = build_fewshot_block(fewshot_rows)
        logger.info(f"Few-shot block chars: {len(fewshot_block)}")

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
        fallback_num_predict=args.fallback_num_predict,
        invalid_json_fallback_retry=args.invalid_json_fallback_retry,
        inference_schema=args.inference_schema,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ambiguous_votes=args.ambiguous_votes,
        vote_temperature=args.vote_temperature,
        vote_top_p=args.vote_top_p,
        vote_top_k=args.vote_top_k,
        ambiguity_min_chars=args.ambiguity_min_chars,
        ambiguity_min_unique_words=args.ambiguity_min_unique_words,
        ambiguity_max_digit_ratio=args.ambiguity_max_digit_ratio,
        fewshot_block=fewshot_block,
    )

    premis.add_event(
        "evaluation",
        "started",
        {"run_id": args.run_id},
        object_id="batch:evaluation",
        agents=["agent:pipeline"],
    )

    metrics = export_results(
        state=state,
        run_paths=run_paths,
        class_map=class_map,
        logger=logger,
        premis=premis,
    )

    save_state(run_paths["manifests"], state)

    premis.add_event(
        "evaluation",
        "success",
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "strict_accuracy": metrics["strict_accuracy"],
            "coverage": metrics["coverage"],
        },
        object_id="obj:metrics",
        object_path=str(run_paths["evaluation"] / "metrics.json"),
        agents=["agent:pipeline"],
    )

    logger.info("Pipeline finished successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        raise
