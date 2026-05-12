from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any


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
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


class PipelineLogger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, message: str) -> None:
        line = f"[{now_iso_utc()}] {level.upper():5s} {message}"
        print(line, flush=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def info(self, message: str) -> None:
        self._write("info", message)

    def warn(self, message: str) -> None:
        self._write("warn", message)

    def error(self, message: str) -> None:
        self._write("error", message)


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


def make_doc_id(rel_path: str) -> str:
    stem = Path(rel_path).stem
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")
    if not stem:
        stem = "doc"
    stem = stem[:32]
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def load_class_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def active_class_ids(class_map: dict[int, str]) -> list[int]:
    return sorted(class_map.keys())


def parse_labels_file(labels_path: Path, class_map: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid labels line at {labels_path}:{i}: {line}")
            rel_path, class_id_s = parts
            class_id = int(class_id_s)
            if class_id not in class_map:
                continue
            rows.append(
                {
                    "rel_path": rel_path,
                    "class_id": class_id,
                    "class_name": class_map[class_id],
                }
            )
    return rows


def stratified_pick_per_class(
    rows: list[dict[str, Any]],
    class_ids: list[int],
    per_class: int,
    seed: int,
) -> list[dict[str, Any]]:
    if per_class <= 0:
        raise ValueError("per_class must be > 0")

    buckets: dict[int, list[dict[str, Any]]] = {cid: [] for cid in class_ids}
    for row in rows:
        cid = int(row["class_id"])
        if cid in buckets:
            buckets[cid].append(row)

    missing = [cid for cid in class_ids if not buckets[cid]]
    if missing:
        raise ValueError(f"Missing classes in source split: {missing}")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []

    for cid in class_ids:
        group = sorted(buckets[cid], key=lambda r: r["rel_path"])
        rng.shuffle(group)
        if len(group) < per_class:
            raise ValueError(
                f"Not enough samples for class {cid}. requested={per_class}, available={len(group)}"
            )
        selected.extend(group[:per_class])

    rng.shuffle(selected)
    return selected


def class_distribution(rows: list[dict[str, Any]], class_ids: list[int]) -> dict[int, int]:
    dist = {cid: 0 for cid in class_ids}
    for row in rows:
        cid = int(row["class_id"])
        if cid in dist:
            dist[cid] += 1
    return dist


def detect_overlap(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    a_paths = {str(row["rel_path"]) for row in a}
    b_paths = {str(row["rel_path"]) for row in b}
    return len(a_paths.intersection(b_paths))


def sample_hash(rows: list[dict[str, Any]]) -> str:
    norm = [f"{row['rel_path']} {int(row['class_id'])}" for row in rows]
    norm.sort()
    joined = "\n".join(norm)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def safe_div(n: float, d: float) -> float:
    return float(n) / float(d) if d else 0.0


def mean(values: list[float]) -> float:
    return safe_div(sum(values), len(values))


def eta_seconds(done: int, total: int, elapsed: float) -> float:
    if done <= 0:
        return 0.0
    rate = done / max(0.001, elapsed)
    return max(0.0, (total - done) / rate)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def expected_class_count(class_map: dict[int, str]) -> int:
    return len(class_map)


def labels_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def check_class_ids_are_sparse_aware(class_map: dict[int, str]) -> None:
    if 3 in class_map:
        raise ValueError("This training stage requires handwritten class_id=3 to be excluded")
    expected = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
    got = set(class_map.keys())
    if got != expected:
        raise ValueError(f"Class map mismatch. expected={sorted(expected)}, got={sorted(got)}")
