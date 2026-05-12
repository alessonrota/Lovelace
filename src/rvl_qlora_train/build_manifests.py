from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import (
    active_class_ids,
    atomic_write_json,
    check_class_ids_are_sparse_aware,
    class_distribution,
    detect_overlap,
    make_doc_id,
    parse_labels_file,
    read_jsonl,
    sample_hash,
    stratified_pick_per_class,
)


@dataclass
class ManifestBundle:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    legacy_rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _with_doc_meta(rows: list[dict[str, Any]], images_root: Path, split_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rel_path = str(row["rel_path"])
        cid = int(row["class_id"])
        out.append(
            {
                "doc_id": make_doc_id(rel_path),
                "split": split_name,
                "rel_path": rel_path,
                "image_path": str(images_root / rel_path),
                "class_id": cid,
                "class_name": str(row["class_name"]),
            }
        )
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_or_build_split(
    out_path: Path,
    source_rows: list[dict[str, Any]],
    class_ids: list[int],
    per_class: int,
    seed: int,
    images_root: Path,
    split_name: str,
    resume: bool,
) -> list[dict[str, Any]]:
    if resume and out_path.exists():
        rows = read_jsonl(out_path)
        if rows:
            return rows

    picked = stratified_pick_per_class(rows=source_rows, class_ids=class_ids, per_class=per_class, seed=seed)
    rows = _with_doc_meta(picked, images_root=images_root, split_name=split_name)
    _write_jsonl(out_path, rows)
    return rows


def _load_or_build_legacy_eval(
    out_path: Path,
    legacy_manifest_path: Path,
    class_map: dict[int, str],
    images_root: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    if resume and out_path.exists():
        rows = read_jsonl(out_path)
        if rows:
            return rows

    if not legacy_manifest_path.exists():
        raise RuntimeError(f"Legacy sample manifest not found: {legacy_manifest_path}")

    source_rows = read_jsonl(legacy_manifest_path)
    if not source_rows:
        raise RuntimeError(f"Legacy sample manifest is empty: {legacy_manifest_path}")

    filtered: list[dict[str, Any]] = []
    for row in source_rows:
        try:
            cid = int(row.get("class_id"))
        except Exception:
            continue
        if cid not in class_map:
            continue
        rel_path = str(row.get("rel_path", "")).strip()
        if not rel_path:
            continue
        filtered.append({"rel_path": rel_path, "class_id": cid, "class_name": class_map[cid]})

    rows = _with_doc_meta(filtered, images_root=images_root, split_name="eval_legacy_93")
    _write_jsonl(out_path, rows)
    return rows


def build_manifests(
    manifests_dir: Path,
    labels_train: Path,
    labels_val: Path,
    labels_test: Path,
    images_root: Path,
    class_map: dict[int, str],
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
    legacy_sample_manifest: Path,
    resume: bool,
) -> ManifestBundle:
    check_class_ids_are_sparse_aware(class_map)

    class_ids = active_class_ids(class_map)

    train_all = parse_labels_file(labels_train, class_map)
    val_all = parse_labels_file(labels_val, class_map)
    test_all = parse_labels_file(labels_test, class_map)

    train_path = manifests_dir / "train_balanced_fast.jsonl"
    val_path = manifests_dir / "val_balanced.jsonl"
    test_path = manifests_dir / "test_holdout_balanced.jsonl"
    legacy_path = manifests_dir / "eval_legacy_93.jsonl"

    train_rows = _load_or_build_split(
        out_path=train_path,
        source_rows=train_all,
        class_ids=class_ids,
        per_class=train_per_class,
        seed=seed,
        images_root=images_root,
        split_name="train_balanced_fast",
        resume=resume,
    )
    val_rows = _load_or_build_split(
        out_path=val_path,
        source_rows=val_all,
        class_ids=class_ids,
        per_class=val_per_class,
        seed=seed + 1,
        images_root=images_root,
        split_name="val_balanced",
        resume=resume,
    )
    test_rows = _load_or_build_split(
        out_path=test_path,
        source_rows=test_all,
        class_ids=class_ids,
        per_class=test_per_class,
        seed=seed + 2,
        images_root=images_root,
        split_name="test_holdout_balanced",
        resume=resume,
    )
    legacy_rows = _load_or_build_legacy_eval(
        out_path=legacy_path,
        legacy_manifest_path=legacy_sample_manifest,
        class_map=class_map,
        images_root=images_root,
        resume=resume,
    )

    overlap_train_val = detect_overlap(train_rows, val_rows)
    overlap_train_test = detect_overlap(train_rows, test_rows)
    overlap_val_test = detect_overlap(val_rows, test_rows)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise RuntimeError(
            "Split overlap detected: "
            f"train_val={overlap_train_val}, train_test={overlap_train_test}, val_test={overlap_val_test}"
        )

    summary = {
        "class_ids": class_ids,
        "counts": {
            "train_balanced_fast": len(train_rows),
            "val_balanced": len(val_rows),
            "test_holdout_balanced": len(test_rows),
            "eval_legacy_93": len(legacy_rows),
        },
        "class_distribution": {
            "train_balanced_fast": class_distribution(train_rows, class_ids),
            "val_balanced": class_distribution(val_rows, class_ids),
            "test_holdout_balanced": class_distribution(test_rows, class_ids),
            "eval_legacy_93": class_distribution(legacy_rows, class_ids),
        },
        "hashes": {
            "train_balanced_fast": sample_hash(train_rows),
            "val_balanced": sample_hash(val_rows),
            "test_holdout_balanced": sample_hash(test_rows),
            "eval_legacy_93": sample_hash(legacy_rows),
        },
        "sources": {
            "labels_train": str(labels_train),
            "labels_val": str(labels_val),
            "labels_test": str(labels_test),
            "legacy_sample_manifest": str(legacy_sample_manifest),
        },
        "seed": seed,
        "per_class": {
            "train": train_per_class,
            "val": val_per_class,
            "test": test_per_class,
        },
    }

    atomic_write_json(manifests_dir / "sampling_summary.json", summary)

    return ManifestBundle(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        legacy_rows=legacy_rows,
        summary=summary,
    )
