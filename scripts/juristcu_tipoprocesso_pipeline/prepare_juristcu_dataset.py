#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
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

from src.rvl_qlora_train.common import PipelineLogger, atomic_write_json, make_doc_id, now_iso_utc


TOP15_CLASSES: list[str] = [
    "TOMADA DE CONTAS ESPECIAL",
    "REPRESENTAÇÃO",
    "APOSENTADORIA",
    "RELATÓRIO DE AUDITORIA",
    "PENSÃO CIVIL",
    "PRESTAÇÃO DE CONTAS",
    "CONSULTA",
    "RELATÓRIO DE LEVANTAMENTO",
    "DENÚNCIA",
    "ADMINISTRATIVO",
    "ATOS DE ADMISSÃO",
    "SOLICITAÇÃO DO CONGRESSO NACIONAL",
    "PRESTAÇÃO DE CONTAS SIMPLIFICADA",
    "TOMADA DE CONTAS",
    "MONITORAMENTO",
]

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


@dataclass
class Row:
    key: str
    group_key: str
    class_id: int
    class_name: str
    text: str
    rel_path: str
    doc_id: str


@dataclass
class SplitBundle:
    train: list[Row]
    val: list[Row]
    test: list[Row]
    leakage_train_val_groups: int
    leakage_train_test_groups: int
    leakage_val_test_groups: int
    score: float
    attempts_used: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare JurisTCU TIPOPROCESSO dataset (Top15 only, no OCR).")
    p.add_argument(
        "--doc-csv",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "JurisTCU_repo" / "JurisTCU" / "doc.csv",
    )
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "juristcu-tipoprocesso",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-text-chars", type=int, default=4000)
    p.add_argument("--target-train", type=int, default=7554)
    p.add_argument("--target-val", type=int, default=3777)
    p.add_argument("--target-test", type=int, default=3778)
    p.add_argument("--expected-rows", type=int, default=15109)
    p.add_argument("--max-split-attempts", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def clean_text(text: str) -> str:
    out = html.unescape(text or "")
    out = TAG_RE.sub(" ", out)
    out = WS_RE.sub(" ", out).strip()
    return out


def make_group_key(raw: dict[str, str]) -> str:
    num = (raw.get("NUMACORDAO") or "").strip()
    ano = (raw.get("ANOACORDAO") or "").strip()
    if num and ano:
        return f"{num}_{ano}"
    key = (raw.get("KEY") or "").strip()
    if key:
        return f"KEY_{key}"
    fallback = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    digest = abs(hash(fallback))
    return f"FALLBACK_{digest}"


def _subset_sum_exact(group_ids: list[int], sizes: list[int], target: int) -> set[int] | None:
    if target < 0:
        return None
    if target == 0:
        return set()
    reachable = [False] * (target + 1)
    prev_gid = [-1] * (target + 1)
    prev_sum = [-1] * (target + 1)
    reachable[0] = True

    for gid in group_ids:
        w = sizes[gid]
        if w <= 0 or w > target:
            continue
        for s in range(target, w - 1, -1):
            if (not reachable[s]) and reachable[s - w]:
                reachable[s] = True
                prev_gid[s] = gid
                prev_sum[s] = s - w

    if not reachable[target]:
        return None

    chosen: set[int] = set()
    cur = target
    while cur > 0:
        gid = prev_gid[cur]
        if gid < 0:
            return None
        chosen.add(gid)
        cur = prev_sum[cur]
    return chosen


def _class_ratio_score(rows: list[Row], global_ratios: dict[int, float], class_ids: list[int]) -> float:
    if not rows:
        return 999999.0
    size = len(rows)
    c = Counter(r.class_id for r in rows)
    return sum(abs((c.get(cid, 0) / size) - global_ratios[cid]) for cid in class_ids)


def _split_score(train: list[Row], val: list[Row], test: list[Row], global_ratios: dict[int, float], class_ids: list[int]) -> float:
    return _class_ratio_score(train, global_ratios, class_ids) + _class_ratio_score(val, global_ratios, class_ids) + _class_ratio_score(test, global_ratios, class_ids)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compose_prompt(system_prompt: str, user_instruction: str, text: str) -> str:
    return (
        f"[SYSTEM]\n{system_prompt.strip()}\n\n"
        f"[USER]\n{user_instruction.strip()}\n"
        "texto:\n\"\"\"\n"
        f"{text}\n"
        "\"\"\"\n"
    )


def _build_system_prompt() -> str:
    return (
        "Você é um classificador de documentos jurídicos do TCU.\n"
        "Tarefa: classificar o texto em exatamente uma classe TIPOPROCESSO.\n"
        "Regras:\n"
        "- Use somente o texto fornecido.\n"
        "- Retorne apenas o id numérico da classe.\n"
        "- Não retorne palavras, JSON ou explicações."
    )


def _build_user_instruction(class_map: dict[int, str]) -> str:
    mapping = "\n".join(f"{cid}={class_map[cid]}" for cid in sorted(class_map.keys()))
    return (
        "Classifique o texto em exatamente uma classe TIPOPROCESSO da lista abaixo.\n"
        "Retorne somente o id numérico.\n\n"
        f"{mapping}\n"
    )


def _group_intersection(a: list[Row], b: list[Row]) -> int:
    aa = {r.group_key for r in a}
    bb = {r.group_key for r in b}
    return len(aa.intersection(bb))


def _prepare_rows(doc_csv: Path, class_to_id: dict[str, int], max_text_chars: int) -> tuple[list[Row], dict[str, int], dict[int, int]]:
    rows: list[Row] = []
    drop_counts = defaultdict(int)
    class_counts: Counter[int] = Counter()

    with doc_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            label = (raw.get("TIPOPROCESSO") or "").strip()
            if not label:
                drop_counts["missing_tipoprocesso"] += 1
                continue
            if label not in class_to_id:
                drop_counts["not_in_top15"] += 1
                continue

            en = clean_text(raw.get("ENUNCIADO") or "")
            ex = clean_text(raw.get("EXCERTO") or "")
            text = f"{en}\n\n{ex}".strip()
            if len(text) > max_text_chars:
                text = text[:max_text_chars]
            text = text.strip()
            if not text:
                drop_counts["empty_text"] += 1
                continue

            key = (raw.get("KEY") or "").strip()
            rel_path = f"juristcu/doc_csv_key/{key or 'sem_key'}"
            doc_id = make_doc_id(rel_path)
            group_key = make_group_key(raw)
            class_id = class_to_id[label]
            class_counts[class_id] += 1
            rows.append(
                Row(
                    key=key,
                    group_key=group_key,
                    class_id=class_id,
                    class_name=label,
                    text=text,
                    rel_path=rel_path,
                    doc_id=doc_id,
                )
            )

    return rows, dict(drop_counts), dict(class_counts)


def _attempt_group_split(
    rows: list[Row],
    target_train: int,
    target_val: int,
    target_test: int,
    seed: int,
    max_attempts: int,
    class_ids: list[int],
) -> SplitBundle:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.group_key].append(r)

    group_keys = list(groups.keys())
    key_to_gid = {g: i for i, g in enumerate(group_keys)}
    sizes = [len(groups[g]) for g in group_keys]
    group_rows = [groups[g] for g in group_keys]

    total_expected = target_train + target_val + target_test
    if sum(sizes) != total_expected:
        raise RuntimeError(
            f"Target split total ({total_expected}) does not match available rows ({sum(sizes)})."
        )

    global_counts = Counter(r.class_id for r in rows)
    global_ratios = {cid: (global_counts.get(cid, 0) / len(rows)) for cid in class_ids}

    best: SplitBundle | None = None

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        all_gids = list(range(len(group_keys)))
        rng.shuffle(all_gids)

        test_set = _subset_sum_exact(all_gids, sizes, target_test)
        if test_set is None:
            continue

        remaining = [gid for gid in all_gids if gid not in test_set]
        rng.shuffle(remaining)
        val_set = _subset_sum_exact(remaining, sizes, target_val)
        if val_set is None:
            continue

        train_set = set(remaining).difference(val_set)

        train_rows = [row for gid in sorted(train_set) for row in group_rows[gid]]
        val_rows = [row for gid in sorted(val_set) for row in group_rows[gid]]
        test_rows = [row for gid in sorted(test_set) for row in group_rows[gid]]

        if len(train_rows) != target_train or len(val_rows) != target_val or len(test_rows) != target_test:
            continue

        leakage_train_val = _group_intersection(train_rows, val_rows)
        leakage_train_test = _group_intersection(train_rows, test_rows)
        leakage_val_test = _group_intersection(val_rows, test_rows)
        if leakage_train_val or leakage_train_test or leakage_val_test:
            continue

        score = _split_score(train_rows, val_rows, test_rows, global_ratios=global_ratios, class_ids=class_ids)
        bundle = SplitBundle(
            train=train_rows,
            val=val_rows,
            test=test_rows,
            leakage_train_val_groups=leakage_train_val,
            leakage_train_test_groups=leakage_train_test,
            leakage_val_test_groups=leakage_val_test,
            score=score,
            attempts_used=attempt + 1,
        )

        if best is None or bundle.score < best.score:
            best = bundle

    if best is None:
        raise RuntimeError(
            f"Could not find exact group split after {max_attempts} attempts "
            f"(targets train/val/test={target_train}/{target_val}/{target_test})."
        )
    return best


def _to_dataset_rows(rows: list[Row], split: str, prompt_system: str, prompt_user: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        prompt = _compose_prompt(prompt_system, prompt_user, r.text)
        out.append(
            {
                "doc_id": r.doc_id,
                "split": split,
                "rel_path": r.rel_path,
                "class_id": r.class_id,
                "class_name": r.class_name,
                "prompt": prompt,
                "target": str(r.class_id),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    run_dir = args.output_root / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(
                f"Run directory exists and is not empty: {run_dir}. "
                "Use --overwrite to replace."
            )
        shutil.rmtree(run_dir)

    (run_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    logger = PipelineLogger(run_dir / "logs" / "prepare.log")
    logger.info(f"Preparing JurisTCU Top15 dataset from: {args.doc_csv}")

    if not args.doc_csv.exists():
        raise RuntimeError(f"doc.csv not found: {args.doc_csv}")

    class_map = {i: label for i, label in enumerate(TOP15_CLASSES)}
    class_to_id = {label: i for i, label in class_map.items()}

    rows, drop_counts, class_counts = _prepare_rows(
        doc_csv=args.doc_csv,
        class_to_id=class_to_id,
        max_text_chars=args.max_text_chars,
    )
    logger.info(f"Rows after top15 filter: {len(rows)}")

    if args.expected_rows > 0 and len(rows) != args.expected_rows:
        raise RuntimeError(
            f"Filtered rows mismatch. expected={args.expected_rows}, got={len(rows)}"
        )

    target_total = args.target_train + args.target_val + args.target_test
    if target_total != len(rows):
        raise RuntimeError(
            f"Target counts must sum to filtered rows. targets_total={target_total}, rows={len(rows)}"
        )

    split = _attempt_group_split(
        rows=rows,
        target_train=args.target_train,
        target_val=args.target_val,
        target_test=args.target_test,
        seed=args.seed,
        max_attempts=args.max_split_attempts,
        class_ids=sorted(class_map.keys()),
    )
    logger.info(
        "Split found: "
        f"train={len(split.train)}, val={len(split.val)}, test={len(split.test)}, "
        f"score={split.score:.6f}, attempts={split.attempts_used}"
    )

    rng = random.Random(args.seed)
    train_rows = list(split.train)
    val_rows = list(split.val)
    test_rows = list(split.test)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)

    prompt_system = _build_system_prompt()
    prompt_user = _build_user_instruction(class_map)

    train_out = _to_dataset_rows(train_rows, "train", prompt_system, prompt_user)
    val_out = _to_dataset_rows(val_rows, "val", prompt_system, prompt_user)
    test_out = _to_dataset_rows(test_rows, "test", prompt_system, prompt_user)

    _write_jsonl(run_dir / "datasets" / "train.jsonl", train_out)
    _write_jsonl(run_dir / "datasets" / "val.jsonl", val_out)
    _write_jsonl(run_dir / "datasets" / "test.jsonl", test_out)

    (run_dir / "configs" / "system_prompt.txt").write_text(prompt_system + "\n", encoding="utf-8")
    atomic_write_json(run_dir / "configs" / "class_map.json", {str(k): v for k, v in class_map.items()})

    leakage_report = {
        "generated_at": now_iso_utc(),
        "leakage_train_val_groups": split.leakage_train_val_groups,
        "leakage_train_test_groups": split.leakage_train_test_groups,
        "leakage_val_test_groups": split.leakage_val_test_groups,
        "is_clean": (
            split.leakage_train_val_groups == 0
            and split.leakage_train_test_groups == 0
            and split.leakage_val_test_groups == 0
        ),
    }
    atomic_write_json(run_dir / "reports" / "split_leakage_check.json", leakage_report)

    split_class_counts = {
        "train": Counter(r.class_id for r in train_rows),
        "val": Counter(r.class_id for r in val_rows),
        "test": Counter(r.class_id for r in test_rows),
    }
    summary = {
        "generated_at": now_iso_utc(),
        "doc_csv": str(args.doc_csv),
        "run_id": args.run_id,
        "selected_classes": TOP15_CLASSES,
        "class_map": {str(k): class_map[k] for k in sorted(class_map)},
        "rows_after_filter": len(rows),
        "expected_rows": args.expected_rows,
        "drop_counts": drop_counts,
        "global_class_counts": {str(k): class_counts.get(k, 0) for k in sorted(class_map)},
        "split_targets": {
            "train": args.target_train,
            "val": args.target_val,
            "test": args.target_test,
        },
        "split_actual": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "split_class_counts": {
            split_name: {str(cid): split_class_counts[split_name].get(cid, 0) for cid in sorted(class_map)}
            for split_name in ["train", "val", "test"]
        },
        "split_score": split.score,
        "split_attempts_used": split.attempts_used,
        "max_text_chars": args.max_text_chars,
    }
    atomic_write_json(run_dir / "reports" / "dataset_summary.json", summary)

    logger.info(f"Done. Output run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
