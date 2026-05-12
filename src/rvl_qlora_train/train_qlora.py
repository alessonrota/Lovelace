from __future__ import annotations

import time
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import atomic_write_json, now_iso_utc, read_jsonl


@dataclass
class QLoRAConfig:
    base_model: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    num_train_epochs: float
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    weight_decay: float
    max_grad_norm: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    max_seq_len: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    optim: str
    gradient_checkpointing: bool
    max_train_hours: float
    seed: int


class _SupervisedDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int):
        self.examples: list[dict[str, Any]] = []
        eos_id = tokenizer.eos_token_id
        for row in rows:
            prompt = str(row.get("prompt", "")).strip()
            target = str(row.get("target", "")).strip()
            if not prompt or not target:
                continue

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                target_ids = target_ids + [eos_id]

            input_ids = prompt_ids + target_ids
            labels = ([-100] * len(prompt_ids)) + target_ids

            if len(input_ids) > max_seq_len:
                overflow = len(input_ids) - max_seq_len
                input_ids = input_ids[overflow:]
                labels = labels[overflow:]

            if not input_ids or all(x == -100 for x in labels):
                continue

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class _Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for feat in features:
            n = len(feat["input_ids"])
            pad_n = max_len - n
            input_ids.append(feat["input_ids"] + [self.pad_token_id] * pad_n)
            attention_mask.append(feat["attention_mask"] + [0] * pad_n)
            labels.append(feat["labels"] + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class _WallClockStopCallback:
    def __init__(self, max_hours: float):
        self.max_seconds = max(0.0, float(max_hours) * 3600.0)
        self.started_at = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        elapsed = time.perf_counter() - self.started_at
        if self.max_seconds > 0 and elapsed >= self.max_seconds:
            control.should_training_stop = True
        return control


def load_qlora_config(config_path: Path, base_model_override: str | None = None) -> QLoRAConfig:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("Missing dependency: pyyaml") from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Invalid config yaml: {config_path}")

    lora = raw.get("lora", {})
    training = raw.get("training", {})

    return QLoRAConfig(
        base_model=base_model_override or str(raw.get("base_model")),
        load_in_4bit=bool(raw.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(raw.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_use_double_quant=bool(raw.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=str(raw.get("bnb_4bit_compute_dtype", "auto")),
        lora_r=int(lora.get("r", 32)),
        lora_alpha=int(lora.get("alpha", 64)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=[str(x) for x in lora.get("target_modules", ["q_proj", "v_proj"])],
        num_train_epochs=float(training.get("num_train_epochs", 1.0)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(training.get("lr_scheduler_type", "cosine")),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 16)),
        max_seq_len=int(training.get("max_seq_len", 3072)),
        logging_steps=int(training.get("logging_steps", 10)),
        eval_steps=int(training.get("eval_steps", 100)),
        save_steps=int(training.get("save_steps", 100)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        optim=str(training.get("optim", "paged_adamw_8bit")),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        max_train_hours=float(training.get("max_train_hours", 6.0)),
        seed=int(training.get("seed", 42)),
    )


def _resolve_dtype(compute_dtype_cfg: str) -> Any:
    import torch

    if compute_dtype_cfg == "bfloat16":
        return torch.bfloat16
    if compute_dtype_cfg == "float16":
        return torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _find_last_checkpoint(checkpoints_dir: Path) -> str | None:
    if not checkpoints_dir.exists():
        return None
    candidates = [p for p in checkpoints_dir.glob("checkpoint-*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(candidates[-1])


def run_qlora_training(
    train_jsonl: Path,
    val_jsonl: Path,
    checkpoints_dir: Path,
    logger: Any,
    config: QLoRAConfig,
    resume: bool,
    gpu_index: int = 0,
) -> dict[str, Any]:
    # Restrict visibility to a single physical GPU to prevent Trainer from wrapping
    # the model in DataParallel across multiple cards (which caused OOM on GPU 1).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install: torch transformers peft bitsandbytes accelerate datasets trl pyyaml"
        ) from exc

    train_rows = read_jsonl(train_jsonl)
    val_rows = read_jsonl(val_jsonl)
    if not train_rows:
        raise RuntimeError(f"Train dataset is empty: {train_jsonl}")

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    dtype = _resolve_dtype(config.bnb_4bit_compute_dtype)

    logger.info(f"Loading tokenizer: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Building datasets: train={len(train_rows)}, val={len(val_rows)}")
    train_ds = _SupervisedDataset(train_rows, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
    val_ds = _SupervisedDataset(val_rows, tokenizer=tokenizer, max_seq_len=config.max_seq_len)
    if len(train_ds) == 0:
        raise RuntimeError("No valid training examples after tokenization")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=dtype,
    )

    # After CUDA_VISIBLE_DEVICES restriction, selected GPU is logical cuda:0.
    logger.info(f"Loading base model in 4-bit on physical cuda:{gpu_index} (logical cuda:0)")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_cfg,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    trainable_params = 0
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    class WallClockStopCallback(TrainerCallback):
        def __init__(self, max_hours: float):
            self.impl = _WallClockStopCallback(max_hours=max_hours)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            return self.impl.on_step_end(args=args, state=state, control=control, **kwargs)

    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_mode = "steps" if len(val_ds) > 0 else "no"
    training_kwargs: dict[str, Any] = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_strategy": "steps",
        "save_total_limit": config.save_total_limit,
        "optim": config.optim,
        "fp16": (dtype == torch.float16),
        "bf16": (dtype == torch.bfloat16),
        "report_to": [],
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }
    # transformers <5 used "evaluation_strategy"; >=5 uses "eval_strategy"
    if "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = eval_mode
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = eval_mode
    # transformers <5 had overwrite_output_dir, removed in >=5
    if "overwrite_output_dir" in ta_params:
        training_kwargs["overwrite_output_dir"] = False
    # transformers >=5 can aggressively release cached blocks between steps.
    if "torch_empty_cache_steps" in ta_params:
        training_kwargs["torch_empty_cache_steps"] = 1

    # Keep only kwargs supported by the installed transformers version.
    training_kwargs = {k: v for k, v in training_kwargs.items() if k in ta_params}
    training_args = TrainingArguments(**training_kwargs)

    data_collator = _Collator(pad_token_id=int(tokenizer.pad_token_id))

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds if len(val_ds) > 0 else None,
        "data_collator": data_collator,
        "callbacks": [WallClockStopCallback(config.max_train_hours)],
    }
    trainer_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    # transformers <5: tokenizer
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    # transformers >=5: processing_class
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    resume_ckpt = _find_last_checkpoint(checkpoints_dir) if resume else None
    if resume_ckpt:
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")

    started_at = time.perf_counter()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_elapsed = time.perf_counter() - started_at

    eval_metrics = trainer.evaluate() if len(val_ds) > 0 else {}

    final_adapter_dir = checkpoints_dir / "adapter_final"
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    trainer.save_state()

    metrics = dict(train_result.metrics)
    for k, v in eval_metrics.items():
        metrics[f"val_{k}"] = v

    result = {
        "generated_at": now_iso_utc(),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_examples_tokenized": len(train_ds),
        "val_examples_tokenized": len(val_ds),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
        "base_model": config.base_model,
        "max_seq_len": config.max_seq_len,
        "target_modules": config.target_modules,
        "elapsed_seconds": train_elapsed,
        "adapter_dir": str(final_adapter_dir),
        "metrics": metrics,
    }

    atomic_write_json(checkpoints_dir / "training_summary.json", result)
    return result
