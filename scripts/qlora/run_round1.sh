#!/usr/bin/env bash
set -euo pipefail

cd /home/arr/Documents/GitHub/Lovelace

RUN_ID="${RUN_ID:-qlora_r1_15cls}"
RUN_DIR="data/processed/qlora-qwen14b/${RUN_ID}"
OCR_DEVICE="${OCR_DEVICE:-gpu}"
OCR_WORKERS="${OCR_WORKERS:-0}"
OCR_MAX_CPU_UTIL="${OCR_MAX_CPU_UTIL:-0.6}"
PADDLE_GPU_MEM_FRACTION="${PADDLE_GPU_MEM_FRACTION:-0.6}"
OCR_WORKER_THREADS="${OCR_WORKER_THREADS:-1}"
LLM_GPU_INDEX="${LLM_GPU_INDEX:-0}"
QLORA_CONFIG="${QLORA_CONFIG:-configs/qlora/qlora_qwen14b_r1.yaml}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF
RESUME_FLAG=()
if [ -d "${RUN_DIR}" ] && [ -n "$(ls -A "${RUN_DIR}" 2>/dev/null || true)" ]; then
  echo "[INFO] Existing run directory detected: ${RUN_DIR} (enabling --resume)"
  RESUME_FLAG=(--resume)
fi

python -m src.rvl_qlora_train.main \
  --labels-train data/rvl-cdip/labels/train.txt \
  --labels-val data/rvl-cdip/labels/val.txt \
  --labels-test data/rvl-cdip/labels/test.txt \
  --images-root data/rvl-cdip/images \
  --base-model Qwen/Qwen2.5-14B-Instruct \
  --run-id "${RUN_ID}" \
  --sample-train-per-class 2000 \
  --sample-val-per-class 300 \
  --sample-test-per-class 300 \
  --seed 42 \
  --llm-gpu-index "${LLM_GPU_INDEX}" \
  --ocr-engine paddle \
  --paddle-variant server_ch \
  --paddle-lang en \
  --paddle-latin-lang en \
  --paddle-device "${OCR_DEVICE}" \
  --ocr-workers "${OCR_WORKERS}" \
  --ocr-max-cpu-util "${OCR_MAX_CPU_UTIL}" \
  --paddle-gpu-mem-fraction "${PADDLE_GPU_MEM_FRACTION}" \
  --ocr-worker-threads "${OCR_WORKER_THREADS}" \
  --paddle-angle-cls \
  --system-prompt-file configs/qlora/rvl_train_prompt_v1_minjson.txt \
  --qlora-config "${QLORA_CONFIG}" \
  --legacy-sample-manifest data/processed/saida-ocr-class/run_001/manifests/sample.jsonl \
  --max-ocr-chars 12000 \
  --min-ocr-chars 40 \
  --max-eval-new-tokens 12 \
  --evaluate-test \
  "${RESUME_FLAG[@]}"
