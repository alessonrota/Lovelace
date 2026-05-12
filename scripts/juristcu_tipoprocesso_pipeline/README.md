# JurisTCU TIPOPROCESSO Pipeline (Top 15, No OCR)

Pipeline isolado para treinar um LoRA novo em `TIPOPROCESSO` usando apenas as 15 classes mais frequentes do `doc.csv`.

## Pré-requisitos

- Repositório em: `/home/arr/Documents/GitHub/Lovelace`
- Dataset em: `data/raw/JurisTCU_repo/JurisTCU/doc.csv`
- Ambiente com dependências QLoRA (ex.: `.venv-qlora`)

## 1) Preparar dataset

```bash
cd /home/arr/Documents/GitHub/Lovelace
/home/arr/Documents/GitHub/Lovelace/.venv-qlora/bin/python \
  scripts/juristcu_tipoprocesso_pipeline/prepare_juristcu_dataset.py \
  --run-id juristcu_top15_qvko_v1
```

Saídas:
- `data/processed/juristcu-tipoprocesso/<run_id>/datasets/{train,val,test}.jsonl`
- `data/processed/juristcu-tipoprocesso/<run_id>/configs/class_map.json`
- `data/processed/juristcu-tipoprocesso/<run_id>/reports/{dataset_summary,split_leakage_check}.json`

## 2) Treinar LoRA (sem eval durante treino, checkpoint a cada 25 steps)

```bash
cd /home/arr/Documents/GitHub/Lovelace
/home/arr/Documents/GitHub/Lovelace/.venv-qlora/bin/python \
  scripts/juristcu_tipoprocesso_pipeline/train_juristcu_qlora.py \
  --run-id juristcu_top15_qvko_v1 \
  --gpu-index 0
```

## 3) Avaliação final estável (val + test)

```bash
cd /home/arr/Documents/GitHub/Lovelace
/home/arr/Documents/GitHub/Lovelace/.venv-qlora/bin/python \
  scripts/juristcu_tipoprocesso_pipeline/eval_juristcu_stable.py \
  --run-id juristcu_top15_qvko_v1 \
  --gpu-index 0
```

Saída principal:
- `data/processed/juristcu-tipoprocesso/<run_id>/evaluation_manual_stable_rest_full_<timestamp>/summary.json`
