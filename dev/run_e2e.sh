#!/usr/bin/env bash
# End-to-end check on a proxy split: train + predict through the CLI,
# with peak memory and wall time, then strict scoring.
# Usage: dev/run_e2e.sh TAG [extra train args...]
set -euo pipefail
cd "$(dirname "$0")/.."
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
TAG=$1; shift
D=runs/proxy/$TAG
PY=.venv/bin/python
/usr/bin/time -f "train: %e s, maxrss %M KB" $PY -m hidden_rule.train \
  --train "$D/train.jsonl" --adapter "$D/adapter.safetensors" "$@" > "$D/train_report.json"
grep -E '"(steps|encode_seconds|train_seconds|adapter_scalars)"' "$D/train_report.json" | tr -d '\n'; echo
for split in test_seen test_unseen; do
  /usr/bin/time -f "predict $split: %e s, maxrss %M KB" $PY -m hidden_rule.predict \
    --data "$D/$split.jsonl" --adapter "$D/adapter.safetensors" --output "$D/$split.pred.jsonl" > /dev/null
  $PY dev/evaluate.py "$D/$split.jsonl" "$D/$split.labels.jsonl" "$D/$split.pred.jsonl"
done
