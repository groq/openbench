#!/bin/bash
source .venv/bin/activate

MODELS=(
  "groq/openai/gpt-oss-120b"
  "groq/minimaxai/minimax-m2"
  "groq/moonshotai/kimi-k2-instruct-0905"
)

STRATEGIES=(
  "minimal-tools"
  "minimal-servers"
  "directory"
  "copilot"
  "distraction-128"
)

for model in "${MODELS[@]}"; do
  echo "===== MODEL: $model ====="
  for strategy in "${STRATEGIES[@]}"; do
    echo "--- Strategy: $strategy ---"
    bench eval progressivemcpbench --model "$model" --alpha --epochs 10 --epochs-reducer mean -T strategy="$strategy" --limit 5 2>&1 | grep -E "(accuracy|tokens \[I:)" | head -2
  done
done
