#!/bin/bash

BATCH_DIR="data/experiments/runs/batch_20260504_200407"

echo "🔍 Waiting for batch to complete..."
echo "Monitoring: $BATCH_DIR"

# Wait for all 6 JSON files to be generated
while true; do
  FILE_COUNT=$(ls -1 "$BATCH_DIR"/*.json 2>/dev/null | wc -l)
  if [ $FILE_COUNT -eq 6 ]; then
    echo "✅ All 6 JSON files generated! Starting evaluation..."
    break
  fi
  echo "⏳ Files generated: $FILE_COUNT/6 ($(date '+%H:%M:%S'))"
  sleep 30
done

echo ""
echo "📊 Running llm_judge.py..."
python -m src.eval.llm_judge --experiment-dir "$BATCH_DIR"

echo ""
echo "📈 Running metrics.py..."
python -m src.eval.metrics --experiment-dir "$BATCH_DIR"

echo ""
echo "✨ Evaluation complete!"
