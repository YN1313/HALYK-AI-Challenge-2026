#!/usr/bin/env bash
# Единственная команда для боевого прогона.
#   ./run.sh <путь-к-датасету> [выходной-файл]
set -euo pipefail

DATASET="${1:?укажите путь к распакованному датасету}"
OUT="${2:-submission.json}"

TEAM="${TEAM:-}"
EMAIL="${EMAIL:-}"
MODEL="${MODEL:-claude-sonnet-5}"

echo "── предполётная проверка ──"
python3 -m src.preflight || { echo "остановлено: окружение не готово"; exit 1; }

echo
echo "── прогон ──"
time python3 -m src.run --dataset "$DATASET" --out "$OUT" --cache cache \
     --team "$TEAM" --email "$EMAIL" --model "$MODEL"

echo
echo "── предупреждения агента ──"
cat "$OUT.log"

echo
echo "── проверка структуры ──"
python3 -m src.validate "$OUT" "$DATASET/submission_template.json"

echo
echo "Готово: $OUT"
