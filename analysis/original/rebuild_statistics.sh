#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
KG_PYTHON="${KG_PYTHON:-python3}"
mkdir -p rebuilt logs/rebuild
for budget in 128 256 512; do
 "$KG_PYTHON" src/analyze_paired_results.py --input "results/qwen3_awq_test300_B${budget}.json" \
 --seed 20260818 --bootstrap-iterations 10000 --permutation-iterations 50000 \
 --output-json "rebuilt/primary_B${budget}.json" --output-csv "rebuilt/primary_B${budget}.csv" \
 > "logs/rebuild/primary_B${budget}.log" 2>&1
done
"$KG_PYTHON" src/analyze_component_ablation.py \
  --primary-results results/qwen3_awq_test300_B256.json \
  --ablation-results results/clean_component_ablation_qwen_B256.json \
  --seed 20260823 --bootstrap-iterations 10000 --permutation-iterations 50000 \
  --output-json rebuilt/clean_components.json --output-csv rebuilt/clean_components.csv \
  > logs/rebuild/clean_components.log 2>&1
"$KG_PYTHON" src/analyze_template_clusters.py \
  --results results/qwen3_awq_test300_B128.json results/qwen3_awq_test300_B256.json results/qwen3_awq_test300_B512.json \
  --seed 20260823 --bootstrap-iterations 20000 --permutation-iterations 100000 \
  --output-json rebuilt/template_clusters.json --output-csv rebuilt/template_clusters.csv \
  > logs/rebuild/template_clusters.log 2>&1
"$KG_PYTHON" src/analyze_webqsp_replication.py \
  --input results/week2_webqsp_qwen_test500_B256.json \
  --seed 20260823 --bootstrap-iterations 10000 --permutation-iterations 50000 \
  --output-json rebuilt/legacy_webqsp.json --output-csv rebuilt/legacy_webqsp.csv \
  > logs/rebuild/legacy_webqsp.log 2>&1
# The legacy WebQSP analyzer reproduces its historical Direct metric.
# Corrected scoring is a separate, immutable derivation; see analyze_revision.py.
"$KG_PYTHON" verify_rebuilt_statistics.py --output rebuilt/validation.json
