#!/bin/bash
# eval_bagel.sh - Evaluate output_bagel data across languages
# Usage:
#   ./eval_bagel.sh                    # Evaluate all languages
#   ./eval_bagel.sh en zh              # Evaluate English and Chinese only

LANGS="${@:-en zh ar es ko yo}"
MODEL="bagel"
OUTPUT_DIR="eval_results"

for lang in $LANGS; do
    echo "=== Evaluating language: $lang ==="
    RESULTS_DIR="output_bagel/outputs_${lang}"
    COMMON_ARGS="--models $MODEL --results-dir $RESULTS_DIR --output-dir $OUTPUT_DIR --lang $lang"

    python metrics_common.py $COMMON_ARGS
    python metrics_knowledge.py $COMMON_ARGS
    python metrics_multi_element.py $COMMON_ARGS
    python metrics_temporal_prediction.py $COMMON_ARGS
    python metrics_view_change.py $COMMON_ARGS
done

echo "=== Done! Results saved to $OUTPUT_DIR/ ==="
