# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KRIS-Bench (Knowledge-based Reasoning in Image-editing Systems Benchmark) is a diagnostic benchmark for evaluating instruction-based image editing models. Accepted at NeurIPS 2025 (arXiv: 2505.16707). It covers 3 knowledge types, 7 reasoning dimensions, 22 tasks, and 1,267 annotated instances. Evaluation uses GPT-4o as an automated judge scoring on a 1-5 scale.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install openai pillow tqdm
export OPENAI_API_KEY=...
```

Python 3.8+ required. No requirements.txt exists; dependencies are `openai`, `Pillow`, `tqdm`.

## Common Commands

```bash
# Generate edited images
python generate_results.py --model-name gpt-image-1 --langs zh es ar --limit 10 --max-workers 5

# Run evaluations (each script covers different category groups)
python metrics_common.py --models doubao gpt gemini
python metrics_knowledge.py --models gpt
python metrics_multi_element.py --models gpt
python metrics_temporal_prediction.py --models gpt
python metrics_view_change.py --models gpt

# Translate annotations
python translate_annotations.py --languages zh ar es --batch-size 10   # ins + explain
python translate_ins_only.py --languages ko yo                          # ins only

# Merge translated annotations into final_data files
python generate_final_data.py --languages ko yo --variant ins

# Run tests
python -m unittest tests/test_generate_final_data.py
python -m unittest tests/test_translate_ins_only.py
```

## Architecture

### Evaluation Pipeline

Five metrics scripts each cover different category groups, all following the same pattern:
1. Parse `--models` CLI arg, iterate over models × categories
2. Load `annotation.json`, encode images to base64, call GPT-4o with prompts from `utils/prompts.py`
3. Extract scores via JSON parsing with regex fallback
4. Save per-category `metrics.json` under `results/<model>/<category>/`

| Script | Categories | Metrics |
|--------|-----------|---------|
| `metrics_common.py` | count_change, color_change, anomaly_correction, position_movement, size_adjustment, part_completion, multi-instruction_execution | consistency, instruction_following, image_quality |
| `metrics_knowledge.py` | abstract_reasoning, mathematics, practical_knowledge, medicine, rule-based_reasoning, biology, geography, chemistry, humanities, physics | consistency, instruction_following, knowledge_plausibility, image_quality |
| `metrics_multi_element.py` | multi-element_composition | consistency, instruction_following, image_quality (3 ref images) |
| `metrics_temporal_prediction.py` | temporal_prediction | consistency, instruction_following, image_quality (3 ref frames) |
| `metrics_view_change.py` | viewpoint_change | consistency, instruction_following, image_quality (uses ground-truth image) |

`metrics_common.py` exports `extract_score_and_reason()` which is reused by multi_element, temporal_prediction, and view_change. `metrics_knowledge.py` has its own extractors (`extract_dual_scores`, `extract_consistency_score`, `extract_quality_score`) and does NOT reuse from metrics_common.

### Annotation Format

Each `KRIS_Bench/<category>/annotation.json`:
```json
{
  "<image_id>": {
    "ori_img": "filename.jpg",      // string or list (multi-element/temporal)
    "gt_img": "",                    // ground-truth (viewpoint_change only)
    "ins_en": "editing instruction",
    "explain_en": "knowledge explanation"  // knowledge categories only
  }
}
```

Translated files: `annotation_{lang}_{variant}.json` where lang is `zh|ar|es|ko|yo` and variant is `ins|explain|all`.

### Prompts

`utils/prompts.py` contains all GPT-4o judge prompt templates. They use `str.format()` with `{instruct}` and `{explanation}` placeholders, and double braces `{{`/`}}` for literal JSON in output format specifications.

## Coding Conventions

- PEP 8, 4-space indentation, `snake_case` for functions/variables, `UPPER_CASE` for constants
- Preserve existing `argparse` CLI patterns when modifying scripts
- Keep category names and image IDs aligned with on-disk paths and annotation keys
- Commit messages: short imperative style (e.g., `fix metrics parsing`, `update README usage`)
