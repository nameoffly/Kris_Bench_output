# Repository Guidelines

## Project Structure & Module Organization
This repo is a Python benchmark/evaluation toolkit for KRIS-Bench.

- `KRIS_Bench/<category>/`: source benchmark data (images + `annotation*.json`).
- `metrics_common.py`, `metrics_knowledge.py`, `metrics_view_change.py`, `metrics_multi_element.py`, `metrics_temporal_prediction.py`: evaluation entry points.
- `generate_results.py`: batch-generate edited images into `results/<model>/<category>/<lang>/`.
- `translate_annotations.py`: translate annotation fields and update language JSON files.
- `utils/prompts.py`: shared judge/evaluation prompts.
- `assets/`: figures for docs/README.
- `results/`, `output_bagel/`: generated outputs and model artifacts.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create local environment.
- `pip install openai pillow tqdm`: install runtime dependencies used by scripts.
- `export OPENAI_API_KEY=...`: required for generation/evaluation scripts.
- `python generate_results.py --model-name gpt-image-1 --langs zh es ar --limit 10 --max-workers 5`: generate sample outputs.
- `python metrics_common.py --models doubao gpt gemini`: run full multi-category evaluation.
- `python metrics_knowledge.py --models gpt`: run knowledge-focused scoring.
- `python translate_annotations.py --languages zh ar es --batch-size 10`: refresh translated annotations.

## Coding Style & Naming Conventions
- Target Python 3.8+; use 4-space indentation and PEP 8-style formatting.
- Use `snake_case` for functions/variables and `UPPER_CASE` for constants (for example, `BENCH_DIR`).
- Keep category names and image IDs aligned with on-disk paths and annotation keys.
- Prefer small, focused script changes; preserve existing CLI argument patterns (`argparse`).

## Testing Guidelines
There is currently no dedicated `tests/` suite. Use script-level validation:

- Run a small generation smoke test (`generate_results.py --limit 10`).
- Run at least one metric script against produced images.
- Confirm output files exist and are parseable JSON (for example `results/<model>/<category>/metrics.json`).

## Commit & Pull Request Guidelines
- Existing history favors short imperative messages (for example `fix`, `update readme`, `delete .DS_Store`).
- Prefer clearer messages in the same style: `fix metrics parsing`, `update README usage`.
- PRs should include: purpose, changed scripts/categories, commands run, and representative output snippets or metric diffs.

## Security & Configuration Tips
- Do not commit API keys or secret endpoints; use environment variables.
- Large generated artifacts in `results/` and `output_bagel/` should be committed only when intentionally updating published outputs.
