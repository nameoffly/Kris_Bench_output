import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("Missing dependency: openai. Install with `pip install openai`.") from exc


LANG_NAME_MAP = {
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "es": "Spanish",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate ins_en/explain_en fields in KRIS_Bench annotation.json files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="KRIS_Bench",
        help="Root directory containing category subfolders with annotation.json.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["zh", "ar", "es"],
        help="Language codes to generate (e.g., zh ar es).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-08-06",
        help="OpenAI model name to use for translation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of strings per translation batch.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retries per API call.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="translate_cache.json",
        help="Cache file for translations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing output files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logging.warning("Failed to read cache %s: %s", path, exc)
        return {}


def save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def cache_key(lang_code: str, text: str) -> str:
    digest = hashlib.sha256((lang_code + "\n" + text).encode("utf-8")).hexdigest()
    return digest


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


def parse_json_array(text: str) -> Optional[List[str]]:
    cleaned = strip_code_fence(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        data = json.loads(snippet)
        if isinstance(data, list):
            return data
    except Exception:
        return None
    return None


def translate_single(
    client: OpenAI,
    model: str,
    text: str,
    language: str,
    max_retries: int,
) -> str:
    if not text:
        return text
    system = (
        "You are a professional translator. Translate the user text into {language}. "
        "Output only the translated text, no quotes or explanations. "
        "Preserve placeholders like {{instruct}} and {{explanation}} exactly. "
        "Preserve numbers, units, chemical formulas, variable names, and math symbols. "
        "Keep line breaks."
    ).format(language=language)
    user = f"TEXT:\n{text}"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=1024,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = strip_code_fence(content)
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1].strip()
            return content
        except Exception as exc:
            wait = 2 ** attempt
            logging.warning("Translate failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            time.sleep(wait)
    logging.error("Translate failed after retries; leaving original text.")
    return text


def translate_batch(
    client: OpenAI,
    model: str,
    texts: List[str],
    language: str,
    max_retries: int,
) -> Optional[List[str]]:
    system = (
        "You are a professional translator. Translate each string in the provided JSON array "
        "into {language}. Output ONLY a JSON array of translated strings in the same order. "
        "Preserve placeholders like {{instruct}} and {{explanation}} exactly. "
        "Preserve numbers, units, chemical formulas, variable names, and math symbols. "
        "Keep line breaks."
    ).format(language=language)
    user = "JSON array:\n" + json.dumps(texts, ensure_ascii=False)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content or ""
            parsed = parse_json_array(content)
            if parsed is None or len(parsed) != len(texts):
                raise ValueError("Batch translation parse failed or length mismatch")
            return [str(item) for item in parsed]
        except Exception as exc:
            wait = 2 ** attempt
            logging.warning("Batch translate failed (attempt %d/%d): %s", attempt + 1, max_retries, exc)
            time.sleep(wait)
    return None


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_translation_map(
    client: OpenAI,
    model: str,
    texts: List[str],
    lang_code: str,
    language: str,
    cache: Dict[str, str],
    batch_size: int,
    max_retries: int,
) -> Dict[str, str]:
    unique = []
    seen = set()
    for text in texts:
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)

    to_translate = []
    for text in unique:
        key = cache_key(lang_code, text)
        if key not in cache:
            to_translate.append(text)

    for batch in chunk_list(to_translate, batch_size):
        translated = translate_batch(client, model, batch, language, max_retries)
        if translated is None:
            translated = [
                translate_single(client, model, item, language, max_retries) for item in batch
            ]
        for src, dst in zip(batch, translated):
            cache[cache_key(lang_code, src)] = dst

    mapping = {}
    for text in unique:
        key = cache_key(lang_code, text)
        mapping[text] = cache.get(key, text)
    return mapping


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    annotation_paths = sorted(root.glob("*/annotation.json"))
    if not annotation_paths:
        raise SystemExit(f"No annotation.json found under {root}")

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    client = OpenAI(api_key=api_key, base_url="https://api.bltcy.ai/v1")

    for lang_code in args.languages:
        language = LANG_NAME_MAP.get(lang_code, lang_code)
        logging.info("Translating for language: %s (%s)", lang_code, language)

        for ann_path in annotation_paths:
            data = load_json(ann_path)
            out_ins = {}
            out_explain = {}
            out_all = {}

            ins_texts = []
            explain_texts = []
            for entry in data.values():
                ins_texts.append(entry.get("ins_en", ""))
                explain_texts.append(entry.get("explain_en", ""))

            ins_map = build_translation_map(
                client,
                args.model,
                ins_texts,
                lang_code,
                language,
                cache,
                args.batch_size,
                args.max_retries,
            )
            explain_map = build_translation_map(
                client,
                args.model,
                explain_texts,
                lang_code,
                language,
                cache,
                args.batch_size,
                args.max_retries,
            )

            for key, entry in data.items():
                entry_ins = dict(entry)
                entry_explain = dict(entry)
                entry_all = dict(entry)

                if "ins_en" in entry:
                    entry_ins["ins_en"] = ins_map.get(entry.get("ins_en", ""), entry.get("ins_en", ""))
                    entry_all["ins_en"] = entry_ins["ins_en"]
                if "explain_en" in entry:
                    entry_explain["explain_en"] = explain_map.get(
                        entry.get("explain_en", ""), entry.get("explain_en", "")
                    )
                    entry_all["explain_en"] = entry_explain["explain_en"]

                out_ins[key] = entry_ins
                out_explain[key] = entry_explain
                out_all[key] = entry_all

            out_ins_path = ann_path.with_name(f"annotation_{lang_code}_ins.json")
            out_explain_path = ann_path.with_name(f"annotation_{lang_code}_explain.json")
            out_all_path = ann_path.with_name(f"annotation_{lang_code}_all.json")

            if not args.dry_run:
                for out_path, payload in [
                    (out_ins_path, out_ins),
                    (out_explain_path, out_explain),
                    (out_all_path, out_all),
                ]:
                    if out_path.exists() and not args.overwrite:
                        logging.info("Skip existing file: %s", out_path)
                        continue
                    write_json(out_path, payload)
                    logging.info("Wrote %s", out_path)

            save_cache(cache_path, cache)

    logging.info("Done.")


if __name__ == "__main__":
    main()
