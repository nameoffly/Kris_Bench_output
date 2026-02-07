import argparse
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("Missing dependency: openai. Install with `pip install openai`.") from exc


LANG_NAME_MAP = {
    "ko": "Korean",
    "yo": "Yoruba",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate only ins_en fields in KRIS_Bench annotation.json files."
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
        default=["ko", "yo"],
        help="Language codes to generate (e.g., ko yo). Run separately with one code if preferred.",
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
        default="translate_cache_ins.json",
        help="Cache file for translations.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.bltcy.ai/v1",
        help="API base URL.",
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
    return hashlib.sha256((lang_code + "\n" + text).encode("utf-8")).hexdigest()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        if text.endswith("```"):
            text = text[:-3]
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
    try:
        data = json.loads(cleaned[start : end + 1])
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
            logging.warning(
                "Batch translate failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries,
                exc,
            )
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
) -> Tuple[Dict[str, str], int, int]:
    unique: List[str] = []
    seen = set()
    for text in texts:
        if not text or text in seen:
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
                translate_single(client, model, item, language, max_retries)
                for item in batch
            ]
        for src, dst in zip(batch, translated):
            cache[cache_key(lang_code, src)] = dst

    mapping = {}
    for text in unique:
        mapping[text] = cache.get(cache_key(lang_code, text), text)

    return mapping, len(unique), len(to_translate)


def build_ins_output(data: dict, ins_map: Dict[str, str]) -> dict:
    out = {}
    for key, entry in data.items():
        new_entry = dict(entry)
        if "ins_en" in entry:
            src = entry.get("ins_en", "")
            new_entry["ins_en"] = ins_map.get(src, src)
        out[key] = new_entry
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be > 0")

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
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    for lang_code in args.languages:
        language = LANG_NAME_MAP.get(lang_code, lang_code)
        logging.info("Translating ins_en for language: %s (%s)", lang_code, language)

        for ann_path in annotation_paths:
            data = load_json(ann_path)
            ins_texts = [entry.get("ins_en", "") for entry in data.values()]

            ins_map, unique_count, uncached_count = build_translation_map(
                client=client,
                model=args.model,
                texts=ins_texts,
                lang_code=lang_code,
                language=language,
                cache=cache,
                batch_size=args.batch_size,
                max_retries=args.max_retries,
            )
            out_ins = build_ins_output(data, ins_map)
            out_ins_path = ann_path.with_name(f"annotation_{lang_code}_ins.json")

            logging.info(
                "[%s] %s: unique_ins=%d uncached=%d",
                lang_code,
                ann_path.parent.name,
                unique_count,
                uncached_count,
            )

            if not args.dry_run:
                if out_ins_path.exists() and not args.overwrite:
                    logging.info("Skip existing file: %s", out_ins_path)
                else:
                    write_json(out_ins_path, out_ins)
                    logging.info("Wrote %s", out_ins_path)

            save_cache(cache_path, cache)

    logging.info("Done.")


if __name__ == "__main__":
    main()
