import argparse
import base64
import json
import logging
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Iterable

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


BENCH_DIR = "KRIS_Bench"
RESULTS_DIR = "results"


def setup_logger() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def list_categories(base_dir: str) -> list[str]:
    categories = []
    if not os.path.isdir(base_dir):
        return categories
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        ann = os.path.join(path, "annotation.json")
        if os.path.isdir(path) and os.path.isfile(ann):
            categories.append(name)
    return sorted(categories)


def load_lang_annotations(category: str, lang: str) -> dict:
    ann_path = os.path.join(BENCH_DIR, category, f"annotation_{lang}_ins.json")
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def sort_image_ids(keys: Iterable[str]) -> list[str]:
    def _key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)
    return sorted(keys, key=_key)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_size(size_str: str) -> tuple[int, int]:
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: {size_str}")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid size value: {size_str}")
    return w, h


def choose_target_size(width: int, height: int) -> tuple[int, int]:
    ratio = width / height if height else 1.0
    if ratio > 1.1:
        return 1536, 1024
    if ratio < 0.9:
        return 1024, 1536
    return 1024, 1024


def pad_to_canvas(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    img = img.convert("RGBA")
    src_w, src_h = img.size
    scale = min(target_w / src_w, target_h / src_h, 1.0)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    if (new_w, new_h) != (src_w, src_h):
        img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(img, offset, img)
    return canvas


def preprocess_image_bytes(path: str, target_w: int, target_h: int) -> bytes:
    with Image.open(path) as img:
        canvas = pad_to_canvas(img, target_w, target_h)
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def extract_image_bytes_from_response(resp) -> bytes:
    data_list = None
    if hasattr(resp, "data"):
        data_list = resp.data
    elif isinstance(resp, dict):
        data_list = resp.get("data")

    if not data_list:
        raise ValueError("No image data in response.")

    item = data_list[0]
    b64 = getattr(item, "b64_json", None)
    if b64 is None and isinstance(item, dict):
        b64 = item.get("b64_json")
    if b64:
        return base64.b64decode(b64)

    url = getattr(item, "url", None)
    if url is None and isinstance(item, dict):
        url = item.get("url")
    if url:
        with urllib.request.urlopen(url) as resp_url:
            return resp_url.read()

    raise ValueError("No b64_json or url in response.")


def save_jpeg(image_bytes: bytes, out_path: str) -> None:
    img = Image.open(BytesIO(image_bytes))
    img = img.convert("RGB")
    img.save(out_path, format="JPEG", quality=95)


def call_openai_edit(
    client: OpenAI,
    model: str,
    prompt: str,
    image_bytes_list: list[bytes],
    response_format: str,
    size: str | None,
    quality: str | None,
    seed: int | None,
    max_retries: int,
) -> bytes:
    kwargs = {
        "model": model,
        "prompt": prompt,
        "image": image_bytes_list,
        "response_format": response_format,
    }
    if size:
        kwargs["size"] = size
    if quality:
        kwargs["quality"] = quality
    if seed is not None:
        kwargs["seed"] = seed

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if hasattr(client.images, "edit"):
                resp = client.images.edit(**kwargs)
            elif hasattr(client.images, "edits"):
                resp = client.images.edits(**kwargs)
            else:
                raise AttributeError("No images.edit(s) method available.")
            return extract_image_bytes_from_response(resp)
        except Exception as e:
            last_err = e
            logging.warning("images.edit(s) failed (attempt %d/%d): %s", attempt, max_retries, e)
            time.sleep(min(2 ** (attempt - 1), 8))

    logging.error("images.edit(s) failed after retries: %s", last_err)
    raise last_err or RuntimeError("images.edit(s) failed after retries")


def process_one(
    client: OpenAI,
    category: str,
    lang: str,
    image_id: str,
    entry: dict,
    args,
    override_size: tuple[int, int] | None,
) -> tuple[str, bool, str | None]:
    ori = entry.get("ori_img")
    if isinstance(ori, list):
        rel_paths = ori
    else:
        rel_paths = [ori]

    image_paths = [os.path.join(BENCH_DIR, category, p) for p in rel_paths if p]
    for p in image_paths:
        if not os.path.isfile(p):
            return image_id, False, f"missing input image: {p}"

    out_dir = os.path.join(RESULTS_DIR, args.model_name, category, lang)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{image_id}.jpg")
    if os.path.isfile(out_path) and not args.overwrite:
        return image_id, True, None

    prompt = entry.get("ins_en", "")
    if not prompt:
        return image_id, False, "missing ins_en"

    is_multi = len(image_paths) > 1
    if override_size:
        target_w, target_h = override_size
    elif is_multi:
        target_w, target_h = 1024, 1024
    else:
        with Image.open(image_paths[0]) as img:
            target_w, target_h = choose_target_size(img.width, img.height)

    size_str = f"{target_w}x{target_h}"
    logging.info("Using size %s for %s/%s/%s", size_str, category, lang, image_id)
    image_bytes_list = [preprocess_image_bytes(p, target_w, target_h) for p in image_paths]
    result_bytes = call_openai_edit(
        client=client,
        model=args.openai_model,
        prompt=prompt,
        image_bytes_list=image_bytes_list,
        response_format=args.response_format,
        size=size_str,
        quality=args.quality,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    save_jpeg(result_bytes, out_path)
    return image_id, True, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate edited images for KRIS_Bench.")
    parser.add_argument("--model-name", type=str, required=True, help="Folder name under results/")
    parser.add_argument("--openai-model", type=str, default="gpt-image-1", help="OpenAI image model")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="Category names")
    parser.add_argument("--langs", type=str, nargs="+", default=["zh", "es", "ar"], help="Language codes")
    parser.add_argument("--limit", type=int, default=None, help="Max samples per category")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images")
    parser.add_argument("--response-format", type=str, default="b64_json", choices=["b64_json", "url"])
    parser.add_argument("--size", type=str, default="auto", help="Image size, e.g. 1024x1024")
    parser.add_argument("--quality", type=str, default=None, help="Image quality, e.g. high")
    parser.add_argument("--seed", type=int, default=None, help="Random seed if supported")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries for API calls")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI base URL (optional)")
    args = parser.parse_args()

    setup_logger()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    categories = args.categories or list_categories(BENCH_DIR)
    langs = [l.strip().lower() for l in args.langs if l.strip()]
    override_size = None
    if args.size and args.size != "auto":
        override_size = parse_size(args.size)
    if not categories:
        raise RuntimeError("No categories found under KRIS_Bench.")
    if not langs:
        raise RuntimeError("No languages specified.")

    total_ok = 0
    total_fail = 0
    for category in tqdm(categories, desc="Categories"):
        for lang in langs:
            try:
                annotations = load_lang_annotations(category, lang)
            except Exception as e:
                logging.error("Failed to load annotations for %s/%s: %s", category, lang, e)
                continue

            image_ids = sort_image_ids(list(annotations.keys()))
            if args.limit is not None:
                image_ids = image_ids[: args.limit]

            logging.info("Category %s/%s: %d samples", category, lang, len(image_ids))
            failed_ids = []

            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {
                    executor.submit(
                        process_one,
                        client,
                        category,
                        lang,
                        img_id,
                        annotations[img_id],
                        args,
                        override_size,
                    ): img_id
                    for img_id in image_ids
                }
                with tqdm(total=len(futures), desc=f"{category}/{lang}", leave=False) as pbar:
                    for fut in as_completed(futures):
                        img_id = futures[fut]
                        try:
                            _, ok, err = fut.result()
                        except Exception as e:
                            ok, err = False, str(e)
                        if ok:
                            total_ok += 1
                        else:
                            total_fail += 1
                            failed_ids.append((img_id, err))
                            logging.error("Failed %s/%s/%s: %s", category, lang, img_id, err)
                        pbar.update(1)

            if failed_ids:
                logging.info("Category %s/%s failed %d samples", category, lang, len(failed_ids))

    logging.info("Done. Success: %d, Failed: %d", total_ok, total_fail)


if __name__ == "__main__":
    main()
