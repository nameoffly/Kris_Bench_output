import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge KRIS_Bench per-category translated annotation files into final_data_<lang>.json."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="KRIS_Bench",
        help="Root directory containing category subfolders.",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["ko", "yo"],
        help="Language codes to export (e.g., ko yo).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="ins",
        choices=["ins", "explain", "all"],
        help="Annotation variant suffix to merge.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where final_data_<lang>.json files are written.",
    )
    parser.add_argument(
        "--output-template",
        type=str,
        default="final_data_{lang}.json",
        help="Output filename template. Must contain {lang}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Skip categories missing annotation_<lang>_<variant>.json instead of failing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sorted_sample_ids(keys: List[str]) -> List[str]:
    def _sort_key(x: str):
        return (0, int(x)) if str(x).isdigit() else (1, str(x))

    return sorted([str(k) for k in keys], key=_sort_key)


def build_final_records_for_category(category: str, annotation: Dict[str, dict]) -> List[dict]:
    out: List[dict] = []
    for sample_id in sorted_sample_ids(list(annotation.keys())):
        record = {"id": str(sample_id), "type": category}
        record.update(annotation[sample_id])
        out.append(record)
    return out


def generate_final_data_for_language(
    root: Path,
    language: str,
    variant: str,
    strict: bool = True,
) -> List[dict]:
    if not root.exists():
        raise FileNotFoundError(f"Root path not found: {root}")

    categories = sorted([p for p in root.iterdir() if p.is_dir()])
    all_records: List[dict] = []

    for category_dir in categories:
        ann_path = category_dir / f"annotation_{language}_{variant}.json"
        if not ann_path.exists():
            if strict:
                raise FileNotFoundError(f"Missing translated annotation file: {ann_path}")
            continue

        data = load_json(ann_path)
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict JSON at {ann_path}, got {type(data).__name__}")

        all_records.extend(build_final_records_for_category(category_dir.name, data))

    return all_records


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "{lang}" not in args.output_template:
        raise SystemExit("--output-template must include '{lang}' placeholder.")

    strict = not args.non_strict

    for language in args.languages:
        records = generate_final_data_for_language(
            root=root,
            language=language,
            variant=args.variant,
            strict=strict,
        )
        output_name = args.output_template.format(lang=language)
        output_path = output_dir / output_name

        if output_path.exists() and not args.overwrite:
            raise SystemExit(
                f"Output exists: {output_path}. Use --overwrite to replace."
            )

        write_json(output_path, records)
        print(f"Wrote {output_path} ({len(records)} samples)")


if __name__ == "__main__":
    main()
