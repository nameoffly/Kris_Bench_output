import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError
except Exception:  # pragma: no cover
    HfApi = None

    class HfHubHTTPError(Exception):
        pass


DEFAULT_LOCAL_FILE = "/home/hisheep/d/Kris_Bench/output_bagel.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local zip file to a Hugging Face repository."
    )
    parser.add_argument(
        "--local-file",
        type=str,
        default=DEFAULT_LOCAL_FILE,
        help=f"Local zip file path (default: {DEFAULT_LOCAL_FILE})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target repository id, e.g. yourname/kris-bench-artifacts",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        required=True,
        help="Destination path in repo, e.g. bagel/output_bagel.zip",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Repository type (default: dataset).",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload output_bagel.zip",
        help="Commit message for upload.",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create repository if it does not exist.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repository as private when --create-repo is set.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token. If omitted, use HF_TOKEN from environment.",
    )
    return parser.parse_args()


def resolve_token(cli_token: Optional[str]) -> str:
    token = cli_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Missing Hugging Face token. Set HF_TOKEN or pass --token.")
    return token


def validate_local_file(path: str) -> None:
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Local file does not exist: {path}")
    if not fp.is_file():
        raise ValueError(f"Local path is not a file: {path}")
    if fp.stat().st_size == 0:
        raise ValueError(f"Local file is empty: {path}")


def ensure_repo_exists(
    api,
    repo_id: str,
    repo_type: str,
    create_repo: bool,
    private: bool,
    token: str,
) -> None:
    if create_repo:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            private=private,
            exist_ok=True,
        )
        return

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Repository '{repo_id}' does not exist or cannot be accessed. "
            "Use --create-repo to create it."
        ) from exc


def upload_zip(
    api,
    local_file: str,
    repo_id: str,
    path_in_repo: str,
    repo_type: str,
    commit_message: str,
    token: str,
) -> str:
    return api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
        commit_message=commit_message,
    )


def main() -> int:
    args = parse_args()

    try:
        validate_local_file(args.local_file)
        token = resolve_token(args.token)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if HfApi is None:
        print(
            "[ERROR] Missing dependency: huggingface_hub. Install with `pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 2

    local_size = Path(args.local_file).stat().st_size
    print(f"[INFO] Local file: {args.local_file}")
    print(f"[INFO] Size: {local_size} bytes")
    print(f"[INFO] Target: {args.repo_type}:{args.repo_id}/{args.path_in_repo}")

    api = HfApi()
    try:
        ensure_repo_exists(
            api=api,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            create_repo=args.create_repo,
            private=args.private,
            token=token,
        )
        url = upload_zip(
            api=api,
            local_file=args.local_file,
            repo_id=args.repo_id,
            path_in_repo=args.path_in_repo,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            token=token,
        )
    except HfHubHTTPError as exc:
        print(f"[ERROR] Hugging Face API error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] Upload completed: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
