"""Upload taxonomy annotations to the Hugging Face Hub.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login   # or set HF_TOKEN env var

Usage:
    python hf_dataset/upload.py --repo ganghyunnnn/rs-taxonomy-labels
    python hf_dataset/upload.py --repo ganghyunnnn/rs-taxonomy-labels --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_DIR = REPO_ROOT / "annotation"
HF_DIR = REPO_ROOT / "hf_dataset"

# Files to publish. Python source files are intentionally excluded — the
# code lives in the companion GitHub repo.
INCLUDE_PATTERNS = ("*.json", "*.csv", "*.md")
EXCLUDE_NAMES = {"compute_iaa.py", "generate_iaa_csv.py", "iaa_measurement.py"}


def collect_files() -> list[tuple[Path, str]]:
    """Return (local_path, repo_path) pairs to upload."""
    pairs: list[tuple[Path, str]] = []

    for pattern in INCLUDE_PATTERNS:
        for src in sorted(ANNOTATION_DIR.glob(pattern)):
            if src.name in EXCLUDE_NAMES:
                continue
            pairs.append((src, src.name))

    pairs.append((HF_DIR / "README.md", "README.md"))
    pairs.append((REPO_ROOT / "NOTICE", "NOTICE"))
    pairs.append((REPO_ROOT / "LICENSE", "LICENSE"))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="HF dataset repo id, e.g. user/name")
    parser.add_argument("--private", action="store_true", help="Create as private")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    parser.add_argument("--message", default="Upload taxonomy annotations", help="Commit message")
    args = parser.parse_args()

    pairs = collect_files()

    print(f"Repo: {args.repo}")
    print(f"Files to upload ({len(pairs)}):")
    total_bytes = 0
    for local, repo_path in pairs:
        size = local.stat().st_size
        total_bytes += size
        print(f"  {repo_path:<55} {size:>10,} B")
    print(f"Total: {total_bytes:,} bytes")

    if args.dry_run:
        print("[dry-run] no upload performed")
        return 0

    api = HfApi()
    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)

    for local, repo_path in pairs:
        print(f"  uploading {repo_path} ...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=repo_path,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=args.message,
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
