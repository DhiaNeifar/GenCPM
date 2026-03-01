#!/usr/bin/env python3
"""
Copy only t_preds.yaml files while preserving directory structure.

Example:
  python legacy/copy_yaml.py \
    C:\\source\\scenario_root \
    C:\\target\\scenario_root
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_preds_yaml_preserve_structure(
    source_root: Path,
    target_root: Path,
    suffix: str = "_preds.yaml",
    overwrite: bool = False,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Copy files matching *{suffix} under source_root into target_root, preserving
    full relative paths.

    Returns:
      (copied_count, skipped_count)
    """
    files = sorted([p for p in source_root.rglob(f"*{suffix}") if p.is_file()])
    copied = 0
    skipped = 0

    for src in tqdm(files, desc="Copy preds YAML", unit="file"):
        rel = src.relative_to(source_root)
        dst = target_root / rel

        if dst.exists() and not overwrite:
            skipped += 1
            continue

        if not dry_run:
            copy_file(src, dst)

        copied += 1

    return copied, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy t_preds.yaml files while preserving folder layout."
    )
    parser.add_argument("source_root", type=str, help="Source root folder.")
    parser.add_argument("target_root", type=str, help="Target root folder.")
    parser.add_argument("--suffix", type=str, default="_preds.yaml",
                        help="Suffix to copy (default: _preds.yaml).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite destination files if they exist.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not write files, only count actions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    target_root.mkdir(parents=True, exist_ok=True)

    copied, skipped = copy_preds_yaml_preserve_structure(
        source_root=source_root,
        target_root=target_root,
        suffix=args.suffix,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    mode = "DRY-RUN" if args.dry_run else "DONE"
    print(f"[{mode}] copied={copied} skipped={skipped}")


if __name__ == "__main__":
    main()
