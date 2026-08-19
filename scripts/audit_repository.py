#!/usr/bin/env python3
"""Fail if a publication tree contains private paths, data, or large binaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


MAX_GIT_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".h5ad",
    ".loom",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
}
SKIP_PARTS = {".git", ".pytest_cache", ".venv", "__pycache__", "build", "dist"}

# Build tokens in pieces so this auditor does not contain the strings it bans.
FORBIDDEN_TEXT = {
    "/" + "beacon" + "-projects" + "/": "cluster project path",
    "/" + "beacon" + "-homes" + "/": "cluster home path",
    "/" + "scratch" + "1" + "/": "cluster scratch path",
    "ih" + "ccs": "cluster hostname",
    "BEGIN " + "OPENSSH PRIVATE KEY": "private SSH key",
    "BEGIN " + "RSA PRIVATE KEY": "private RSA key",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def audit(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or SKIP_PARTS.intersection(path.relative_to(root).parts):
            continue
        relative = path.relative_to(root)
        size = path.stat().st_size
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden binary/data suffix: {relative}")
        if size > MAX_GIT_FILE_BYTES:
            errors.append(f"file exceeds 10 MiB Git limit: {relative} ({size} bytes)")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for token, label in FORBIDDEN_TEXT.items():
            if token.lower() in text.lower():
                errors.append(f"{label} in {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} in {relative}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".", type=Path)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    errors = audit(root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"PASS: publication audit ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
