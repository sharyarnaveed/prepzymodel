#!/usr/bin/env python3
"""
Walk dataset/ subfolders, read .txt files, split into ~400-character chunks,
and emit JSONL with three training rows per chunk (instruction variants).
Folder prefix: bio* -> biology, che* -> chemistry, phy* -> physics
(e.g. phy1styear, phy2ndyear, bio2ndyear — any direct child of dataset/ matching).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CHUNK_SIZE = 400
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
DEFAULT_OUT = Path(__file__).resolve().parent / "dataset_train.jsonl"


def subject_for_folder(name: str) -> str | None:
    lower = name.lower()
    if lower.startswith("bio"):
        return "biology"
    if lower.startswith("che"):
        return "chemistry"
    if lower.startswith("phy"):
        return "physics"
    return None


def instruction_triples(subject: str) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    if subject == "biology":
        return (
            (
                "Learn and understand the following biology content.",
                "OK",
            ),
            (
                "Summarize the key biological concepts from this text.",
                "Summary of the text.",
            ),
            (
                "Explain the most important concept in simple words.",
                "Explanation.",
            ),
        )
    if subject == "chemistry":
        return (
            (
                "Learn and understand the following chemistry content.",
                "OK",
            ),
            (
                "Summarize the key chemical concepts from this text.",
                "Summary of the text.",
            ),
            (
                "Explain the most important concept in simple words.",
                "Explanation.",
            ),
        )
    if subject == "physics":
        return (
            (
                "Learn and understand the following physics content.",
                "OK",
            ),
            (
                "Summarize the key physics concepts from this text.",
                "Summary of the text.",
            ),
            (
                "Explain the most important concept in simple words.",
                "Explanation.",
            ),
        )
    raise ValueError(f"Unknown subject: {subject}")


def normalize_text(raw: str) -> str:
    # Collapse whitespace to single spaces; strip
    return re.sub(r"\s+", " ", raw).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            space = text.rfind(" ", start, end + 1)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end
        while start < n and text[start] == " ":
            start += 1
    return chunks


def iter_txt_files(root: Path) -> list[tuple[Path, str]]:
    """Return (file_path, subject) for each .txt under recognized subfolders."""
    out: list[tuple[Path, str]] = []
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        subject = subject_for_folder(sub.name)
        if subject is None:
            continue
        for p in sorted(sub.rglob("*.txt")):
            if p.is_file():
                out.append((p, subject))
    return out


def rows_for_file(path: Path, subject: str) -> list[dict[str, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = normalize_text(raw)
    triples = instruction_triples(subject)
    rows: list[dict[str, str]] = []
    for chunk in chunk_text(body):
        for instruction, output in triples:
            rows.append(
                {
                    "instruction": instruction,
                    "input": chunk,
                    "output": output,
                    "subject": subject,
                }
            )
    return rows


def main() -> None:
    pairs = iter_txt_files(DATASET_DIR)
    all_rows: list[dict[str, str]] = []
    for path, subject in pairs:
        all_rows.extend(rows_for_file(path, subject))

    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUT.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_rows)} lines to {DEFAULT_OUT} from {len(pairs)} text files.")


if __name__ == "__main__":
    main()
