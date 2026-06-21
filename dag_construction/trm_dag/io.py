from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from pathlib import Path
from typing import Any, TextIO


def ensure_parent(path: str | os.PathLike[str]) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def _valid_jsonl_prefix(raw: bytes, path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], bytes]:
    parts = raw.split(b"\n")
    had_trailing_newline = raw.endswith(b"\n")
    while parts and parts[-1].strip() == b"":
        parts.pop()
    rows: list[dict[str, Any]] = []
    valid_chunks: list[bytes] = []
    for idx, chunk in enumerate(parts):
        if not chunk:
            continue
        is_last = idx == len(parts) - 1
        try:
            text = chunk.decode("utf-8")
            obj = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if is_last:
                warnings.warn(f"ignored invalid last JSONL line in {path}", stacklevel=3)
                break
            raise
        if not isinstance(obj, dict):
            raise TypeError(f"JSONL line {idx + 1} in {path} is not an object")
        rows.append(obj)
        valid_chunks.append(chunk)
    if not valid_chunks:
        return rows, b""
    suffix = b"\n" if had_trailing_newline or len(valid_chunks) != len(parts) else b"\n"
    return rows, b"\n".join(valid_chunks) + suffix


def tolerant_load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load JSONL, ignoring a broken or half-written final line only."""
    if not Path(path).exists():
        return []
    rows, _valid_prefix = _valid_jsonl_prefix(Path(path).read_bytes(), path)
    return rows


def truncate_invalid_jsonl_tail(path: str | os.PathLike[str]) -> int:
    """Drop a broken final JSONL line and return the number of valid rows kept."""
    target = Path(path)
    if not target.exists():
        return 0
    rows, valid_prefix = _valid_jsonl_prefix(target.read_bytes(), path)
    target.write_bytes(valid_prefix)
    return len(rows)


def _safe_json_like(value: Any) -> Any:
    if isinstance(value, str) and value.strip()[:1] in {"{", "["}:
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def record_key(rec: dict[str, Any]) -> str:
    schema_payload = {
        "prompt": rec.get("prompt", ""),
        "steps": rec.get("steps"),
        "trace": rec.get("trace"),
        "fillers": rec.get("fillers"),
    }
    schema_hash = _canonical_hash(schema_payload)
    ds = rec.get("data_source", "<NA>")
    extra = rec.get("extra_info", {})
    if not isinstance(extra, dict):
        extra = _safe_json_like(extra)
    if not isinstance(extra, dict):
        extra = {}
    split_v = extra.get("split")
    idx_v = extra.get("index")
    raw_ct = extra.get("raw_code_tests", {})
    subset = raw_ct.get("subset", "") if isinstance(raw_ct, dict) else ""
    return f"{ds}::{split_v}::{subset}::{idx_v}::{schema_hash}"


def compute_resume_start_and_check_order(
    src_rows: list[dict[str, Any]], out_path: str | os.PathLike[str]
) -> int:
    truncate_invalid_jsonl_tail(out_path)
    done_rows = tolerant_load_jsonl(out_path)
    if not done_rows:
        return 0
    if len(done_rows) > len(src_rows):
        raise RuntimeError(
            f"output has more rows ({len(done_rows)}) than input ({len(src_rows)})"
        )
    for idx, saved in enumerate(done_rows):
        src_key = record_key(src_rows[idx])
        out_key = record_key(saved)
        if src_key != out_key:
            raise RuntimeError(f"resume key mismatch at line {idx}: out={out_key} vs src={src_key}")
    return len(done_rows)


class OrderedBufferedWriter:
    """Thread-safe ordered JSONL writer with append/resume support."""

    def __init__(self, out_path: str | os.PathLike[str], total_items: int, start_idx: int) -> None:
        self.out_path = str(out_path)
        ensure_parent(self.out_path)
        self.fp: TextIO = open(self.out_path, "a", encoding="utf-8")  # noqa: SIM115
        self.lock = threading.Lock()
        self.pending: dict[int, dict[str, Any]] = {}
        self.next_to_write = int(start_idx)
        self.total_items = int(total_items)
        self.saved_flags = [idx < start_idx for idx in range(total_items)]

    def offer(self, idx: int, obj: dict[str, Any]) -> None:
        with self.lock:
            self.pending[int(idx)] = obj
            if int(idx) != self.next_to_write:
                return
            lines: list[str] = []
            while self.next_to_write in self.pending:
                cur = self.pending.pop(self.next_to_write)
                lines.append(json.dumps(cur, ensure_ascii=False) + "\n")
                self.saved_flags[self.next_to_write] = True
                self.next_to_write += 1
            if lines:
                self.fp.write("".join(lines))
                self.fp.flush()

    def is_saved(self, idx: int) -> bool:
        return 0 <= idx < self.total_items and self.saved_flags[idx]

    def close(self) -> None:
        self.fp.close()

    def __enter__(self) -> OrderedBufferedWriter:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return tolerant_load_jsonl(path)


def write_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
