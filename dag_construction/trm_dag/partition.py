from __future__ import annotations

import re

DEFAULT_KEYWORDS = ("Step", "Then", "Next", "Finally")


def _split_with_seps_paragraph(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(\n{2,})", text)
    return [(parts[i], parts[i + 1] if i + 1 < len(parts) else "") for i in range(0, len(parts), 2)]


def _first_word(seg: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9_]+)", seg)
    return match.group(1) if match else None


def _split_sep_dot_rule(sep: str) -> tuple[str, str]:
    if not sep:
        return "", ""
    return (".", sep[1:]) if sep.startswith(".") else ("", sep)


def partition_keyword(
    trace: str,
    *,
    keywords: tuple[str, ...] | list[str] = DEFAULT_KEYWORDS,
) -> tuple[list[str], list[str]]:
    """Split a reasoning trace into paragraph-level keyword-led steps."""
    text = trace if isinstance(trace, str) else str(trace)
    keywords_set = {kw.lower() for kw in keywords}
    chunks = [
        {"seg": seg, "sep": sep}
        for seg, sep in _split_with_seps_paragraph(text)
    ]
    if not chunks:
        return [""], []

    def is_step_idx(idx: int) -> bool:
        first = _first_word(chunks[idx].get("seg", ""))
        return first is not None and first.lower() in keywords_set

    step_idxs = [idx for idx in range(len(chunks)) if is_step_idx(idx)]
    if not step_idxs:
        full = "".join((chunk.get("seg", "") or "") + (chunk.get("sep", "") or "") for chunk in chunks)
        return [full], []

    steps: list[str] = []
    fillers: list[str] = []
    first_idx = step_idxs[0]
    prefix = "".join(
        (chunks[idx].get("seg", "") or "") + (chunks[idx].get("sep", "") or "")
        for idx in range(first_idx)
    )
    steps.append(prefix + (chunks[first_idx].get("seg", "") or ""))

    for pos in range(1, len(step_idxs)):
        prev_idx = step_idxs[pos - 1]
        cur_idx = step_idxs[pos]
        dot, whitespace = _split_sep_dot_rule(chunks[prev_idx].get("sep", "") or "")
        if dot:
            steps[-1] += dot
        fillers.append(whitespace)
        between = "".join(
            (chunks[idx].get("seg", "") or "") + (chunks[idx].get("sep", "") or "")
            for idx in range(prev_idx + 1, cur_idx)
        )
        if between:
            steps[-1] += between
        steps.append(chunks[cur_idx].get("seg", "") or "")

    last_idx = step_idxs[-1]
    dot_last, whitespace_last = _split_sep_dot_rule(chunks[last_idx].get("sep", "") or "")
    if dot_last:
        steps[-1] += dot_last
    tail_rest = whitespace_last + "".join(
        (chunks[idx].get("seg", "") or "") + (chunks[idx].get("sep", "") or "")
        for idx in range(last_idx + 1, len(chunks))
    )
    steps[-1] += tail_rest
    if len(steps) != len(fillers) + 1:
        full = "".join((chunk.get("seg", "") or "") + (chunk.get("sep", "") or "") for chunk in chunks)
        return [full], []
    return steps, fillers
