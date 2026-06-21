from __future__ import annotations

import random
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepBlock:
    action: str
    prev_list: list[int]
    explanation: str
    raw_text: str


ACTION_RE = re.compile(r"(?:<\|action\|>|\|action\|\s*:)\s*(\w+)", re.IGNORECASE)
PREV_RE = re.compile(r"(?:<\|previous\|>|\|previous\|\s*:)\s*([\d,\ssS]+)", re.IGNORECASE)


def _format_step_block(ids: list[int], steps: list[str]) -> str:
    return "\n".join(f"s{idx}: {steps[idx]}" for idx in ids if 0 <= idx < len(steps))


def render_attachment_pool(
    steps: list[str],
    current_id: int,
    main_path: list[int],
    leaves: list[int],
    *,
    main_path_cap: int,
    other_leaf_cap: int,
) -> str:
    """Render the simplified attachment pool used by the paper implementation."""
    main_ids = main_path[-main_path_cap:] if main_path else []
    other_ids = leaves[-other_leaf_cap:] if leaves else []
    blocks = ["MAIN_PATH:", _format_step_block(main_ids, steps)]
    blocks.extend(["", "CURRENT_STEP:", _format_step_block([current_id], steps)])
    blocks.extend(["", "OTHER_OPEN_BRANCH_LEAVES:"])
    blocks.append(_format_step_block(other_ids, steps) if other_ids else "(none)")
    return "\n".join(blocks)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if (text.startswith("```") and text.endswith("```")) or (
        text.startswith("~~~") and text.endswith("~~~")
    ):
        lines = text.splitlines()
        if lines and (lines[0].startswith("```") or lines[0].startswith("~~~")):
            lines = lines[1:]
        if lines and (lines[-1].startswith("```") or lines[-1].startswith("~~~")):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def parse_action_previous(text: str) -> tuple[StepBlock | None, list[str]]:
    """Parse <|action|> and optional <|previous|> lines from an LLM response."""
    if not isinstance(text, str):
        return None, ["non-string output"]
    cleaned = _strip_code_fences(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if not lines:
        return None, ["empty output"]
    action_match = ACTION_RE.search(cleaned)
    action = action_match.group(1).lower() if action_match else None
    prev_match = PREV_RE.search(cleaned)
    prev_list = [int(x) for x in re.findall(r"\d+", prev_match.group(1))] if prev_match else []
    explanation = ""
    for line in lines:
        low = line.lower()
        if "<|action|>" in line or "<|previous|>" in line:
            continue
        if low.startswith("|action|") or low.startswith("|previous|"):
            continue
        explanation = line
        break
    if not explanation:
        explanation = lines[0]
    if action not in {"continue", "backtrack", "merge"}:
        return None, [f"invalid action: {action}"]
    return StepBlock(action=action, prev_list=prev_list, explanation=explanation, raw_text=text), []


def _pick_explanation(blocks: list[StepBlock], action: str, prev_list: list[int] | None = None) -> str:
    if prev_list is not None:
        for block in blocks:
            if block.action == action and sorted(block.prev_list) == sorted(prev_list):
                return block.explanation
    for block in blocks:
        if block.action == action:
            return block.explanation
    return ""


def majority_vote(
    blocks: list[StepBlock],
    *,
    last_previous_step_id: int,
    main_path: list[int],
    other_leaves: list[int],
    random_seed: int | None = None,
) -> tuple[str, list[int], str]:
    """Aggregate valid candidates using action majority and deterministic tie policy."""
    if not blocks:
        return "continue", [last_previous_step_id], ""

    action_counts = Counter(block.action for block in blocks)
    max_votes = max(action_counts.values())
    tied_actions = {action for action, count in action_counts.items() if count == max_votes}
    for preference in ("continue", "backtrack", "merge"):
        if preference in tied_actions:
            action = preference
            break
    else:  # pragma: no cover - impossible with non-empty counts
        action = "continue"

    if action == "continue":
        return action, [last_previous_step_id], _pick_explanation(blocks, action)

    rng = random.Random(random_seed)
    if action == "backtrack":
        candidates = [
            block.prev_list[0]
            for block in blocks
            if block.action == "backtrack" and len(block.prev_list) == 1
        ]
        if not candidates:
            return "continue", [last_previous_step_id], _pick_explanation(blocks, "continue")
        parent_counts = Counter(candidates)
        max_parent_votes = max(parent_counts.values())
        tied = sorted(parent for parent, count in parent_counts.items() if count == max_parent_votes)
        chosen = rng.choice(tied) if random_seed is not None and len(tied) > 1 else tied[0]
        return action, [chosen], _pick_explanation(blocks, action, [chosen])

    merge_sets = [
        tuple(sorted(dict.fromkeys(block.prev_list)))
        for block in blocks
        if block.action == "merge" and len(block.prev_list) >= 2
    ]
    if not merge_sets:
        return "continue", [last_previous_step_id], _pick_explanation(blocks, "continue")
    set_counts = Counter(merge_sets)
    max_set_votes = max(set_counts.values())
    tied_sets = sorted(s for s, count in set_counts.items() if count == max_set_votes)
    chosen_set = rng.choice(tied_sets) if random_seed is not None and len(tied_sets) > 1 else tied_sets[0]
    prev_list = list(chosen_set)
    return action, prev_list, _pick_explanation(blocks, action, prev_list)


def _compute_depth(children_map: dict[int, list[int]], root_id: int = 0) -> dict[int, int]:
    depth: dict[int, int] = {root_id: 0}
    queue: deque[int] = deque([root_id])
    while queue:
        cur = queue.popleft()
        for nxt in children_map.get(cur, []):
            cand = depth[cur] + 1
            if nxt not in depth or cand < depth[nxt]:
                depth[nxt] = cand
                queue.append(nxt)
    for node in children_map:
        depth.setdefault(node, 0)
    return depth


def _graph_to_jsonable(
    parents: dict[int, list[int]],
    children: dict[int, list[int]],
    depth: dict[int, int],
    explanations: dict[int, str],
    node_texts: dict[int, str],
    actions: dict[int, str],
    errors: list[str],
) -> dict[str, Any]:
    leaves = sorted(node for node in set(parents) | set(children) if not children.get(node))
    return {
        "parents": {str(k): sorted(v) for k, v in parents.items()},
        "children": {str(k): sorted(v) for k, v in children.items()},
        "depth": {str(k): int(v) for k, v in depth.items()},
        "node_texts": {str(k): v for k, v in node_texts.items()},
        "explanations": {str(k): v for k, v in explanations.items()},
        "actions": {str(k): v for k, v in actions.items()},
        "leaves": leaves,
        "errors": list(errors),
    }


def ensure_chain_dag_from_steps(steps: list[str]) -> dict[str, Any]:
    parents: dict[int, list[int]] = {}
    children: dict[int, list[int]] = {}
    node_texts: dict[int, str] = {}
    explanations: dict[int, str] = {}
    actions: dict[int, str] = {}
    if not steps:
        parents[0] = []
        children[0] = []
        node_texts[0] = ""
        explanations[0] = ""
        actions[0] = "root"
    for idx, step in enumerate(steps):
        parents[idx] = [] if idx == 0 else [idx - 1]
        children[idx] = [] if idx == len(steps) - 1 else [idx + 1]
        node_texts[idx] = step
        explanations[idx] = "root" if idx == 0 else ""
        actions[idx] = "root" if idx == 0 else "continue"
    return _graph_to_jsonable(
        parents,
        children,
        _compute_depth(children, 0),
        explanations,
        node_texts,
        actions,
        [],
    )


def collapse_continue_chains(dag_raw: dict[str, Any], step_texts: list[str]) -> dict[str, Any]:
    """Compress maximal single-parent/single-child continue chains into super-nodes."""
    parents = {int(k): [int(x) for x in v] for k, v in (dag_raw.get("parents") or {}).items()}
    children = {int(k): [int(x) for x in v] for k, v in (dag_raw.get("children") or {}).items()}
    actions = {int(k): str(v) for k, v in (dag_raw.get("actions") or {}).items()}

    if not parents and not children:
        dag_raw = ensure_chain_dag_from_steps(step_texts)
        parents = {int(k): [int(x) for x in v] for k, v in dag_raw["parents"].items()}
        children = {int(k): [int(x) for x in v] for k, v in dag_raw["children"].items()}
        actions = {int(k): str(v) for k, v in dag_raw["actions"].items()}

    all_nodes = sorted(set(parents) | set(children))
    if not all_nodes:
        parents = {0: []}
        children = {0: []}
        actions = {0: "root"}
        all_nodes = [0]

    indeg = {node: len(parents.get(node, [])) for node in all_nodes}
    outdeg = {node: len(children.get(node, [])) for node in all_nodes}

    def is_continue_edge(src: int, dst: int) -> bool:
        return (
            actions.get(dst) == "continue"
            and indeg.get(dst, 0) == 1
            and parents.get(dst, [None])[0] == src
        )

    cont_children = {
        node: [child for child in children.get(node, []) if is_continue_edge(node, child)]
        for node in all_nodes
    }
    assigned: set[int] = set()
    chains: list[list[int]] = []

    def is_chain_head(node: int) -> bool:
        if indeg.get(node, 0) != 1 or actions.get(node) != "continue":
            return True
        parent = parents[node][0]
        return outdeg.get(parent, 0) != 1

    for node in all_nodes:
        if node in assigned or not is_chain_head(node):
            continue
        chain = [node]
        assigned.add(node)
        cur = node
        while True:
            candidates = cont_children.get(cur, [])
            if outdeg.get(cur, 0) != 1 or len(candidates) != 1:
                break
            nxt = candidates[0]
            chain.append(nxt)
            assigned.add(nxt)
            if outdeg.get(nxt, 0) != 1:
                break
            cur = nxt
        chains.append(chain)

    for node in all_nodes:
        if node not in assigned:
            chains.append([node])
            assigned.add(node)

    chains.sort(key=lambda seq: seq[-1])
    raw_to_merged: dict[int, int] = {}
    merged_nodes: list[dict[str, Any]] = []
    for mid, seq in enumerate(chains):
        for raw_id in seq:
            raw_to_merged[raw_id] = mid
        merged_nodes.append({"id": mid, "raw": seq[:], "last_raw": seq[-1]})

    parents_m: dict[int, list[int]] = {int(node["id"]): [] for node in merged_nodes}
    children_m: dict[int, list[int]] = {int(node["id"]): [] for node in merged_nodes}
    for child_raw, parent_list in parents.items():
        if child_raw not in raw_to_merged:
            continue
        child_mid = raw_to_merged[child_raw]
        for parent_raw in parent_list:
            if parent_raw not in raw_to_merged:
                continue
            parent_mid = raw_to_merged[parent_raw]
            if parent_mid == child_mid:
                continue
            if parent_mid not in parents_m[child_mid]:
                parents_m[child_mid].append(parent_mid)
            if child_mid not in children_m[parent_mid]:
                children_m[parent_mid].append(child_mid)

    for mapping in (parents_m, children_m):
        for key, values in mapping.items():
            mapping[key] = sorted(dict.fromkeys(values))

    root_mid = raw_to_merged.get(0, 0)
    depth_m = _compute_depth(children_m, root_mid)

    node_texts_m: dict[int, str] = {}
    explanations_m: dict[int, str] = {}
    for merged_node in merged_nodes:
        mid = int(merged_node["id"])
        seq = [int(x) for x in merged_node["raw"]]
        header = "[MergedNode] contains raw: " + ", ".join(f"s{raw_id}" for raw_id in seq)
        body_parts = []
        for raw_id in seq:
            raw_text = step_texts[raw_id] if 0 <= raw_id < len(step_texts) else ""
            body_parts.append(f"-- s{raw_id} --\n{raw_text}".rstrip())
        node_texts_m[mid] = header + "\n\n" + "\n\n".join(body_parts)
        explanations_m[mid] = f"Merged tail = s{seq[-1]}"

    last_raw_by_mid = {int(merged_node["id"]): int(merged_node["last_raw"]) for merged_node in merged_nodes}
    mids_sorted = sorted(last_raw_by_mid, key=lambda mid: last_raw_by_mid[mid])
    names = {mid: f"s{rank}" for rank, mid in enumerate(mids_sorted)}
    return {
        "root_merged": root_mid,
        "merged_nodes": [
            {"id": int(node["id"]), "raw": list(node["raw"]), "last_raw": int(node["last_raw"])}
            for node in merged_nodes
        ],
        "raw_to_merged": {str(raw): int(mid) for raw, mid in raw_to_merged.items()},
        "parents": {str(k): v for k, v in parents_m.items()},
        "children": {str(k): v for k, v in children_m.items()},
        "depth": {str(k): int(v) for k, v in depth_m.items()},
        "node_texts": {str(k): v for k, v in node_texts_m.items()},
        "explanations": {str(k): v for k, v in explanations_m.items()},
        "names": {str(k): v for k, v in names.items()},
        "actions": {},
        "errors": [],
    }


def graph_to_jsonable(
    parents: dict[int, list[int]],
    children: dict[int, list[int]],
    explanations: dict[int, str],
    node_texts: dict[int, str],
    actions: dict[int, str],
    errors: list[str],
) -> dict[str, Any]:
    for node in parents:
        children.setdefault(node, [])
    return _graph_to_jsonable(
        parents,
        children,
        _compute_depth(children, 0),
        explanations,
        node_texts,
        actions,
        errors,
    )
