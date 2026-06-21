from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template
from typing import Any

from .client import ChatClient, OpenAIChatClient, add_usage
from .components import (
    StepBlock,
    collapse_continue_chains,
    graph_to_jsonable,
    majority_vote,
    parse_action_previous,
    render_attachment_pool,
)
from .config import DagParams, OpenAIConfig
from .io import OrderedBufferedWriter, compute_resume_start_and_check_order
from .partition import partition_keyword

SLEEP_BETWEEN_RETRIES = 0.5
MAX_N_PER_REQUEST = 4


def load_prompt_template(path: str | os.PathLike[str] | None = None) -> Template:
    if path is None:
        path = Path(__file__).with_name("prompts") / "dag_construction.txt"
    return Template(Path(path).read_text(encoding="utf-8"))


def _validate_candidate(
    block: StepBlock,
    *,
    idx: int,
    main_path: list[int],
    other_leaves: list[int],
    last_previous_step_id: int,
) -> StepBlock | None:
    cand_prev = [int(p) for p in block.prev_list if isinstance(p, int) and 0 <= p < idx]
    action = block.action

    if action == "continue":
        cand_prev = []
    elif action == "backtrack":
        backtrack_candidates = [
            p for p in cand_prev if p in main_path and p < last_previous_step_id
        ]
        if not backtrack_candidates:
            return None
        cand_prev = [max(backtrack_candidates, key=lambda x: main_path.index(x))]
    elif action == "merge":
        uniq_prev = list(dict.fromkeys(cand_prev))
        main_hits = [p for p in uniq_prev if p in main_path]
        leaf_hits = [p for p in uniq_prev if p in other_leaves]
        if len(uniq_prev) < 2 or len(leaf_hits) < 1 or len(main_hits) > 1:
            return None
        cand_prev = uniq_prev
    else:
        return None

    return StepBlock(
        action=action,
        prev_list=cand_prev,
        explanation=block.explanation,
        raw_text=block.raw_text,
    )


def build_dag(
    prompt: str,
    steps: list[str],
    *,
    openai_config: OpenAIConfig,
    dag_params: DagParams | None = None,
    client: ChatClient | None = None,
) -> dict[str, Any]:
    """Build raw and merged DAGs for already partitioned reasoning steps."""
    if dag_params is None:
        dag_params = DagParams()
    if client is None:
        client = OpenAIChatClient()
    template = load_prompt_template()

    clean_steps = [str(step) for step in steps]
    if not clean_steps:
        raise ValueError("steps must contain at least one step")

    parents: dict[int, list[int]] = {0: []}
    children: dict[int, list[int]] = {0: []}
    actions: dict[int, str] = {0: "root"}
    explanations: dict[int, str] = {0: "root"}
    node_texts: dict[int, str] = {0: clean_steps[0]}
    raw_outputs: dict[int, str] = {0: ""}
    errors: list[str] = []
    usage_all = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    main_path: list[int] = [0]
    leaf_set: set[int] = {0}
    base_n = max(1, int(openai_config.n or 1))
    attempts_max = max(0, int(dag_params.regen_limit) - 1)
    max_n_per_request = MAX_N_PER_REQUEST

    for idx in range(1, len(clean_steps)):
        last_previous_step_id = main_path[-1]
        other_leaves = sorted(leaf_set - set(main_path))
        input_block = render_attachment_pool(
            clean_steps,
            idx,
            main_path,
            other_leaves,
            main_path_cap=dag_params.main_path_cap,
            other_leaf_cap=dag_params.other_leaf_cap,
        )
        prompt_text = template.safe_substitute(
            current_id=idx,
            input_steps=input_block,
            available_steps=",".join(str(i) for i in main_path),
            leaf_steps=",".join(str(i) for i in other_leaves) or "(none)",
            last_previous_step_id=last_previous_step_id,
        )

        aggregated_valid: list[StepBlock] = []
        parse_errors_before_vote: list[str] = []
        for attempt in range(attempts_max + 1):
            n_target = base_n + 2 * attempt
            n_this = min(max_n_per_request, n_target)
            outputs, usage = client.complete(prompt_text, openai_config, n=n_this)
            add_usage(usage_all, usage)
            for candidate in outputs:
                block, parse_errors = parse_action_previous(candidate)
                parse_errors_before_vote.extend(parse_errors)
                if block is None:
                    continue
                valid_block = _validate_candidate(
                    block,
                    idx=idx,
                    main_path=main_path,
                    other_leaves=other_leaves,
                    last_previous_step_id=last_previous_step_id,
                )
                if valid_block is not None:
                    aggregated_valid.append(valid_block)

            if aggregated_valid:
                counts: dict[str, int] = {}
                for block in aggregated_valid:
                    counts[block.action] = counts.get(block.action, 0) + 1
                ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
                if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                    break
            if attempt < attempts_max:
                time.sleep(SLEEP_BETWEEN_RETRIES)

        errors.extend(parse_errors_before_vote)
        if not aggregated_valid:
            errors.append(
                f"s{idx}: no valid candidates after {attempts_max + 1} rounds "
                f"(n up to {base_n + 2 * attempts_max})"
            )
            action = "continue"
            prev_list = [last_previous_step_id]
            explanation = ""
            raw_text = ""
        else:
            action, prev_list, explanation = majority_vote(
                aggregated_valid,
                last_previous_step_id=last_previous_step_id,
                main_path=main_path,
                other_leaves=other_leaves,
                random_seed=(
                    None
                    if dag_params.random_seed is None
                    else int(dag_params.random_seed) + idx
                ),
            )
            raw_text = next(
                (block.raw_text for block in aggregated_valid if block.action == action),
                "",
            )

        if action == "continue":
            prev_list = [main_path[-1]]
        elif action == "backtrack":
            valid = [p for p in prev_list if p < idx and p in main_path]
            if not valid:
                action = "continue"
                prev_list = [main_path[-1]]
            else:
                parent = max(valid, key=lambda x: main_path.index(x))
                prev_list = [parent]
                old_tail = main_path[-1]
                if old_tail != parent:
                    leaf_set.add(old_tail)
                main_path = main_path[: main_path.index(parent) + 1]
        elif action == "merge":
            valid = list(dict.fromkeys(p for p in prev_list if p < idx))
            if len(valid) < 2:
                action = "continue"
                prev_list = [main_path[-1]]
            else:
                prev_list = valid
                main_candidates = [p for p in prev_list if p in main_path]
                if main_candidates:
                    main_parent = max(main_candidates, key=lambda x: main_path.index(x))
                    main_path = main_path[: main_path.index(main_parent) + 1]
                for parent in prev_list:
                    leaf_set.discard(parent)
        else:
            action = "continue"
            prev_list = [main_path[-1]]

        parents[idx] = list(prev_list)
        children[idx] = []
        actions[idx] = action
        explanations[idx] = explanation
        node_texts[idx] = clean_steps[idx]
        raw_outputs[idx] = raw_text

        for parent in prev_list:
            children.setdefault(parent, [])
            if idx not in children[parent]:
                children[parent].append(idx)
            leaf_set.discard(parent)
        leaf_set.add(idx)

        if idx not in main_path:
            main_path.append(idx)

    for node in parents:
        children.setdefault(node, [])

    dag_raw = graph_to_jsonable(parents, children, explanations, node_texts, actions, errors)
    dag_merged = collapse_continue_chains(dag_raw, clean_steps)
    nodes = [{"id": i, "text": clean_steps[i]} for i in range(len(clean_steps))]
    edges = [{"from": p, "to": i} for i, plist in parents.items() for p in plist]
    return {
        "prompt": prompt,
        "nodes": nodes,
        "edges": edges,
        "parents": parents,
        "children": children,
        "actions": actions,
        "raw_outputs": raw_outputs,
        "dag_graph_raw": dag_raw,
        "dag_graph_merged": dag_merged,
        "dag_parse_errors": errors,
        "dag_usage": usage_all,
    }


def _steps_from_item(
    item: dict[str, Any],
    *,
    keywords: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    if isinstance(item.get("steps"), list):
        return [str(x) for x in item["steps"]], [str(x) for x in item.get("fillers", [])]
    if "trace" in item:
        return partition_keyword(str(item.get("trace", "")), keywords=keywords)
    raise ValueError("item must contain either 'steps' or 'trace'")


def build_dag_batch(
    items: list[dict[str, Any]],
    *,
    openai_config: OpenAIConfig,
    dag_params: DagParams | None = None,
    client: ChatClient | None = None,
    num_threads: int | None = None,
    output_path: str | os.PathLike[str] | None = None,
    resume: bool = True,
    keywords: tuple[str, ...] = ("Step", "Then", "Next", "Finally"),
) -> list[dict[str, Any]]:
    """Build DAGs for records, optionally writing ordered JSONL with resume checks."""
    if dag_params is None:
        dag_params = DagParams()
    workers = num_threads if num_threads is not None else min(8, os.cpu_count() or 4)
    start_idx = 0
    if output_path and resume:
        start_idx = compute_resume_start_and_check_order(items, output_path)
    elif output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("", encoding="utf-8")

    results: list[dict[str, Any] | None] = [None] * len(items)
    writer = (
        OrderedBufferedWriter(output_path, total_items=len(items), start_idx=start_idx)
        if output_path
        else None
    )

    def worker(index: int, item: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if index < start_idx:
            return index, item
        steps, fillers = _steps_from_item(
            item,
            keywords=keywords,
        )
        result = build_dag(
            str(item.get("prompt", "")),
            steps,
            openai_config=openai_config,
            dag_params=dag_params,
            client=client,
        )
        out = dict(item)
        out.setdefault("steps", steps)
        if fillers:
            out.setdefault("fillers", fillers)
        out.update(
            {
                "dag_graph_raw": result["dag_graph_raw"],
                "dag_graph_merged": result["dag_graph_merged"],
                "dag_parse_errors": result["dag_parse_errors"],
                "dag_usage": result["dag_usage"],
            }
        )
        return index, out

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {
                pool.submit(worker, idx, item): idx
                for idx, item in enumerate(items)
                if idx >= start_idx
            }
            for future in as_completed(futures):
                idx, row = future.result()
                results[idx] = row
                if writer is not None:
                    writer.offer(idx, row)
    finally:
        if writer is not None:
            writer.close()

    if output_path:
        from .io import tolerant_load_jsonl

        return tolerant_load_jsonl(output_path)

    for idx, item in enumerate(items):
        if idx < start_idx:
            results[idx] = item
    return [row for row in results if row is not None]
