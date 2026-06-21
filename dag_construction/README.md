# TRM DAG Construction

Build DAGs over reasoning traces.

> Example / reference implementation. Defaults reflect one specific setup — swap pieces out for your own data and models.

## Pipeline

Three steps, in order:

```text
(1) partition  ->  (2) build_dag  ->  (3) merge_view
```

1. **`partition`** — split a raw `trace` into an ordered list of steps.
2. **`build_dag`** — for each new step, ask an LLM to classify how it attaches to earlier steps as `continue` / `backtrack` / `merge`, then validate and majority-vote into parents.
3. **`merge_view`** — collapse linear `continue` chains into super-nodes.

## Quickstart

```bash
cd dag_construction
pip install -e ".[dev]"
cp .env.example .env
# Fill DAG_CONSTRUCTION_API_KEY, DAG_CONSTRUCTION_MODEL, optional DAG_CONSTRUCTION_BASE_URL.
python examples/run_single.py
```

Batch JSONL:

```bash
python -m trm_dag build --input items.jsonl --output out.jsonl --num-threads 8 --resume
```

## Customizing each step

### 1. `partition`

- Different keywords: `partition_keyword(..., keywords=("Step", "Stage", ...))` or CLI `--keywords ...`.
- Skip it: pass rows with `"steps": [...]` directly to `build_dag_batch`.
- Custom splitter: write your own `(steps, fillers)` function and feed `"steps"` in. See `trm_dag/partition.py` for the contract.

### 2. `build_dag`

- Prompt: edit `trm_dag/prompts/dag_construction.txt` (keep the `$current_id`, `$input_steps`, `$available_steps`, `$leaf_steps`, `$last_previous_step_id` placeholders).
- LLM backend: any OpenAI-compatible endpoint via `DAG_CONSTRUCTION_BASE_URL` / `DAG_CONSTRUCTION_MODEL`. For non-OpenAI APIs, implement the `ChatClient` protocol in `trm_dag/client.py` and pass `client=` to `build_dag` / `build_dag_batch`.
- Sampling: `OpenAIConfig(temperature, top_p, n, max_tokens)` and `DagParams(regen_limit)`, or matching CLI flags.
- Attachment pool size: `DagParams(main_path_cap=..., other_leaf_cap=...)`.
- Action set / voting: edit `parse_action_previous`, `_validate_candidate`, `majority_vote` in `trm_dag/components.py` (and the prompt).
- Determinism: `DagParams(random_seed=0)`.

### 3. `merge_view`

- Different merge rule: edit `collapse_continue_chains` in `trm_dag/components.py` (`is_continue_edge`, `is_chain_head`).
- Skip it: just read `result["dag_graph_raw"]`.
- Per-super-node summarization: post-process `dag_graph_merged["merged_nodes"]` with your own summarizer.

## Python API

```python
from trm_dag import DagParams, OpenAIConfig, build_dag

result = build_dag(
    prompt,
    steps,
    openai_config=OpenAIConfig.from_env(),
    dag_params=DagParams(random_seed=0),
)

raw = result["dag_graph_raw"]
merged = result["dag_graph_merged"]
```

Batch:

```python
from trm_dag import DagParams, OpenAIConfig, build_dag_batch

rows = [
    {"prompt": "...", "steps": ["Step one.", "Then two."]},
    {"prompt": "...", "trace": "Step one.\n\nThen two."},
]

out_rows = build_dag_batch(
    rows,
    openai_config=OpenAIConfig.from_env(),
    dag_params=DagParams(regen_limit=5, random_seed=0),
    output_path="items_dag.jsonl",
    resume=True,
)
```

## CLI

```bash
python -m trm_dag build --input items.jsonl --output items_dag.jsonl
```

Build flags: `--input`, `--output`, `--num-threads`, `--resume` / `--no-resume` / `--rebuild`, `--keywords`, `--regen-limit`, `--main-path-cap`, `--other-leaf-cap`, `--random-seed`.

LLM flags: `--api-key`, `--base-url`, `--model`, `--temperature`, `--top-p`, `--max-tokens`, `--n`, `--timeout`.

## Input And Output

Input rows can contain pre-split steps:

```json
{"prompt": "...", "steps": ["Step one.", "Then two."]}
```

or a raw trace:

```json
{"prompt": "...", "trace": "Step one.\\n\\nThen two."}
```

Output rows add `dag_graph_raw`, `dag_graph_merged`, `dag_parse_errors`, `dag_usage`. When input contains `trace`, `build_dag_batch` also keeps generated `steps` and `fillers` on the output row.
