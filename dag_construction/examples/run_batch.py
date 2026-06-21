from __future__ import annotations

import argparse
import json
from pathlib import Path

from trm_dag import DagParams, OpenAIConfig, build_dag_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(Path(__file__).with_name("data.json")))
    parser.add_argument("--output", default="out.jsonl")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    build_dag_batch(
        rows,
        openai_config=OpenAIConfig.from_env(max_tokens=1024),
        dag_params=DagParams(random_seed=0),
        num_threads=args.num_threads,
        output_path=args.output,
        resume=True,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
