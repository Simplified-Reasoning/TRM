from __future__ import annotations

import json
from pathlib import Path

from trm_dag import DagParams, OpenAIConfig, build_dag


def main() -> None:
    item = json.loads((Path(__file__).with_name("data.json")).read_text(encoding="utf-8"))[0]
    result = build_dag(
        item["prompt"],
        item["steps"],
        openai_config=OpenAIConfig.from_env(max_tokens=1024),
        dag_params=DagParams(random_seed=0),
    )
    print(
        json.dumps(
            {
                "dag_graph_raw": result["dag_graph_raw"],
                "dag_graph_merged": result["dag_graph_merged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
