from collections import deque

from conftest import MockOpenAIClient, openai_config

from trm_dag import DagParams, build_dag


def assert_valid_dag(dag: dict) -> None:
    parents = {int(k): [int(x) for x in v] for k, v in dag["parents"].items()}
    children = {int(k): [int(x) for x in v] for k, v in dag["children"].items()}
    actions = {int(k): v for k, v in dag["actions"].items()}
    assert actions[0] == "root"
    assert all(action in {"root", "continue", "backtrack", "merge"} for action in actions.values())
    for node, parent_list in parents.items():
        if node != 0:
            assert len(parent_list) >= 1

    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for child in children.get(node, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    assert seen == set(parents)

    visiting: set[int] = set()
    done: set[int] = set()

    def dfs(node: int) -> None:
        assert node not in visiting
        if node in done:
            return
        visiting.add(node)
        for child in children.get(node, []):
            dfs(child)
        visiting.remove(node)
        done.add(node)

    dfs(0)


def test_mock_pipeline_produces_valid_dag() -> None:
    client = MockOpenAIClient(
        [
            ["s1 follows\n<|action|>continue"],
            ["restart\n<|action|>backtrack\n<|previous|>0"],
            ["join\n<|action|>merge\n<|previous|>1,2"],
        ]
    )
    result = build_dag(
        "p",
        ["root", "a", "b", "c"],
        openai_config=openai_config(),
        dag_params=DagParams(regen_limit=1, random_seed=0),
        client=client,
    )
    assert_valid_dag(result["dag_graph_raw"])
    assert result["dag_graph_raw"]["leaves"] == [3]
