from trm_dag.components import collapse_continue_chains


def test_linear_chain_collapses_to_single_super_node() -> None:
    dag = {
        "parents": {"0": [], "1": [0], "2": [1]},
        "children": {"0": [1], "1": [2], "2": []},
        "actions": {"0": "root", "1": "continue", "2": "continue"},
    }
    merged = collapse_continue_chains(dag, ["a", "b", "c"])
    assert merged["merged_nodes"] == [{"id": 0, "raw": [0, 1, 2], "last_raw": 2}]


def test_branch_and_merge_super_dag() -> None:
    dag = {
        "parents": {"0": [], "1": [0], "2": [0], "3": [1, 2]},
        "children": {"0": [1, 2], "1": [3], "2": [3], "3": []},
        "actions": {"0": "root", "1": "continue", "2": "backtrack", "3": "merge"},
    }
    merged = collapse_continue_chains(dag, ["a", "b", "c", "d"])
    assert len(merged["merged_nodes"]) == 4
    assert merged["parents"]["3"] == [1, 2]


def test_empty_dag_falls_back_to_chain() -> None:
    merged = collapse_continue_chains({}, ["a", "b"])
    assert merged["merged_nodes"] == [{"id": 0, "raw": [0, 1], "last_raw": 1}]


def test_empty_node_set_recovers_single_node() -> None:
    merged = collapse_continue_chains({"parents": {}, "children": {}, "actions": {}}, [])
    assert merged["merged_nodes"] == [{"id": 0, "raw": [0], "last_raw": 0}]
