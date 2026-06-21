from trm_dag.components import StepBlock, majority_vote


def test_strict_majority_action() -> None:
    blocks = [
        StepBlock("backtrack", [1], "a", "raw"),
        StepBlock("backtrack", [1], "b", "raw"),
        StepBlock("continue", [], "c", "raw"),
    ]
    action, prev, _ = majority_vote(
        blocks,
        last_previous_step_id=3,
        main_path=[0, 1, 2, 3],
        other_leaves=[],
    )
    assert action == "backtrack"
    assert prev == [1]


def test_action_tie_prefers_continue() -> None:
    blocks = [
        StepBlock("continue", [], "a", "raw"),
        StepBlock("backtrack", [1], "b", "raw"),
    ]
    action, prev, _ = majority_vote(
        blocks,
        last_previous_step_id=2,
        main_path=[0, 1, 2],
        other_leaves=[],
    )
    assert action == "continue"
    assert prev == [2]


def test_random_seed_makes_parent_choice_deterministic() -> None:
    blocks = [
        StepBlock("backtrack", [1], "a", "raw"),
        StepBlock("backtrack", [2], "b", "raw"),
    ]
    first = majority_vote(
        blocks,
        last_previous_step_id=3,
        main_path=[0, 1, 2, 3],
        other_leaves=[],
        random_seed=7,
    )
    second = majority_vote(
        blocks,
        last_previous_step_id=3,
        main_path=[0, 1, 2, 3],
        other_leaves=[],
        random_seed=7,
    )
    assert first == second


def test_merge_vote_aggregates_parent_sets() -> None:
    blocks = [
        StepBlock("merge", [1, 4], "a", "raw"),
        StepBlock("merge", [4, 1], "b", "raw"),
        StepBlock("merge", [2, 4], "c", "raw"),
    ]
    action, prev, explanation = majority_vote(
        blocks,
        last_previous_step_id=3,
        main_path=[0, 1, 2, 3],
        other_leaves=[4],
    )
    assert action == "merge"
    assert prev == [1, 4]
    assert explanation == "a"


def test_merge_vote_falls_back_without_parent_sets() -> None:
    action, prev, _ = majority_vote(
        [StepBlock("merge", [], "bad", "raw")],
        last_previous_step_id=3,
        main_path=[0, 1, 2, 3],
        other_leaves=[4],
    )
    assert action == "continue"
    assert prev == [3]
