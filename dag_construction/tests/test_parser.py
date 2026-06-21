from trm_dag.components import parse_action_previous


def test_parse_continue_tag() -> None:
    block, errors = parse_action_previous("continues normally\n<|action|>continue")
    assert errors == []
    assert block is not None
    assert block.action == "continue"
    assert block.prev_list == []
    assert block.explanation == "continues normally"


def test_parse_colon_style_and_previous() -> None:
    block, errors = parse_action_previous("restart\n|action|: backtrack\n|previous|: s1")
    assert errors == []
    assert block is not None
    assert block.action == "backtrack"
    assert block.prev_list == [1]


def test_parse_code_fence() -> None:
    text = "```text\nmerge branches\n<|action|>merge\n<|previous|>1, 4\n```"
    block, errors = parse_action_previous(text)
    assert errors == []
    assert block is not None
    assert block.action == "merge"
    assert block.prev_list == [1, 4]


def test_json_style_fails() -> None:
    block, errors = parse_action_previous('{"action": "continue"}')
    assert block is None
    assert errors


def test_empty_fails() -> None:
    block, errors = parse_action_previous("")
    assert block is None
    assert errors == ["empty output"]
