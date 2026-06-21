from trm_dag.partition import partition_keyword


def test_paragraph_mode_round_trip() -> None:
    trace = "Intro\n\nStep one.\n\nThen two.\n\nFinally done."
    steps, fillers = partition_keyword(trace, keywords=("Step", "Then", "Finally"))
    assert len(steps) == 3
    assert len(fillers) == 2
    assert "".join(part for pair in zip(steps, fillers + [""], strict=True) for part in pair) == trace


def test_default_keywords_split_paragraphs_only() -> None:
    steps, fillers = partition_keyword("Step one. Then same paragraph.\n\nNext two.")
    assert steps == ["Step one. Then same paragraph.", "Next two."]
    assert fillers == ["\n\n"]


def test_paragraph_partition_preserves_indentation_and_blank_lines() -> None:
    trace = "Step one:\n    code line\n\n\nThen two:\n  aligned"
    steps, fillers = partition_keyword(trace, keywords=("Step", "Then"))
    assert steps == ["Step one:\n    code line", "Then two:\n  aligned"]
    assert fillers == ["\n\n\n"]
