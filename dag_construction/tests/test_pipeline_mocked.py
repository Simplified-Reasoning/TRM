from pathlib import Path

from conftest import MockOpenAIClient, openai_config

from trm_dag import DagParams, build_dag, build_dag_batch


def test_end_to_end_mocked_small_example() -> None:
    client = MockOpenAIClient(
        [
            ["continue\n<|action|>continue"],
            ["continue\n<|action|>continue"],
        ]
    )
    result = build_dag(
        "prompt",
        ["s0", "s1", "s2"],
        openai_config=openai_config(),
        dag_params=DagParams(regen_limit=1),
        client=client,
    )
    assert result["dag_graph_raw"]["parents"]["2"] == [1]
    assert result["dag_graph_merged"]["merged_nodes"][0]["raw"] == [0, 1, 2]
    assert result["dag_usage"]["calls"] == 2


def test_persistent_parse_failure_falls_back_to_continue() -> None:
    client = MockOpenAIClient([["not parseable"], ["still bad"]])
    result = build_dag(
        "prompt",
        ["s0", "s1"],
        openai_config=openai_config(),
        dag_params=DagParams(regen_limit=2),
        client=client,
    )
    assert result["dag_graph_raw"]["parents"]["1"] == [0]
    assert result["dag_graph_raw"]["actions"]["1"] == "continue"
    assert result["dag_graph_raw"]["node_texts"]["1"] == "s1"
    assert any("no valid candidates" in err for err in result["dag_parse_errors"])


def test_batch_rebuild_truncates_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"old": true}\n', encoding="utf-8")
    client = MockOpenAIClient([["continue\n<|action|>continue"]])
    rows = build_dag_batch(
        [{"prompt": "p", "steps": ["s0", "s1"]}],
        openai_config=openai_config(),
        dag_params=DagParams(regen_limit=1),
        client=client,
        output_path=output_path,
        resume=False,
    )
    text = output_path.read_text(encoding="utf-8")
    assert text.count("\n") == 1
    assert '"old"' not in text
    assert rows[0]["dag_graph_raw"]["parents"]["1"] == [0]


def test_batch_resume_truncates_invalid_tail(tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"prompt":"p","steps":["s0","s1"],"done":true}\n{"bad"', encoding="utf-8")
    rows = build_dag_batch(
        [
            {"prompt": "p", "steps": ["s0", "s1"]},
            {"prompt": "q", "steps": ["s0", "s1"]},
        ],
        openai_config=openai_config(),
        dag_params=DagParams(regen_limit=1),
        client=MockOpenAIClient([["continue\n<|action|>continue"]]),
        output_path=output_path,
        resume=True,
    )
    text = output_path.read_text(encoding="utf-8")
    assert '{"bad"' not in text
    assert len(rows) == 2
    assert rows[1]["dag_graph_raw"]["parents"]["1"] == [0]


def test_batch_resume_rejects_changed_steps(tmp_path: Path) -> None:
    output_path = tmp_path / "out.jsonl"
    output_path.write_text('{"prompt":"same","steps":["old"]}\n', encoding="utf-8")
    try:
        build_dag_batch(
            [{"prompt": "same", "steps": ["new"]}],
            openai_config=openai_config(),
            dag_params=DagParams(regen_limit=1),
            client=MockOpenAIClient([["continue\n<|action|>continue"]]),
            output_path=output_path,
            resume=True,
        )
    except RuntimeError as exc:
        assert "resume key mismatch" in str(exc)
    else:
        raise AssertionError("expected resume key mismatch")
