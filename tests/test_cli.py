import json
import subprocess
import sys

from rag_lab.cli import benchmark, evaluate


def test_evaluation_reports_stages_and_slices():
    result = evaluate("data/retrieval_queries.json", "holdout")
    assert result["query_count"] == 8
    assert set(result["modes"]) == {"bm25", "dense", "hybrid", "weighted", "rerank"}
    assert result["modes"]["bm25"]["macro"]["recall"] >= 0.75
    assert "union_recall" in result["modes"]["hybrid"]["details"][0]


def test_cli_installed_entrypoint(tmp_path):
    output = tmp_path / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from rag_lab.cli import main; main()",
            "evaluate",
            "--output",
            str(output),
        ],
        check=True,
    )
    assert json.loads(output.read_text())["query_count"] == 8


def test_benchmark_error_guard():
    assert benchmark(10, 2)["errors"] == 0


def test_cli_ask_capacity_benchmark_and_sweep(monkeypatch, capsys, tmp_path):
    from rag_lab.cli import main

    for arguments in (
        ["ask", "PHONE-A-256 S01 到店多少钱"],
        ["capacity"],
        ["benchmark", "--requests", "4", "--workers", "2"],
        ["sweep", "--split", "dev", "--output", str(tmp_path / "sweep.json")],
        ["evaluate", "--split", "dev", "--output", str(tmp_path / "dev.json")],
    ):
        monkeypatch.setattr(sys, "argv", ["rag-lab", *arguments])
        main()
    output = capsys.readouterr().out
    assert "amount_cents" in output and "raw_vectors" in output
    assert len(json.loads((tmp_path / "sweep.json").read_text())) == 6


def test_holdout_cannot_be_used_for_parameter_search(monkeypatch):
    import pytest

    from rag_lab.cli import main

    monkeypatch.setattr(sys, "argv", ["rag-lab", "sweep", "--split", "holdout"])
    with pytest.raises(SystemExit):
        main()
