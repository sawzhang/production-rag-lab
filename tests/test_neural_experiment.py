"""Offline experiment orchestration tests; real inference lives in test_real_models.py."""

import json

import pytest

from rag_lab import neural_experiment as experiment
from rag_lab.retrieval import LSADense, OverlapReranker


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(
        experiment, "load_models", lambda *args, **kwargs: (LSADense(), OverlapReranker())
    )
    monkeypatch.setattr(experiment, "verify_weights", lambda profile: {"test": True})
    monkeypatch.setattr(experiment.importlib.metadata, "version", lambda name: "test-version")


def test_paired_evaluation_pins_inputs_and_candidate_pool(fake_models, tmp_path):
    rows = json.loads(open("data/neural_challenge_queries.json").read())[:2]
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(rows))
    output = tmp_path / "result.json"
    result = experiment.neural_evaluate([path], output)
    assert result["query_count"] == 2 and len(result["modes"]) == 7
    assert len(result["inputs"][0]["sha256"]) == 64
    a = result["modes"]["neural_hybrid_cross_encoder"]["details"]
    b = result["modes"]["neural_hybrid_rrf"]["details"]
    assert all(x["fused_candidate_ids"] == y["fused_candidate_ids"] for x, y in zip(a, b))
    assert result["paired_comparisons"][-1]["mean_ndcg_delta"] == 0
    assert json.loads(output.read_text())["seed"] == 42


def test_empty_or_duplicate_queries_rejected(fake_models, tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="nonempty"):
        experiment.neural_evaluate([path])
    rows = json.loads(open("data/neural_challenge_queries.json").read())[:1]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="unique"):
        experiment.neural_evaluate([path, path])


def test_illegal_gold_rejected(fake_models, tmp_path):
    rows = json.loads(open("data/neural_challenge_queries.json").read())[:1]
    rows[0]["gold"] = {"other-tenant": 2}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="gold outside"):
        experiment.neural_evaluate([path])


def test_fallback_cannot_masquerade_as_model_effect(fake_models, monkeypatch, tmp_path):
    rows = json.loads(open("data/neural_challenge_queries.json").read())[:1]
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(rows))
    original = experiment.Retriever.search

    def degraded(self, *args, **kwargs):
        docs, trace = original(self, *args, **kwargs)
        trace["degraded"] = True
        return docs, trace

    monkeypatch.setattr(experiment.Retriever, "search", degraded)
    with pytest.raises(RuntimeError, match="degraded"):
        experiment.neural_evaluate([path])


def test_generation_checkpoint_reuses_valid_calls_and_preserves_failures(
    fake_models, monkeypatch, tmp_path
):
    import rag_lab.generation as generation

    calls = []
    monkeypatch.setattr(generation, "resolve_key", lambda *a: "fake-only")

    def generate(bundle, query, key):
        calls.append(query)
        citations = [{"id": e["id"], "quote": e["text"]} for e in bundle["evidence"][:4]]
        status = "answered" if citations or bundle["fact"] else "insufficient"
        if len([c for c in citations if c["id"].startswith("conflict")]) == 2:
            status = "conflict"
        if any(c["id"] == "injection" for c in citations):
            status = "insufficient"
            citations = []
        return {"status": status, "fact": bundle["fact"], "citations": citations}, {
            "usage": {"total_tokens": 1}
        }

    monkeypatch.setattr(generation, "generate", generate)
    path = tmp_path / "generation.json"
    first = experiment.generation_evaluate(output=path)
    assert first["completed"] == 8 and len(calls) == 8
    second = experiment.generation_evaluate(output=path)
    assert second["completed"] == 8 and len(calls) == 8 + (8 - first["passed"])
    assert "fake-only" not in path.read_text()


def test_generation_api_failures_are_recorded_without_false_success(
    fake_models, monkeypatch, tmp_path
):
    import rag_lab.generation as generation

    monkeypatch.setattr(generation, "resolve_key", lambda *a: "fake-only")

    def fail(*args, **kwargs):
        raise RuntimeError("DeepSeek network/timeout failure")

    monkeypatch.setattr(generation, "generate", fail)
    result = experiment.generation_evaluate(output=tmp_path / "fail.json")
    assert result["completed"] == 8 and result["passed"] == 0
    assert result["usage_totals"]["total_tokens"] == 0
    assert all(r["draft"] is None and not r["validation_passed"] for r in result["details"])


def test_constrained_ablation_records_code_and_scoped_candidate_pool(fake_models, tmp_path):
    rows = json.loads(open("data/neural_challenge_queries.json").read())
    rows = [r for r in rows if r["id"] in {"N06", "N10", "N20"}]
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(rows))
    r = experiment.neural_evaluate([path], include_constrained=True)
    assert len(r["modes"]) == 8 and len(r["routing_source_sha256"]) == 5
    d = {x["id"]: x for x in r["modes"]["constrained_cross_encoder"]["details"]}
    assert d["N20"]["ranking"] == ["phone128"] and d["N10"]["ranking"] == ["graph-rule"]
    assert set(d["N06"]["scoped_ids"]) == {"menu-channel", "menu-contract"}
    assert r["paired_comparisons"][-1]["left"] == "constrained_cross_encoder"
