"""Adapter contract tests; fake model objects, not a model quality claim."""

import sys
import types

import numpy as np
import pytest

from rag_lab.contracts import Document
from rag_lab.neural import NeuralDense, NeuralReranker


def test_neural_revision_mandatory():
    for adapter in (NeuralDense, NeuralReranker):
        with pytest.raises(ValueError):
            adapter("anything", "main")


def test_real_adapter_contract_with_fake_models(monkeypatch):
    calls = []

    class Encoder:
        def __init__(self, id, **kwargs):
            calls.append(kwargs)

        def encode_document(self, texts, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.array([[1.0, 0.0], [0.0, 1.0]])

        def encode_query(self, texts, **kwargs):
            return np.array([[0.0, 1.0]])

    class Cross(Encoder):
        def predict(self, pairs):
            assert pairs[0][0] == "query"
            return [0.1, 0.9]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=Encoder, CrossEncoder=Cross),
    )
    docs = [Document("a", "a", 1, "A", "a"), Document("b", "b", 1, "B", "b")]
    dense = NeuralDense("encoder", "a" * 40)
    cross = NeuralReranker("cross", "b" * 40)
    assert dense.search("query", docs)[0].id == "b"
    assert cross.rank("query", docs)[0].id == "b"
    assert dense.search("query", []) == []
    assert all(c["trust_remote_code"] is False for c in calls)
    assert calls[0]["revision"] == "a" * 40
