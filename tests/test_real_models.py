"""Opt-in integration tests execute actual pinned weights, no fake inference."""

import os
from dataclasses import replace

import numpy as np
import pytest

from rag_lab.fixtures import ALICE, AT, corpus
from rag_lab.neural_experiment import load_models, load_profile
from rag_lab.retrieval import Retriever
from rag_lab.service import Service

pytestmark = [
    pytest.mark.real_models,
    pytest.mark.skipif(
        os.environ.get("RAG_LAB_REAL_MODELS") != "1",
        reason="set RAG_LAB_REAL_MODELS=1 to load real weights",
    ),
]


@pytest.fixture(scope="module")
def models():
    return load_models(load_profile(), local_files_only=True)


def test_real_embedding_shape_normalization_and_semantics(models):
    dense, _ = models
    matrix = np.asarray(
        dense.model.encode_document(
            ["支付扣款成功不等于饮品制作完成", "红富士水果按公斤售卖"], normalize_embeddings=True
        )
    )
    assert matrix.shape == (2, 512)
    assert np.isfinite(matrix).all()
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    query = "钱付了就代表饮料做好了吗"
    docs = corpus().visible(ALICE, AT)
    assert dense.search(query, docs, k=1)[0].id == "states"


def test_real_cross_encoder_prefers_relevant_evidence(models):
    _, cross = models
    store = corpus()
    docs = [store.documents["fruit"], store.documents["states"]]
    ranked = cross.rank("扣款成功是否等于出杯完成", docs)
    assert ranked[0].id == "states"
    assert all(np.isfinite(c.score) for c in ranked)
    assert cross.rank("q", []) == []


def test_real_pipeline_acl_zero_scope_and_exact_fact(models):
    dense, cross = models
    store = corpus()
    retriever = Retriever(store, dense=dense, reranker=cross)
    docs, trace = retriever.search("付款与退款状态", ALICE, AT, mode="rerank", timeout=30)
    assert docs and trace["degraded"] is False
    assert "other-tenant" not in trace["union"] and "private-parent" not in trace["union"]
    assert all("SECRET" not in d.text for d in docs)
    assert retriever.search("支付", replace(ALICE, groups=frozenset()), AT)[0] == []
    r = Service(store, retriever=retriever).answer("PHONE-A-256 S01 到店会员价多少钱", ALICE, AT)
    assert r["fact"]["amount_cents"] == 450000


def test_real_revocation_and_updated_embedding_cache(models):
    dense, cross = models
    store = corpus()
    retriever = Retriever(store, dense=dense, reranker=cross)
    original = store.documents["pay-error"]
    retriever.search("ERR-PAY-042", ALICE, AT, mode="dense")
    store.upsert(replace(original, revision=2, text="新的支付回调排查规则"))
    assert store.fetch(original.id, ALICE, AT).revision == 2
    retriever.search("新的支付规则", ALICE, AT, mode="dense")
    store.revoke(ALICE.principal, original.id)
    docs, trace = retriever.search("ERR-PAY-042", ALICE, AT, mode="rerank", timeout=30)
    assert original.id not in trace["union"] and original.id not in {d.id for d in docs}
    store.delete("states")
    assert "states" not in retriever.search("付款", ALICE, AT, mode="dense")[1]["union"]
    dense.clear_cache()
    assert not dense.cache


def test_real_weight_integrity_and_repeatability(models):
    from rag_lab.neural_experiment import verify_weights

    hashes = verify_weights(load_profile())
    assert hashes["embedding"]["bytes"] == 95827648
    assert hashes["reranker"]["bytes"] == 1112206140
    dense, cross = models
    docs = corpus().visible(ALICE, AT)
    a = dense.search("钱付了就代表饮料做好了吗", docs, k=4)
    b = dense.search("钱付了就代表饮料做好了吗", docs, k=4)
    assert [c.id for c in a] == [c.id for c in b]
    assert np.allclose([c.score for c in a], [c.score for c in b], atol=1e-6)
    assert str(next(dense.model.parameters()).dtype) == "torch.float32"
    assert str(next(cross.model.model.parameters()).dtype) == "torch.float32"
