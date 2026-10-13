import math
from dataclasses import replace

import pytest

from rag_lab.contracts import Candidate, cents, instant, protected_rewrite
from rag_lab.metrics import ann_recall, capacity, cost, macro, retrieval_metrics
from rag_lab.retrieval import Retriever, bm25, rrf, tokens, weighted_fusion
from rag_lab.service import structural_chunks, validate_citations, validate_fact


def test_metric_known_values():
    r = retrieval_metrics(["x", "a", "b"], {"a": 2, "b": 1}, k=2)
    assert r["recall"] == 0.5 and r["mrr"] == 0.5 and r["hit_rate"] == 1
    assert r["ndcg"] == pytest.approx((3 / math.log2(3)) / (3 + 1 / math.log2(3)))
    assert retrieval_metrics([], {})["recall"] is None
    assert retrieval_metrics(["a", "a"], {"a": 1, "b": 1})["recall"] == 0.5
    assert ann_recall(["a"], ["a", "b"]) == 0.5
    assert ann_recall([], []) is None
    assert macro([retrieval_metrics([], {})])["recall"] is None


def test_rrf_known_rank_math():
    a, b = Candidate("a", 999, "bm25"), Candidate("b", 0.2, "cosine")
    r = rrf([[a, b], [b]], c=60)
    assert r[0].id == "b" and r[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert rrf([[a, a]])[0].score == pytest.approx(1 / 61)
    with pytest.raises(ValueError):
        rrf([[a]], weights=[])
    assert weighted_fusion([[a], [b]], alpha=0)[0].id == "b"


@pytest.mark.parametrize("value,expected", [("0.01", 1), ("-0.10", -10), ("5000", 500000)])
def test_integer_minor_units(value, expected):
    assert cents(value) == expected


@pytest.mark.parametrize("value", ["0.001", "NaN", "Infinity"])
def test_invalid_amount(value):
    with pytest.raises(ValueError):
        cents(value)


def test_time_and_rewrite():
    with pytest.raises(ValueError):
        instant("2026-01-01")
    with pytest.raises(ValueError):
        protected_rewrite("订单 00001234 不要 Pro", "订单 1234 要 Pro")
    assert protected_rewrite("订单 00001234", "订单 00001234 事件线")
    assert "err-pay-042" in tokens("ERR-PAY-042")


def test_ledger_survives_new_process_restore(store, identity, at, tmp_path):
    from rag_lab.store import Store

    snapshot, ledger = tmp_path / "s.json", tmp_path / "l.json"
    store.snapshot(snapshot)
    store.delete("phone128")
    store.revoke("alice", "phone256")
    store.save_ledger(ledger)
    fresh = Store()
    fresh.load_ledger(ledger)
    fresh.restore(snapshot)
    assert fresh.fetch("phone128", identity, at) is None
    assert fresh.fetch("phone256", identity, at) is None


def test_idempotency_and_out_of_order(store):
    d = store.documents["phone128"]
    assert store.upsert(d) is False
    assert store.upsert(replace(d, revision=2, text="new")) is True
    assert store.upsert(d) is False
    with pytest.raises(ValueError):
        store.upsert(replace(d, revision=2, text="conflicting"))
    store.delete(d.id)
    with pytest.raises(ValueError):
        store.upsert(d)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "weighted", "rerank"])
def test_acl_before_external_models(store, identity, at, mode):
    class Spy:
        version = "spy"

        def search(self, query, documents, k):
            assert all(d.tenant == "demo" for d in documents)
            assert all("SECRET" not in d.text for d in documents)
            return bm25(query, documents, k)

        def rank(self, query, documents):
            assert all("SECRET" not in d.text for d in documents)
            return bm25(query, documents)

    docs, trace = Retriever(store, Spy(), Spy()).search("支付手机", identity, at, mode=mode)
    assert "other-tenant" not in trace["union"]
    assert all("SECRET" not in d.text for d in docs)


def test_zero_scope_and_auth_outage(store, identity, at, service):
    assert Retriever(store).search("手机", replace(identity, groups=frozenset()), at)[0] == []
    store.auth_available = False
    with pytest.raises(PermissionError):
        service.answer("PHONE-A-256 S01 到店多少钱", identity, at)


def test_revocation_during_rerank(store, identity, at):
    class Revoke:
        def rank(self, query, documents):
            for d in documents:
                store.revoke(identity.principal, d.id)
            return [Candidate(d.id, 1, "spy") for d in documents]

    assert (
        Retriever(store, reranker=Revoke()).search("支付订单", identity, at, mode="rerank")[0] == []
    )


def test_model_hallucinated_candidate_rejected(store, identity, at):
    class Bad:
        def rank(self, query, documents):
            return [Candidate("other-tenant", 999, "spy")]

    with pytest.raises(ValueError):
        Retriever(store, reranker=Bad()).search("支付订单", identity, at, mode="rerank")


def test_generation_and_identity_cache_separation(store, identity, at):
    a = store.cache_key("q", identity, at, "v1")
    assert a != store.cache_key("q", replace(identity, principal="bob"), at, "v1")
    assert a != store.cache_key("q", identity, at, "v2")
    store.upsert(replace(store.documents["phone128"], revision=2))
    assert a != store.cache_key("q", identity, at, "v1")


def test_fact_and_citation_validation(gateway, identity, at, store):
    fact = gateway.quote(identity, "PHONE-A-128", "S01", "到店", at)
    assert validate_fact(dict(fact), fact) == fact
    with pytest.raises(ValueError):
        validate_fact(dict(fact, amount_cents=1), fact)
    with pytest.raises(ValueError):
        validate_citations(["invented"], [store.documents["phone128"]])
    assert validate_citations(["phone128"], [store.documents["phone128"]])


def test_semantic_chunks():
    table = "SKU | 金额 | 单位\nA | 100 | 分\n仅适用 S01，不适用加盟店"
    assert structural_chunks("规则\n\n" + table) == ["规则", table]
    with pytest.raises(ValueError):
        structural_chunks(table, max_chars=5)


def test_capacity_and_success_cost():
    assert capacity(2_000_000, 1536)["GB"] == 12.288
    assert capacity(2_000_000, 1536, copies=2)["GB"] == 24.576
    assert cost(4000, 600, 10, 40)["per_attempt"] == 0.064
    assert cost(4000, 600, 10, 40, success_rate=0.7)["per_success"] == pytest.approx(0.064 / 0.7)


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"window": -1}, {"mode": "unknown"}])
def test_invalid_retrieval_budget(store, identity, at, kwargs):
    with pytest.raises(ValueError):
        Retriever(store).search("q", identity, at, **kwargs)


def test_graph_fanout(gateway, identity):
    assert gateway.dependencies(identity, "menu", "prod", limit=1)["complete"] is False
    assert len(gateway.dependencies(identity, "menu", "prod", hops=2)["edges"]) == 3
    with pytest.raises(ValueError):
        gateway.dependencies(identity, "menu", "prod", hops=3)


def test_invalid_quote_schema(gateway, identity, at):
    for bad in (-1, 4, True):
        with pytest.raises(ValueError):
            gateway.quote(identity, "PHONE-A-128", "S01", "到店", at, extras=bad)


def test_single_channel_does_not_call_unused_model(store, identity, at):
    class Forbidden:
        version = "forbidden"

        def search(self, *args):
            raise AssertionError("BM25 ablation should not invoke dense")

    assert Retriever(store, dense=Forbidden()).search("手机", identity, at, mode="bm25")[0]


def test_historical_and_multi_variant_price_clarifies(service, identity, at):
    for query in ("PHONE-A-256 S01 到店去年多少钱", "PHONE-A-256 PHONE-A-128 S01 到店多少钱"):
        assert service.answer(query, identity, at)["status"] == "clarification"


def test_service_menu_and_cmdb_routes(service, identity, at):
    assert service.answer("POSKEY 001234 S01 外送", identity, at)["fact"]["sellable"] is False
    assert service.answer("POSKEY 001234", identity, at)["status"] == "clarification"
    assert service.answer("菜单服务影响哪些生产消费者", identity, at)["fact"]["edges"]
    assert service.answer("菜单服务影响哪些消费者", identity, at)["status"] == "clarification"
    assert service.answer("zxqv-unseen-token", identity, at)["status"] == "no_answer"


def test_empty_and_single_document_dense(store, identity, at):
    from rag_lab.contracts import Document
    from rag_lab.retrieval import LSADense

    dense = LSADense()
    assert dense.search("query", []) == []
    assert dense.search("x", [Document("a", "a", 1, "", "")]) == []
    assert dense.search("word", [Document("a", "a", 1, "word", "")])[0].id == "a"


def test_numeric_and_budget_validation(gateway, identity, at):
    from rag_lab.business import discount, filter_variants

    for operation in (
        lambda: retrieval_metrics([], {}, k=0),
        lambda: capacity(-1, 1),
        lambda: cost(1, 1, 1, 1, success_rate=0),
        lambda: filter_variants(operator="approx"),
        lambda: discount(-1, 1, False, False),
        lambda: rrf([], c=-1),
        lambda: weighted_fusion([], alpha=0.5),
        lambda: bm25("x", [], k=0),
        lambda: gateway.quote(identity, "PHONE-A-128", "S01", "invalid", at),
    ):
        with pytest.raises(ValueError):
            operation()
