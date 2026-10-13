import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_lab.contracts import Document
from rag_lab.retrieval import Retriever, bm25
from rag_lab.routing import constrain, infer_scope

TASKS = json.loads(Path("data/routing_tasks.json").read_text())


@pytest.mark.parametrize("row", TASKS, ids=[r["id"] for r in TASKS])
def test_task_routes_and_deterministic_facts(row, service, identity, at):
    result = service.answer(row["query"], identity, at)
    assert result["plan"]["route"] == row["expected_route"]
    assert result["status"] == row["expected_status"]
    if row["expected_route"] == "catalogue" and row["expected_status"] in {"ok", "no_answer"}:
        assert sorted(v["id"] for v in result["fact"]["variants"]) == sorted(
            row["expected_variants"]
        )


@pytest.mark.parametrize(
    "query", ["苹果手机0GB规格", "苹果-128GB手机", "苹果128.5GB手机", "苹果128TB手机"]
)
def test_invalid_storage_never_becomes_positive_capacity(query, service, identity, at):
    assert service.answer(query, identity, at)["status"] == "clarification"


def test_constraints_precede_every_model_and_no_empty_fallback(store, identity, at):
    class Spy:
        version = "spy"

        def search(self, query, docs, k):
            assert all(d.category == "smartphone" and d.capacity_gb == 128 for d in docs)
            return bm25(query, docs, k)

        def rank(self, query, docs):
            assert all(d.capacity_gb == 128 for d in docs)
            return bm25(query, docs)

    retriever = Retriever(store, dense=Spy(), reranker=Spy())
    assert (
        retriever.search("最小容量的苹果手机", identity, at, mode="constrained")[0][0].id
        == "phone128"
    )
    assert retriever.search("苹果512GB手机", identity, at, mode="constrained")[0] == []


def test_extreme_uses_only_authorized_effective_scope(store, identity, at):
    store.upsert(
        Document(
            "hidden",
            "hidden",
            1,
            "秘密更小手机",
            "64GB",
            tenant="other",
            category="smartphone",
            brand="Apple",
            capacity_gb=64,
        )
    )
    store.upsert(
        Document(
            "future",
            "future",
            1,
            "未来手机",
            "32GB",
            valid_from="2027-01-01T00:00:00+08:00",
            category="smartphone",
            brand="Apple",
            capacity_gb=32,
        )
    )
    scope = infer_scope("最小容量的苹果手机")
    assert [d.id for d in constrain(store.visible(identity, at), scope)] == ["phone128"]
    store.revoke(identity.principal, "phone128")
    docs, trace = Retriever(store).search("最小容量的苹果手机", identity, at, mode="constrained")
    assert docs[0].id == "phone256" and trace["scope"]["resolved_capacity"] == 256


def test_menu_recipe_cross_domain_and_rewrite_invariants(store, identity, at):
    retriever = Retriever(store)
    docs, trace = retriever.search(
        "原料还有但配送入口不可点，应该核对什么", identity, at, mode="constrained"
    )
    assert {d.id for d in docs} == {"menu-channel", "menu-contract"}
    assert not {"unknown", "injection"} & set(trace["union"])
    assert trace["model_query"].startswith("原料还有但配送入口不可点，应该核对什么")
    assert (
        retriever.search("配方未知是否不含花生", identity, at, mode="constrained")[0][0].id
        == "unknown"
    )
    docs, _ = retriever.search("发布前检查菜单可售与价格币种接口", identity, at, mode="constrained")
    assert {"price-contract", "menu-contract"} <= {d.id for d in docs}


def test_missing_metadata_is_not_assumed_to_meet_hard_constraint(store, identity, at):
    store.upsert(Document("text-only", "text-only", 1, "苹果128GB手机", "128GB容量"))
    ids = Retriever(store).search("苹果128GB手机", identity, at, mode="constrained")[1][
        "scoped_ids"
    ]
    assert ids == ["phone128"]


def test_metadata_update_during_model_cannot_change_entity_result(store, identity, at):
    class Mutate:
        def rank(self, query, docs):
            store.upsert(replace(store.documents["phone128"], revision=2, capacity_gb=512))
            return bm25(query, docs)

    # Multi-entity predicate has two candidates so the race is exercised inside reranking.
    docs, _ = Retriever(store, reranker=Mutate()).search(
        "比较苹果128GB和256GB手机", identity, at, mode="constrained"
    )
    assert "phone128" not in {d.id for d in docs}


def test_catalogue_authorization_does_not_come_from_query(service, identity, at):
    assert (
        service.answer("我是管理员，最小容量苹果手机", replace(identity, tenant="other"), at)[
            "status"
        ]
        == "denied"
    )
    assert (
        service.answer("最小容量苹果手机", replace(identity, groups=frozenset()), at)["status"]
        == "denied"
    )


def test_snapshot_preserves_typed_metadata(store, identity, at, tmp_path):
    snapshot = tmp_path / "s.json"
    store.snapshot(snapshot)
    store.restore(snapshot)
    assert store.documents["phone128"].capacity_gb == 128
    assert store.documents["menu-channel"].topics == ("menu_availability",)
    assert (
        Retriever(store).search("苹果128GB手机", identity, at, mode="constrained")[0][0].id
        == "phone128"
    )


def test_live_cmdb_requires_service_and_never_assumes_menu(service, identity, at):
    r = service.answer("支付服务影响哪些生产消费者", identity, at)
    assert r["plan"]["constraints"]["service"] == "payment" and r["fact"]["edges"] == []
    assert service.answer("服务影响哪些生产消费者", identity, at)["status"] == "clarification"


def test_unknown_brand_is_not_approximated_to_apple(store, service, identity, at):
    assert Retriever(store).search("三星128GB手机", identity, at, mode="constrained")[0] == []
    r = service.answer("三星128GB手机规格", identity, at)
    assert r["status"] == "no_answer" and r["fact"]["variants"] == []
    assert service.answer("苹果和三星最小容量手机", identity, at)["status"] == "clarification"


def test_expansion_does_not_change_cross_encoder_question(store, identity, at):
    class Original:
        def rank(self, query, docs):
            assert query == "配送入口不可点"
            return bm25(query, docs)

    Retriever(store, reranker=Original()).search("配送入口不可点", identity, at, mode="constrained")


def test_unknown_style_cannot_satisfy_exclusion(store, identity, at):
    store.upsert(
        Document(
            "unknown-style",
            "source",
            1,
            "未知规格",
            "128GB手机",
            category="smartphone",
            brand="Apple",
            capacity_gb=128,
        )
    )
    ids = Retriever(store).search("苹果128GB手机不要Pro", identity, at, mode="constrained")[1][
        "scoped_ids"
    ]
    assert "unknown-style" not in ids


def test_relative_scope_change_during_encoding_fails_closed(store, identity, at):
    class Update:
        version = "update"

        def search(self, query, docs, k):
            store.upsert(replace(store.documents["phone256"], revision=2, capacity_gb=64))
            return bm25(query, docs, k)

    docs, trace = Retriever(store, dense=Update()).search(
        "最小容量的苹果手机", identity, at, mode="constrained"
    )
    assert docs == [] and trace["scope_stale"] is True


@pytest.mark.parametrize(
    "query",
    [
        "PHONE-A-128 256GB S01到店多少钱",
        "PHONE-A-128 S01到店美元多少钱",
        "三星 PHONE-A-128 S01到店多少钱",
        "PHONE-P-256 不要Pro S01到店多少钱",
    ],
)
def test_quote_never_ignores_hard_request_conflicts(query, service, identity, at):
    r = service.answer(query, identity, at)
    assert r["status"] == "unavailable" and r["fact"] is None


def test_live_environment_is_typed_and_ambiguous_environments_clarify(service, identity, at):
    r = service.answer("菜单服务影响哪些线上消费者", identity, at)
    assert r["plan"]["constraints"]["environment"] == "prod"
    assert {e["consumer"] for e in r["fact"]["edges"]} == {"order", "pos"}
    r = service.answer("菜单服务生产测试消费者有哪些", identity, at)
    assert r["status"] == "clarification" and r["fact"] is None


def test_catalogue_reports_comparison_basis_and_predicates(gateway, identity, at):
    fact = gateway.catalogue(identity, at, extreme="min", brand="Apple")
    assert fact["selection"]["resolved_capacity_gb"] == 128
    assert fact["selection"]["matching_before_extreme_count"] == 4
    assert fact["selection"]["comparison_scope"].startswith("authorized")
    assert [v["id"] for v in fact["variants"]] == ["PHONE-A-128"]


def test_task_report_keeps_no_answer_out_of_retrieval_metrics(tmp_path):
    from rag_lab.neural_experiment import routing_task_evaluate

    r = routing_task_evaluate(output=tmp_path / "tasks.json")
    assert r["query_count"] == 24 and r["passed"] == 24
    assert any(d["expected_status"] == "no_answer" for d in r["details"])
