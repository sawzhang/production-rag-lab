import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


def instant(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return result


def cents(value: str) -> int:
    amount = Decimal(value) * 100
    if not amount.is_finite() or amount != amount.to_integral_value():
        raise ValueError("amount requires finite integer minor units")
    return int(amount)


@dataclass(frozen=True)
class Identity:
    tenant: str
    principal: str
    groups: frozenset[str] = frozenset()
    stores: frozenset[str] = frozenset({"S01"})
    policy_version: int = 1


@dataclass(frozen=True)
class Document:
    id: str
    source: str
    revision: int
    title: str
    text: str
    tenant: str = "demo"
    groups: frozenset[str] = frozenset({"staff"})
    valid_from: str = "2025-01-01T00:00:00+08:00"
    valid_to: str | None = None
    entities: tuple[str, ...] = ()
    parent: str | None = None
    claim: str | None = None
    value: str | None = None
    approved: bool = True

    def visible(self, identity: Identity, at: str) -> bool:
        time = instant(at)
        return (
            self.tenant == identity.tenant
            and bool(self.groups & identity.groups)
            and self.approved
            and instant(self.valid_from) <= time
            and (self.valid_to is None or time < instant(self.valid_to))
        )


@dataclass(frozen=True)
class Candidate:
    id: str
    score: float
    score_type: str


@dataclass
class Plan:
    original: str
    route: str
    ids: list[str] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def parse(query: str) -> Plan:
    ids = re.findall(
        r"(?:PHONE-[A-Z]-\d+|FRUIT-[A-Z]+-\d+KG|ERR-[A-Z]+-\d+|svc-[a-z]+-v\d+|\b0\d{5,}\b)", query
    )
    route = "documents"
    missing = []
    if any(x in query for x in ("退款并", "重启服务", "导出到", "修改价格")):
        route = "deny_write"
    elif "订单" in query and any(x in query for x in ("现在", "到哪", "事件")):
        route = "order"
        if not any(x.isdigit() for x in ids):
            missing.append("order_id")
    elif "多少钱" in query or "会员价" in query or "低于" in query:
        route = "price"
        if not any(x.startswith(("PHONE-", "FRUIT-")) for x in ids):
            missing.append("variant_id")
        if not re.search(r"S\d{2}", query):
            missing.append("store")
        if not any(x in query for x in ("到店", "外送")):
            missing.append("channel")
    elif "POSKEY" in query:
        route = "menu"
        if not any(x.isdigit() for x in ids):
            missing.append("poskey")
    elif "消费者" in query or "依赖" in query:
        route = "cmdb"
        if "生产" not in query and "测试" not in query:
            missing.append("environment")
    if route == "price" and len([x for x in ids if x.startswith(("PHONE-", "FRUIT-"))]) > 1:
        missing.append("single_variant_id")
    if route == "price" and any(
        x in query for x in ("去年", "历史", "昨", "明天", "低于", "不高于")
    ):
        missing.append("unsupported_historical_or_filter_intent")
    constraints = {}
    store = re.search(r"S\d{2}", query)
    if store:
        constraints["store"] = store.group()
    if "不要 Pro" in query:
        constraints["exclude"] = "Pro"
    if "不高于" in query:
        constraints["operator"] = "le"
    elif "低于" in query:
        constraints["operator"] = "lt"
    return Plan(query, route, ids, constraints, missing)


def protected_rewrite(original: str, rewritten: str) -> str:
    # Conservative: expansions must preserve original verbatim, protecting all constraints.
    if original not in rewritten:
        raise ValueError("rewrite changed protected original")
    return rewritten
