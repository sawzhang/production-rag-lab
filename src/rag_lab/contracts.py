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
    topics: tuple[str, ...] = ()
    category: str | None = None
    brand: str | None = None
    capacity_gb: int | None = None
    style: str | None = None

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
    from .routing import infer_scope, normalized

    scope = infer_scope(query)
    text = normalized(query)
    constraints["retrieval_scope"] = {
        "topics": scope.topics,
        "category": scope.category,
        "brand": scope.brand,
        "capacity_operator": scope.capacity_operator,
        "capacity_values": scope.capacity_values,
        "extreme": scope.extreme,
        "exclude_style": scope.exclude_style,
    }
    if route != "deny_write":
        if scope.status == "clarification":
            missing.extend(scope.missing)
        if scope.topics and all(
            t in {"environment_identity", "dependency_safety"} for t in scope.topics
        ):
            route = "documents"
            missing = [x for x in missing if x != "environment"]
        elif (
            scope.category == "smartphone"
            and (scope.capacity_operator or scope.extreme or "comparison_logic" in scope.missing)
            and not any(term in text for term in ("多少钱", "会员价"))
        ):
            route = "catalogue"
            missing = [
                x
                for x in missing
                if x
                not in {"variant_id", "store", "channel", "unsupported_historical_or_filter_intent"}
            ]
            if any(x in text for x in ("去年", "历史", "昨天", "明天")):
                missing.append("historical_catalogue_source")
        if route == "price":
            constraints["currency"] = "USD" if any(x in text for x in ("美元", "usd")) else "CNY"
        if route == "cmdb":
            production = any(x in text for x in ("生产", "线上", "prod"))
            testing = any(x in text for x in ("测试", "沙箱", "test", "staging"))
            if production and testing:
                missing.append("environment_choice")
            elif production or testing:
                constraints["environment"] = "prod" if production else "test"
                missing = [x for x in missing if x != "environment"]
            aliases = {"菜单服务": "menu", "订单服务": "order", "支付服务": "payment"}
            service = next((v for alias, v in aliases.items() if alias in text), None)
            if service:
                constraints["service"] = service
            else:
                missing.append("service_identity")
        if route == "catalogue":
            price = re.search(
                r"(不高于|不超过|至多|至少|不低于|超过|大于|低于|小于)\s*(\d+(?:\.\d+)?)\s*(?:美元|元)",
                text,
            )
            if price:
                try:
                    constraints["maximum_cents"] = cents(price[2])
                    constraints["amount_operator"] = {
                        "不高于": "le",
                        "不超过": "le",
                        "至多": "le",
                        "至少": "ge",
                        "不低于": "ge",
                        "超过": "gt",
                        "大于": "gt",
                        "低于": "lt",
                        "小于": "lt",
                    }[price[1]]
                except ValueError:
                    missing.append("amount_precision")
            constraints["currency"] = "USD" if any(x in text for x in ("美元", "usd")) else "CNY"
    return Plan(query, route, ids, constraints, list(dict.fromkeys(missing)))


def protected_rewrite(original: str, rewritten: str) -> str:
    # Conservative: expansions must preserve original verbatim, protecting all constraints.
    if original not in rewritten:
        raise ValueError("rewrite changed protected original")
    return rewritten
