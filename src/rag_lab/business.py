from dataclasses import dataclass

from .contracts import Identity, instant


@dataclass(frozen=True)
class Variant:
    id: str
    category: str
    capacity: int | None
    price_cents: int
    style: str = "standard"
    currency: str = "CNY"


VARIANTS = [
    Variant("PHONE-A-128", "smartphone", 128, 400000),
    Variant("PHONE-A-256", "smartphone", 256, 500000),
    Variant("PHONE-B-256", "smartphone", 256, 499999),
    Variant("PHONE-P-256", "smartphone", 256, 600000, "Pro"),
    Variant("FRUIT-FUJI-1KG", "fruit", None, 1800),
]


class Gateway:
    """Synthetic read-only authority. Identity comes from the caller's trusted context."""

    allowed = frozenset({"quote", "order", "menu", "dependencies"})

    def __init__(self):
        self.price_available = True
        self.orders = {
            ("demo", "00001234"): {
                "owner": "alice",
                "store": "S01",
                "payment": "captured",
                "fulfillment": "pending",
                "refund": "accepted",
                "settlement_cents": 1800,
                "events": ["payment.captured", "order.pending"],
            }
        }
        self.menu_rows = [
            dict(
                poskey="001234",
                item="tea",
                store="S01",
                channel="到店",
                valid_from="2025-01-01T00:00:00+08:00",
                valid_to=None,
                enabled=True,
                stock=True,
            ),
            dict(
                poskey="001234",
                item="tea",
                store="S01",
                channel="外送",
                valid_from="2025-01-01T00:00:00+08:00",
                valid_to=None,
                enabled=False,
                stock=True,
            ),
        ]
        self.edges = [
            ("menu", "order", "prod"),
            ("menu", "pos", "prod"),
            ("menu", "sandbox", "test"),
            ("order", "payment", "prod"),
        ]

    def authorize_store(self, identity, store):
        if identity.tenant != "demo" or store not in identity.stores:
            raise PermissionError("object unavailable")

    def quote(
        self,
        identity: Identity,
        variant_id: str,
        store: str,
        channel: str,
        at: str,
        member=False,
        extras=0,
    ):
        self.authorize_store(identity, store)
        instant(at)
        if channel not in {"到店", "外送"} or not isinstance(member, bool):
            raise ValueError("invalid quote schema")
        if not isinstance(extras, int) or isinstance(extras, bool) or not 0 <= extras <= 3:
            raise ValueError("extras outside budget")
        if not self.price_available:
            raise TimeoutError("price API unavailable; no historical fallback")
        variant = next((x for x in VARIANTS if x.id == variant_id), None)
        if not variant:
            raise LookupError("exact variant not found")
        amount = variant.price_cents + (200 if channel == "外送" else 0) + extras * 100
        if member:
            amount = amount * 9 // 10
        return {
            "variant_id": variant.id,
            "amount_cents": amount,
            "currency": "CNY",
            "store": store,
            "channel": channel,
            "price_type": "member" if member else "list",
            "observed_at": at,
            "source": "synthetic-price-api/v1",
        }

    def order(self, identity: Identity, order_id: str):
        row = self.orders.get((identity.tenant, order_id))
        if not row or row["owner"] != identity.principal or row["store"] not in identity.stores:
            raise PermissionError("object unavailable")
        return {k: v for k, v in row.items() if k != "owner"}

    def menu(self, identity: Identity, poskey: str, store: str, channel: str, at: str):
        self.authorize_store(identity, store)
        if channel not in {"到店", "外送"}:
            raise ValueError("invalid channel")
        time = instant(at)
        rows = [
            r
            for r in self.menu_rows
            if r["poskey"] == poskey
            and r["store"] == store
            and r["channel"] == channel
            and instant(r["valid_from"]) <= time
            and (r["valid_to"] is None or time < instant(r["valid_to"]))
        ]
        if len(rows) != 1:
            raise LookupError("mapping absent or ambiguous")
        r = rows[0]
        return {
            "item": r["item"],
            "sellable": r["enabled"] and r["stock"],
            "reason": "channel_disabled"
            if not r["enabled"]
            else "out_of_stock"
            if not r["stock"]
            else "available",
            "source": "synthetic-menu-api/v1",
            "observed_at": at,
        }

    def dependencies(self, identity: Identity, service: str, environment: str, hops=1, limit=10):
        if identity.tenant != "demo" or "engineer" not in identity.groups:
            raise PermissionError("object unavailable")
        if environment not in {"prod", "test"} or not 1 <= hops <= 2 or not 1 <= limit <= 20:
            raise ValueError("graph budget or environment invalid")
        seen = {service}
        frontier = {service}
        result = []
        for _ in range(hops):
            following = set()
            for a, b, env in self.edges:
                if a in frontier and env == environment and b not in seen:
                    result.append(
                        {"consumer": b, "environment": env, "source": "synthetic-cmdb/v1"}
                    )
                    following.add(b)
                    seen.add(b)
                    if len(result) >= limit:
                        return {"edges": result, "complete": False, "registered_only": True}
            frontier = following
        return {"edges": result, "complete": True, "registered_only": True}

    def call(self, name, identity, **parameters):
        if name not in self.allowed:
            raise PermissionError("tool is not in read-only registry")
        return getattr(self, name)(identity, **parameters)


def filter_variants(capacity=None, maximum=None, operator="lt", exclude=None, currency="CNY"):
    if operator not in {"lt", "le"}:
        raise ValueError("unsupported operator")
    return [
        v
        for v in VARIANTS
        if v.category == "smartphone"
        and v.currency == currency
        and (capacity is None or v.capacity == capacity)
        and v.style != exclude
        and (
            maximum is None
            or (v.price_cents < maximum if operator == "lt" else v.price_cents <= maximum)
        )
    ]


def safe_menu_items(rows):
    # Unknown recipe or trace risk cannot be treated as explicitly peanut-free.
    return [r["id"] for r in rows if r.get("peanut_free") is True and r.get("trace_risk") is False]


def discount(amount_cents, quantity, member, stack_allowed):
    if amount_cents < 0 or quantity < 0:
        raise ValueError("negative amount or quantity")
    amount = amount_cents
    if quantity > 2 and amount >= 10000:
        amount -= 2000
        if not stack_allowed:
            return amount
    return amount * 9 // 10 if member else amount
