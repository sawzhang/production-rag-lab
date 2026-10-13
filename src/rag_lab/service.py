import re
import time
from dataclasses import asdict

from .business import Gateway
from .contracts import parse
from .retrieval import Retriever


class Service:
    """Bounded router with extractive evidence output; no LLM calls by default."""

    def __init__(self, store, gateway=None, retriever=None):
        self.store = store
        self.gateway = gateway or Gateway()
        self.retriever = retriever or Retriever(store)

    def answer(self, query, identity, at, mode="hybrid"):
        start = time.perf_counter()
        self.store.visible(identity, at)  # Fail closed before any business or retrieval call.
        plan = parse(query)
        result = {
            "plan": asdict(plan),
            "status": "ok",
            "fact": None,
            "evidence": [],
            "trace": {"route": plan.route},
        }
        if plan.route == "deny_write":
            result["status"] = "denied"
        elif plan.missing:
            result["status"] = "clarification"
        elif plan.route in {"price", "order", "menu", "cmdb"}:
            try:
                if plan.route == "price":
                    id = next(x for x in plan.ids if x.startswith(("PHONE-", "FRUIT-")))
                    fact = self.gateway.quote(
                        identity,
                        id,
                        plan.constraints["store"],
                        "外送" if "外送" in query else "到店",
                        at,
                        member="会员" in query,
                    )
                elif plan.route == "order":
                    fact = self.gateway.order(identity, next(x for x in plan.ids if x.isdigit()))
                elif plan.route == "menu":
                    store = plan.constraints.get("store")
                    if not store or not any(x in query for x in ("到店", "外送")):
                        result["status"] = "clarification"
                        fact = None
                    else:
                        fact = self.gateway.menu(
                            identity,
                            next(x for x in plan.ids if x.isdigit()),
                            store,
                            "外送" if "外送" in query else "到店",
                            at,
                        )
                else:
                    fact = self.gateway.dependencies(
                        identity, "menu", "prod" if "生产" in query else "test"
                    )
                result["fact"] = fact
            except (TimeoutError, LookupError):
                result["status"] = "unavailable"
            except PermissionError:
                result["status"] = "denied"
        else:
            exact_ids = [x for x in plan.ids if x.startswith("ERR-")]
            if exact_ids and not any(
                x in d.title or x in d.text
                for x in exact_ids
                for d in self.store.visible(identity, at)
            ):
                result["status"] = "no_answer"
            else:
                documents, trace = self.retriever.search(query, identity, at, mode)
                result["trace"].update(trace)
                # Evidence is data only; no tool execution or arbitrary external URLs.
                result["evidence"] = [
                    {
                        "id": d.id,
                        "source": d.source,
                        "revision": d.revision,
                        "text": d.text,
                        "valid_from": d.valid_from,
                    }
                    for d in documents
                ]
                claims = {}
                for d in documents:
                    if d.claim:
                        claims.setdefault(d.claim, set()).add(d.value)
                if any(len(values) > 1 for values in claims.values()):
                    result["status"] = "conflict"
                elif not documents:
                    result["status"] = "no_answer"
        result["trace"]["elapsed_ms"] = (time.perf_counter() - start) * 1000
        return result


def validate_fact(draft, fact):
    """Validate structured generation fields before rendering, not prose similarity."""
    protected = {"variant_id", "amount_cents", "currency", "store", "channel", "observed_at"}
    if any(draft.get(key) != fact.get(key) for key in protected):
        raise ValueError("generated fact changed authority fields")
    return draft


def validate_citations(ids, evidence):
    if not set(ids) <= {d.id for d in evidence}:
        raise ValueError("unsupported citation")
    return True


def structural_chunks(text, max_chars=800):
    """Paragraph chunks: preserve entire tables/clauses, reject oversize instead of splitting."""
    if max_chars < 1:
        raise ValueError("invalid chunk budget")
    paragraphs = re.split(r"\n\s*\n", text.strip())
    if any(len(p) > max_chars for p in paragraphs):
        raise ValueError("oversize semantic block; manual structure required")
    return [p for p in paragraphs if p]
