import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .contracts import Document, Identity


class Store:
    """Single-process teaching store. Ledger is never rolled back with corpus snapshots."""

    def __init__(self, documents=()):
        self.documents = {}
        self.deleted = set()
        self.revoked = set()
        self.generation = 0
        self.policy_epoch = 0
        self.auth_available = True
        for document in documents:
            self.upsert(document)

    def upsert(self, document: Document):
        if document.id in self.deleted:
            raise ValueError("tombstoned IDs cannot be resurrected")
        old = self.documents.get(document.id)
        if old and document.revision < old.revision:
            return False
        if old and document.revision == old.revision:
            if old != document:
                raise ValueError("same revision has conflicting content")
            return False
        self.documents[document.id] = document
        self.generation += 1
        return True

    def visible(self, identity: Identity, at: str):
        if not self.auth_available:
            raise PermissionError("authorization unavailable")
        return [
            d
            for d in self.documents.values()
            if d.id not in self.deleted
            and (identity.principal, d.id) not in self.revoked
            and d.visible(identity, at)
        ]

    def fetch(self, id: str, identity: Identity, at: str):
        return next((d for d in self.visible(identity, at) if d.id == id), None)

    def revoke(self, principal: str, id: str):
        self.revoked.add((principal, id))
        self.policy_epoch += 1

    def delete(self, id: str):
        self.deleted.add(id)
        self.documents.pop(id, None)
        self.generation += 1
        self.policy_epoch += 1

    def cache_key(self, query: str, identity: Identity, at: str, model: str):
        value = (
            query,
            identity.tenant,
            identity.principal,
            sorted(identity.groups),
            sorted(identity.stores),
            identity.policy_version,
            self.policy_epoch,
            self.generation,
            at,
            model,
        )
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()

    def snapshot(self, path: Path):
        rows = []
        for d in self.documents.values():
            row = asdict(d)
            row["groups"] = sorted(d.groups)
            rows.append(row)
        path.write_text(json.dumps(rows, ensure_ascii=False))

    def restore(self, path: Path):
        self.documents.clear()
        for row in json.loads(path.read_text()):
            row["groups"] = frozenset(row["groups"])
            row["entities"] = tuple(row["entities"])
            d = Document(**row)
            if d.id not in self.deleted:
                self.documents[d.id] = d
        self.generation += 1

    def save_ledger(self, path: Path):
        path.write_text(
            json.dumps(
                {
                    "deleted": sorted(self.deleted),
                    "revoked": sorted(self.revoked),
                    "epoch": self.policy_epoch,
                }
            )
        )

    def load_ledger(self, path: Path):
        data = json.loads(path.read_text())
        self.deleted.update(data["deleted"])
        self.revoked.update(tuple(x) for x in data["revoked"])
        self.policy_epoch = max(self.policy_epoch, data["epoch"]) + 1
        for id in self.deleted:
            self.documents.pop(id, None)

    def expand_parent(self, id: str, identity: Identity, at: str):
        child = self.fetch(id, identity, at)
        if not child:
            return None
        return self.fetch(child.parent, identity, at) if child.parent else child
