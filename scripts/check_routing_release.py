"""Check recorded real-model evidence; does not rerun/download models or call an API."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
r = json.loads((root / "reports/routing-evaluation.json").read_text())
t = json.loads((root / "reports/routing-tasks.json").read_text())
g = json.loads((root / "reports/deepseek-routing-generation.json").read_text())
a = r["modes"]["constrained_cross_encoder"]
b = r["modes"]["neural_hybrid_cross_encoder"]
known = {d["id"]: d for d in a["details"]}
checks = {
    "input_fingerprints": all(
        hashlib.sha256((root / i["path"]).read_bytes()).hexdigest() == i["sha256"]
        for i in r["inputs"]
    ),
    "routing_code_fingerprints": all(
        hashlib.sha256((root / "src/rag_lab" / name).read_bytes()).hexdigest() == digest
        for name, digest in r["routing_source_sha256"].items()
    ),
    "expanded_retrieval_count": r["query_count"] == 120,
    "no_rerank_fallback": all(m["degraded_queries"] == 0 for m in r["modes"].values()),
    "no_per_query_ndcg_regression": all(
        x["metrics"]["ndcg"] + 1e-9 >= y["metrics"]["ndcg"]
        for x, y in zip(a["details"], b["details"])
    ),
    "all_required_groups_covered": a["macro"]["coverage"] == 1,
    "capacity_known_regression": known["N20"]["ranking"] == ["phone128"],
    "environment_known_regression": known["N10"]["ranking"] == ["graph-rule"],
    "menu_known_regression": {"menu-channel", "menu-contract"} <= set(known["N06"]["ranking"]),
    "menu_no_recipe_or_injection": not {"unknown", "injection"} & set(known["N06"]["scoped_ids"]),
    "new_validation_coverage": a["datasets"]["routing_validation_queries/holdout"]["coverage"] == 1,
    "task_contracts": t["passed"] == t["query_count"] == 24,
    "generation_contracts": g["passed"] == g["completed"] == 12,
}
report = {
    "scope": "recorded synthetic release evidence, not production acceptance",
    "passed": all(checks.values()),
    "checks": checks,
}
(root / "reports/routing-release-checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(1)
