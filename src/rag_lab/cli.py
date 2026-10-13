import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .fixtures import ALICE, AT, corpus
from .metrics import capacity, cost, macro, retrieval_metrics
from .retrieval import Retriever
from .service import Service


def emit(value, output=None):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(text + "\n")
    else:
        print(text)


def evaluate(path, split, window=20, budget=20, top_k=4):
    rows = [r for r in json.loads(Path(path).read_text()) if split == "all" or r["split"] == split]
    if not rows:
        raise ValueError("empty evaluation split")
    store = corpus()
    retriever = Retriever(store)
    result = {
        "configuration": {
            "window": window,
            "budget": budget,
            "top_k": top_k,
            "dense": "LSA statistical vectors",
            "reranker": "token overlap",
            "split": split,
        },
        "query_count": len(rows),
        "modes": {},
    }
    for mode in ("bm25", "dense", "hybrid", "weighted", "rerank"):
        details = []
        for row in rows:
            visible = {d.id for d in store.visible(ALICE, row["at"])}
            if not set(row["gold"]) <= visible:
                raise ValueError("gold is not legal for evaluation identity/time")
            started = time.perf_counter()
            docs, trace = retriever.search(
                row["query"], ALICE, row["at"], mode, window, budget, top_k
            )
            ranking = [d.id for d in docs]
            scores = retrieval_metrics(ranking, row["gold"], top_k, row["required_groups"])
            details.append(
                {
                    "id": row["id"],
                    "slice": row["slice"],
                    "ranking": ranking,
                    "metrics": scores,
                    "union_recall": retrieval_metrics(
                        trace["union"], row["gold"], len(trace["union"]) or 1
                    )["recall"],
                    "fused_recall": retrieval_metrics(trace["fused"], row["gold"], budget)[
                        "recall"
                    ],
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                }
            )
        result["modes"][mode] = {
            "macro": macro([d["metrics"] for d in details]),
            "slices": {
                s: {
                    "n": sum(d["slice"] == s for d in details),
                    **macro([d["metrics"] for d in details if d["slice"] == s]),
                }
                for s in sorted({d["slice"] for d in details})
            },
            "details": details,
        }
    return result


def benchmark(requests, workers):
    if requests < 1 or workers < 1:
        raise ValueError("positive load required")
    service = Service(corpus())

    def run(i):
        started = time.perf_counter()
        try:
            if i % 2:
                service.answer("PHONE-A-256 S01 到店多少钱", ALICE, AT)
            else:
                service.answer("支付回调超时", ALICE, AT)
            return (time.perf_counter() - started) * 1000, False
        except Exception:
            return (time.perf_counter() - started) * 1000, True

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, range(requests)))
    elapsed = time.perf_counter() - started
    times = [x[0] for x in rows]
    return {
        "scope": "local synthetic in-process mixed requests, no network or neural GPU",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requests": requests,
        "workers": workers,
        "errors": sum(x[1] for x in rows),
        "elapsed_seconds": elapsed,
        "throughput_rps": requests / elapsed,
        "latency_ms": {f"p{q}": float(np.percentile(times, q)) for q in (50, 95, 99)},
    }


def main():
    parser = argparse.ArgumentParser(description="企业 RAG 学习实验室（合成数据）")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument(
        "--mode",
        default="constrained",
        choices=["bm25", "dense", "hybrid", "weighted", "rerank", "constrained"],
    )
    for name in ("evaluate", "sweep"):
        command = sub.add_parser(name)
        command.add_argument("--data", default="data/retrieval_queries.json")
        command.add_argument("--split", choices=["dev", "holdout", "all"], default="holdout")
        command.add_argument("--output")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--requests", type=int, default=200)
    bench.add_argument("--workers", type=int, default=8)
    bench.add_argument("--output")
    real = sub.add_parser("neural-evaluate", help="Run pinned real embedding and CrossEncoder")
    real.add_argument(
        "--data",
        nargs="+",
        default=["data/retrieval_queries.json", "data/neural_challenge_queries.json"],
    )
    real.add_argument("--profile")
    real.add_argument("--include-constrained", action="store_true")
    real.add_argument("--output", default="reports/neural-evaluation.json")
    real.add_argument("--local-files-only", action="store_true")
    real_ask = sub.add_parser("neural-ask")
    real_ask.add_argument("query")
    real_ask.add_argument("--profile")
    real_ask.add_argument("--local-files-only", action="store_true")
    generation = sub.add_parser("generation-evaluate")
    generation.add_argument("--pi-auth", help="Explicit Pi auth.json; never saved in results")
    generation.add_argument("--profile")
    generation.add_argument("--optimized", action="store_true")
    generation.add_argument("--output", default="reports/deepseek-generation.json")
    sub.add_parser("routing-tasks")
    sub.add_parser("capacity")
    args = parser.parse_args()
    if args.command == "routing-tasks":
        from .neural_experiment import routing_task_evaluate

        emit(routing_task_evaluate())
    elif args.command == "generation-evaluate":
        from .neural_experiment import generation_evaluate

        result = generation_evaluate(args.pi_auth, args.output, args.profile, args.optimized)
        emit({"output": args.output, "completed": result["completed"], "passed": result["passed"]})
    elif args.command == "neural-evaluate":
        from .neural_experiment import neural_evaluate

        result = neural_evaluate(
            args.data, args.output, args.profile, args.local_files_only, args.include_constrained
        )
        emit(
            {
                "output": args.output,
                "query_count": result["query_count"],
                "metrics": {name: mode["macro"] for name, mode in result["modes"].items()},
            }
        )
    elif args.command == "neural-ask":
        from .neural_experiment import load_models, load_profile

        profile = load_profile(args.profile)
        dense, cross = load_models(profile, args.local_files_only)
        store = corpus()
        service = Service(store, retriever=Retriever(store, dense=dense, reranker=cross))
        emit(
            service.answer(
                args.query,
                ALICE,
                AT,
                mode="constrained",
                rerank_timeout=profile["rerank_timeout_seconds"],
            )
        )
    elif args.command == "ask":
        emit(Service(corpus()).answer(args.query, ALICE, AT, args.mode))
    elif args.command == "evaluate":
        emit(evaluate(args.data, args.split), args.output)
    elif args.command == "sweep":
        if args.split != "dev":
            parser.error("parameter search is restricted to dev; evaluate holdout only once")
        emit(
            [
                evaluate(args.data, "dev", window, budget, 4)
                for window in (4, 8, 20)
                for budget in (4, 8)
            ],
            args.output,
        )
    elif args.command == "benchmark":
        emit(benchmark(args.requests, args.workers), args.output)
    else:
        emit(
            {
                "raw_vectors": capacity(2_000_000, 1536),
                "two_copies": capacity(2_000_000, 1536, copies=2),
                "hypothetical_cost": cost(4000, 600, 10, 40, extra=0.02, success_rate=0.7),
            }
        )
