import math


def retrieval_metrics(ranking, gold, k=4, required_groups=()):
    if k < 1:
        raise ValueError("k must be positive")
    # Deduplicate to prevent duplicate chunks from inflating recall or DCG.
    ranking = list(dict.fromkeys(ranking))[:k]
    positives = {id for id, relevance in gold.items() if relevance > 0}
    if not positives:
        return {"recall": None, "hit_rate": None, "mrr": None, "ndcg": None, "coverage": None}
    hits = positives & set(ranking)
    dcg = sum(
        (2 ** gold.get(id, 0) - 1) / math.log2(rank + 1) for rank, id in enumerate(ranking, 1)
    )
    ideal = sorted((gold[id] for id in positives), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    first = next((1 / rank for rank, id in enumerate(ranking, 1) if id in positives), 0)
    coverage = (
        (sum(bool(set(group) & set(ranking)) for group in required_groups) / len(required_groups))
        if required_groups
        else None
    )
    return {
        "recall": len(hits) / len(positives),
        "hit_rate": float(bool(hits)),
        "mrr": first,
        "ndcg": dcg / idcg,
        "coverage": coverage,
    }


def ann_recall(approximate, exact):
    exact = set(exact)
    return len(set(approximate) & exact) / len(exact) if exact else None


def macro(rows):
    keys = ("recall", "hit_rate", "mrr", "ndcg", "coverage")
    result = {}
    for key in keys:
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = sum(values) / len(values) if values else None
    return result


def capacity(chunks, dimensions, copies=1, bytes_per_dimension=4):
    if min(chunks, dimensions, copies, bytes_per_dimension) < 0:
        raise ValueError("negative capacity input")
    size = chunks * dimensions * copies * bytes_per_dimension
    return {"bytes": size, "GB": size / 10**9, "GiB": size / 2**30}


def cost(
    input_tokens, output_tokens, input_per_million, output_per_million, extra=0.0, success_rate=1.0
):
    if (
        not 0 < success_rate <= 1
        or min(input_tokens, output_tokens, input_per_million, output_per_million, extra) < 0
    ):
        raise ValueError("invalid cost input")
    attempt = (input_tokens * input_per_million + output_tokens * output_per_million) / 1e6 + extra
    return {"per_attempt": attempt, "per_success": attempt / success_rate}
