import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import asdict

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .contracts import Candidate
from .routing import VERSION, constrain, infer_scope, matches


def tokens(text):
    text = text.lower().translate(str.maketrans("顯賣罄", "显卖罄"))
    output = re.findall(r"[a-z0-9]+(?:[-_][a-z0-9]+)*", text)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        output.extend(run)
        output.extend(run[i : i + 2] for i in range(len(run) - 1))
    return output


def bm25(query, documents, k=20, k1=1.5, b=0.75):
    if k < 1 or k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("invalid BM25 parameters")
    if not documents:
        return []
    bags = [Counter(tokens(d.title + " " + d.text)) for d in documents]
    lengths = [sum(x.values()) for x in bags]
    average = sum(lengths) / len(bags) or 1
    scores = []
    for d, bag, length in zip(documents, bags, lengths):
        score = 0.0
        for token in set(tokens(query)):
            df = sum(token in x for x in bags)
            tf = bag[token]
            idf = math.log(1 + (len(bags) - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
        if score > 0:
            scores.append(Candidate(d.id, score, "bm25"))
    return sorted(scores, key=lambda c: (-c.score, c.id))[:k]


class LSADense:
    """Actual latent dense vectors, NOT neural embeddings or an ANN index.

    Fit on the authorized collection only; no shared IDF across tenants.
    """

    version = "lsa-tfidf-v1"

    def search(self, query, documents, k=20):
        if not documents:
            return []
        vectorizer = TfidfVectorizer(analyzer=tokens)
        try:
            matrix = vectorizer.fit_transform([d.title + " " + d.text for d in documents])
        except ValueError:
            return []
        q = vectorizer.transform([query])
        dimensions = min(16, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if dimensions >= 1:
            svd = TruncatedSVD(n_components=dimensions, random_state=42)
            dense = normalize(svd.fit_transform(matrix))
            encoded = normalize(svd.transform(q))
            scores = (dense @ encoded.T).ravel()
        else:
            scores = (matrix @ q.T).toarray().ravel()
        return sorted(
            [
                Candidate(d.id, float(s), "lsa_cosine")
                for d, s in zip(documents, scores)
                if s > 1e-9
            ],
            key=lambda c: (-c.score, c.id),
        )[:k]


def rrf(rankings, c=60, weights=None):
    if c < 0:
        raise ValueError("negative RRF constant")
    weights = weights if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(w < 0 for w in weights):
        raise ValueError("invalid channel weights")
    scores = Counter()
    for ranking, weight in zip(rankings, weights):
        seen = set()
        for rank, item in enumerate(ranking, 1):
            if item.id not in seen:
                scores[item.id] += weight / (c + rank)
                seen.add(item.id)
    return sorted(
        [Candidate(id, score, "rrf") for id, score in scores.items()],
        key=lambda x: (-x.score, x.id),
    )


def weighted_fusion(rankings, alpha=0.5):
    if len(rankings) != 2 or not 0 <= alpha <= 1:
        raise ValueError("requires two channels and alpha in [0,1]")
    scores = Counter()
    for ranking, weight in zip(rankings, (alpha, 1 - alpha)):
        if not ranking or weight == 0:
            continue
        low, high = min(x.score for x in ranking), max(x.score for x in ranking)
        for item in ranking:
            scores[item.id] += weight * ((item.score - low) / (high - low) if high > low else 1.0)
    return sorted(
        [Candidate(id, score, "weighted") for id, score in scores.items()],
        key=lambda x: (-x.score, x.id),
    )


class OverlapReranker:
    """Deterministic teaching baseline; not a CrossEncoder."""

    version = "overlap-v1"

    def rank(self, query, documents):
        query_tokens = set(tokens(query))
        return sorted(
            [
                Candidate(d.id, len(query_tokens & set(tokens(d.title + d.text))), "token_overlap")
                for d in documents
            ],
            key=lambda x: (-x.score, x.id),
        )


def bounded_rerank(reranker, query, documents, timeout):
    # Shutdown does not wait on a slow worker. Python cannot kill the worker;
    # production HTTP/GPU adapters must enforce cancellation at their boundary.
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(reranker.rank, query, documents)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        raise TimeoutError("reranker deadline") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


class Retriever:
    def __init__(self, store, dense=None, reranker=None):
        self.store = store
        self.dense = dense or LSADense()
        self.reranker = reranker or OverlapReranker()

    def search(
        self, query, identity, at, mode="hybrid", window=20, budget=20, top_k=4, timeout=0.2
    ):
        if mode not in {"bm25", "dense", "hybrid", "weighted", "rerank", "constrained"}:
            raise ValueError("unknown retrieval mode")
        if min(window, budget, top_k) < 1 or timeout <= 0:
            raise ValueError("invalid budget")
        generation = self.store.generation
        allowed = self.store.visible(identity, at)
        scope = infer_scope(query) if mode == "constrained" else None
        scoped_query = query
        authorized_count = len(allowed)
        if scope:
            allowed = constrain(allowed, scope)
            if scope.expansion:
                scoped_query = query + " " + scope.expansion
        scoped_ids = {d.id for d in allowed}
        lexical = bm25(scoped_query, allowed, window) if mode != "dense" else []
        dense = self.dense.search(scoped_query, allowed, window) if mode != "bm25" else []
        union = sorted({c.id for c in lexical + dense})
        if mode == "bm25":
            fused = lexical
        elif mode == "dense":
            fused = dense
        elif mode == "weighted":
            fused = weighted_fusion([lexical, dense])
        else:
            fused = rrf([lexical, dense])
        shortlist = fused[:budget]
        documents = [
            d
            for c in shortlist
            if (d := self.store.fetch(c.id, identity, at))
            and (not scope or (d.id in scoped_ids and matches(d, scope)))
        ]
        degraded = False
        if mode in {"rerank", "constrained"} and len(documents) > 1:
            try:
                ranked = bounded_rerank(self.reranker, query, documents, timeout)
                ids = {d.id for d in documents}
                if any(c.id not in ids for c in ranked):
                    raise ValueError("reranker returned an unknown candidate")
                shortlist = ranked
            except (TimeoutError, RuntimeError):
                degraded = True
        evidence = []
        for c in shortlist:
            d = self.store.fetch(c.id, identity, at)
            if (
                d
                and (not scope or (d.id in scoped_ids and matches(d, scope)))
                and d.id not in {x.id for x in evidence}
            ):
                evidence.append(d)
            if len(evidence) >= top_k:
                break
        scope_stale = bool(scope and scope.extreme and self.store.generation != generation)
        if scope_stale:
            evidence = []
        trace = {
            "scope": asdict(scope) if scope else None,
            "scope_version": VERSION if scope else None,
            "scope_stale": scope_stale,
            "authorized_count": authorized_count,
            "scoped_ids": sorted(scoped_ids),
            "model_query": scoped_query,
            "rerank_query": query,
            "lexical": [c.id for c in lexical],
            "dense": [c.id for c in dense],
            "union": union,
            "fused": [c.id for c in fused],
            "shortlist": [c.id for c in shortlist],
            "degraded": degraded,
            "generation": self.store.generation,
            "model": self.dense.version,
            "reranker": self.reranker.version
            if hasattr(self.reranker, "version")
            else type(self.reranker).__name__,
        }
        return evidence, trace
