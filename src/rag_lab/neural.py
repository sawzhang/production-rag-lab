"""Pinned real models; bounded in-memory vector caches contain authorized inputs only."""

import hashlib
from collections import OrderedDict

import numpy as np

from .contracts import Candidate


def pinned(revision):
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("pin a full model commit SHA")


class NeuralDense:
    def __init__(
        self,
        model_id,
        revision,
        device="cpu",
        query_prefix="",
        batch_size=8,
        max_length=512,
        cache_size=256,
        local_files_only=False,
    ):
        pinned(revision)
        if min(batch_size, max_length) < 1 or cache_size < 0:
            raise ValueError("invalid model budget")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_id,
            revision=revision,
            device=device,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.model.max_seq_length = max_length
        self.version = f"{model_id}@{revision}"
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def clear_cache(self):
        self.cache.clear()

    def search(self, query, documents, k=20):
        if k < 1:
            raise ValueError("k must be positive")
        if not documents:
            return []
        texts = [d.title + "\n" + d.text for d in documents]
        keys = [hashlib.sha256((self.version + text).encode()).hexdigest() for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        encoded_by_key = {}
        if missing:
            lookup = dict(zip(keys, texts))
            matrix = np.asarray(
                self.model.encode_document(
                    [lookup[key] for key in missing],
                    normalize_embeddings=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            )
            if matrix.ndim != 2 or len(matrix) != len(missing) or not np.isfinite(matrix).all():
                raise ValueError("invalid document vectors")
            encoded_by_key.update(zip(missing, matrix))
        matrix = np.stack(
            [self.cache[key] if key in self.cache else encoded_by_key[key] for key in keys]
        )
        for key, vector in zip(keys, matrix):
            if self.cache_size:
                self.cache[key] = vector
                self.cache.move_to_end(key)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        q = np.asarray(
            self.model.encode_query(
                [self.query_prefix + query],
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        )
        if q.shape != (1, matrix.shape[1]) or not np.isfinite(q).all():
            raise ValueError("invalid query vector or mismatched vector space")
        scores = matrix @ q.T
        return sorted(
            [
                Candidate(d.id, float(score), "neural_cosine")
                for d, score in zip(documents, scores.ravel())
            ],
            key=lambda c: (-c.score, c.id),
        )[:k]


class NeuralReranker:
    def __init__(
        self, model_id, revision, device="cpu", max_length=512, batch_size=8, local_files_only=False
    ):
        pinned(revision)
        if min(max_length, batch_size) < 1:
            raise ValueError("invalid reranker budget")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_id,
            revision=revision,
            device=device,
            max_length=max_length,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        self.version = f"{model_id}@{revision}"
        self.batch_size = batch_size

    def rank(self, query, documents):
        if not documents:
            return []
        scores = np.asarray(
            self.model.predict(
                [(query, d.title + "\n" + d.text) for d in documents],
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        ).reshape(-1)
        if len(scores) != len(documents) or not np.isfinite(scores).all():
            raise ValueError("invalid reranker scores")
        return sorted(
            [Candidate(d.id, float(s), "cross_encoder_score") for d, s in zip(documents, scores)],
            key=lambda c: (-c.score, c.id),
        )
