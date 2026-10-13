import numpy as np

from .contracts import Candidate


class NeuralDense:
    """Optional real bi-encoder. Model ID and immutable revision are mandatory."""

    def __init__(self, model_id, revision, device="cpu"):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("pin a full model commit SHA")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_id, revision=revision, device=device, trust_remote_code=False
        )
        self.version = f"{model_id}@{revision}"

    def search(self, query, documents, k=20):
        if not documents:
            return []
        matrix = self.model.encode_document(
            [d.title + "\n" + d.text for d in documents], normalize_embeddings=True
        )
        q = self.model.encode_query([query], normalize_embeddings=True)
        scores = np.asarray(matrix) @ np.asarray(q).T
        return sorted(
            [
                Candidate(d.id, float(score), "neural_cosine")
                for d, score in zip(documents, scores.ravel())
            ],
            key=lambda c: (-c.score, c.id),
        )[:k]


class NeuralReranker:
    def __init__(self, model_id, revision, device="cpu", max_length=512):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("pin a full model commit SHA")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_id,
            revision=revision,
            device=device,
            max_length=max_length,
            trust_remote_code=False,
        )
        self.version = f"{model_id}@{revision}"

    def rank(self, query, documents):
        scores = self.model.predict([(query, d.title + "\n" + d.text) for d in documents])
        return sorted(
            [Candidate(d.id, float(s), "cross_encoder_score") for d, s in zip(documents, scores)],
            key=lambda c: (-c.score, c.id),
        )
