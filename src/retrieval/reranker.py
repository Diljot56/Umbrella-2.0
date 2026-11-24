from typing import List
from sentence_transformers import CrossEncoder
from src.models.search import SearchResultItem

# Score below which the reranker signals "no confident match".
# The generator checks this flag and emits the standard abstention message.
ABSTENTION_THRESHOLD = -5.0


class RerankerService:
    _instance = None

    def __new__(cls, model_name: str = "BAAI/bge-reranker-large"):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance.encoder = CrossEncoder(model_name)
        return cls._instance

    def rerank(
        self,
        query: str,
        candidates: List[SearchResultItem],
        top_k: int,
        max_rerank_pool: int = 15,
    ) -> List[SearchResultItem]:
        """Cross-encoder reranking with expanded token window and abstention gating.

        Changes vs original:
        - max_rerank_pool default raised to 15 (was 5) so diverse candidates
          from fuse_rrf() are actually scored by the cross-encoder.
        - max_length raised to 512 (was 192) for code / ticket text.
        - Sets item.low_confidence = True on all items when the top score
          falls below ABSTENTION_THRESHOLD, enabling the generator to refuse.
        """
        if not candidates:
            return []

        to_rerank = candidates[:max_rerank_pool]
        unranked = candidates[max_rerank_pool:]

        pairs = [[query, item.text] for item in to_rerank]
        scores = self.encoder.predict(pairs, max_length=512)

        # Handle shape edge-cases
        if isinstance(scores, float) or (hasattr(scores, "ndim") and scores.ndim == 0):
            scores = [scores]

        for i, item in enumerate(to_rerank):
            item.scores["cross_encoder_score"] = float(scores[i])

        for item in unranked:
            item.scores["cross_encoder_score"] = -999.0

        to_rerank.sort(key=lambda x: x.scores["cross_encoder_score"], reverse=True)
        reranked = (to_rerank + unranked)[:top_k]

        # ── Abstention gating ─────────────────────────────────────────────
        top_score = reranked[0].scores["cross_encoder_score"] if reranked else -999.0
        low_confidence = top_score < ABSTENTION_THRESHOLD
        for item in reranked:
            item.metadata["low_confidence"] = low_confidence  # type: ignore[index]

        return reranked
