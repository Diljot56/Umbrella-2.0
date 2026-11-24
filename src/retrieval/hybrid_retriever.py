import string
from typing import List, Tuple, Dict
import numpy as np
from src.indexing.indexer import DualIndexerService
from src.models.chunk import Chunk
from src.models.search import SearchResultItem

# Maximum chunks per unique source_id allowed in the fused candidate pool.
# This prevents any single large document (e.g. configure_runners.md) from
# monopolizing all slots before the cross-encoder sees diverse candidates.
SOURCE_CAP = 3


class HybridRetriever:
    def __init__(self, indexer: DualIndexerService):
        self.indexer = indexer

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def search_dense(self, query: str, top_n: int) -> List[Tuple[Chunk, float, int]]:
        query_vector = self.indexer.encoder.encode([query], normalize_embeddings=True)[0]
        results = self.indexer.qdrant.query_points(
            collection_name=self.indexer.collection_name,
            query=query_vector.tolist(),
            limit=top_n
        )

        dense_results = []
        for i, hit in enumerate(results.points):
            chunk = Chunk(**hit.payload)
            dense_results.append((chunk, hit.score, i + 1))  # 1-based ranking

        return dense_results

    def search_sparse(self, query: str, top_n: int) -> List[Tuple[Chunk, float, int]]:
        if not self.indexer.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.indexer.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_n]

        sparse_results = []
        rank = 1
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue

            chunk_id = self.indexer.idx_to_chunk_id[idx]
            chunk = self.indexer.chunk_store[chunk_id]
            sparse_results.append((chunk, float(score), rank))
            rank += 1

        return sparse_results

    def fuse_rrf(self, dense_results: List[Tuple[Chunk, float, int]],
                 sparse_results: List[Tuple[Chunk, float, int]],
                 k: int = 60) -> List[SearchResultItem]:
        """Reciprocal Rank Fusion with a per-source diversity cap.

        After computing RRF scores for every candidate, the final sorted list
        is trimmed so that no single source_id occupies more than SOURCE_CAP
        slots in the candidate pool handed to the cross-encoder reranker.
        This prevents large documents from monopolizing top-K positions.
        """
        combined_scores: Dict[str, SearchResultItem] = {}

        def _get_or_create_item(chunk: Chunk) -> SearchResultItem:
            if chunk.chunk_id not in combined_scores:
                combined_scores[chunk.chunk_id] = SearchResultItem(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_id=chunk.source_id,
                    source_type=chunk.source_type,
                    location=chunk.location,
                    metadata=chunk.metadata,
                    scores={
                        "dense_score": None,
                        "dense_rank": None,
                        "bm25_score": None,
                        "bm25_rank": None,
                        "rrf_score": 0.0,
                        "cross_encoder_score": 0.0,
                    }
                )
            return combined_scores[chunk.chunk_id]

        for chunk, score, rank in dense_results:
            item = _get_or_create_item(chunk)
            item.scores["dense_score"] = score
            item.scores["dense_rank"] = rank
            item.scores["rrf_score"] += 1.0 / (k + rank)

        for chunk, score, rank in sparse_results:
            item = _get_or_create_item(chunk)
            item.scores["bm25_score"] = score
            item.scores["bm25_rank"] = rank
            item.scores["rrf_score"] += 1.0 / (k + rank)

        # Sort by RRF score (highest first)
        fused = sorted(combined_scores.values(),
                       key=lambda x: x.scores["rrf_score"], reverse=True)

        # ── Per-source diversity cap ──────────────────────────────────────
        source_counts: Dict[str, int] = {}
        diverse_fused: List[SearchResultItem] = []
        overflow: List[SearchResultItem] = []   # extras beyond SOURCE_CAP

        for item in fused:
            src = item.source_id
            count = source_counts.get(src, 0)
            if count < SOURCE_CAP:
                source_counts[src] = count + 1
                diverse_fused.append(item)
            else:
                overflow.append(item)

        # Append overflow at the end so callers still see all candidates
        return diverse_fused + overflow
