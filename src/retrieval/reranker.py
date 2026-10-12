from typing import List
from sentence_transformers import CrossEncoder
from src.models.search import SearchResultItem

class RerankerService:
    _instance = None
    
    def __new__(cls, model_name: str = "BAAI/bge-reranker-large"):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
            cls._instance.encoder = CrossEncoder(model_name)
        return cls._instance

    def rerank(self, query: str, candidates: List[SearchResultItem], top_k: int, max_rerank_pool: int = 5) -> List[SearchResultItem]:
        if not candidates:
            return []
            
        # Optimize latency on CPU: rerank top RRF candidates instead of the entire candidate set
        to_rerank = candidates[:max_rerank_pool]
        unranked = candidates[max_rerank_pool:]

        pairs = [[query, item.text] for item in to_rerank]
        scores = self.encoder.predict(pairs, max_length=192)
        
        # Handle case where predict returns a single float instead of array if len == 1
        if isinstance(scores, float) or (hasattr(scores, 'ndim') and scores.ndim == 0):
            scores = [scores]
            
        for i, item in enumerate(to_rerank):
            item.scores["cross_encoder_score"] = float(scores[i])

        for item in unranked:
            item.scores["cross_encoder_score"] = -999.0
            
        to_rerank.sort(key=lambda x: x.scores["cross_encoder_score"], reverse=True)
        return (to_rerank + unranked)[:top_k]
