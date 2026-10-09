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

    def rerank(self, query: str, candidates: List[SearchResultItem], top_k: int) -> List[SearchResultItem]:
        if not candidates:
            return []
            
        pairs = [[query, item.text] for item in candidates]
        scores = self.encoder.predict(pairs)
        
        # Handle case where predict returns a single float instead of array if len == 1
        if isinstance(scores, float) or (hasattr(scores, 'ndim') and scores.ndim == 0):
            scores = [scores]
            
        for i, item in enumerate(candidates):
            item.scores["cross_encoder_score"] = float(scores[i])
            
        candidates.sort(key=lambda x: x.scores["cross_encoder_score"], reverse=True)
        return candidates[:top_k]
