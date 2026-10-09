from fastapi import APIRouter, Depends, HTTPException
from typing import List
from src.models.search import SearchRequest, SearchResultItem
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerService
from src.indexing.indexer import DualIndexerService
from src.models.generation import AskRequest, AskResponse
from src.generation.generator import GroundedGenerator

router = APIRouter()

def get_indexer():
    from src.main import app
    return app.state.indexer

def get_reranker():
    from src.main import app
    return app.state.reranker

def get_generator():
    from src.main import app
    return app.state.generator

@router.post("/search", response_model=List[SearchResultItem])
async def search(request: SearchRequest, indexer: DualIndexerService = Depends(get_indexer), reranker: RerankerService = Depends(get_reranker)):
    try:
        retriever = HybridRetriever(indexer)
        
        # 1. Retrieve candidates
        dense_results = retriever.search_dense(request.query, top_n=request.top_dense)
        sparse_results = retriever.search_sparse(request.query, top_n=request.top_sparse)
        
        # 2. Fuse
        fused_candidates = retriever.fuse_rrf(dense_results, sparse_results, k=request.rrf_k)
        
        # 3. Rerank
        final_results = reranker.rerank(request.query, fused_candidates, top_k=request.top_final)
        
        return final_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, indexer: DualIndexerService = Depends(get_indexer), reranker: RerankerService = Depends(get_reranker), generator: GroundedGenerator = Depends(get_generator)):
    try:
        retriever = HybridRetriever(indexer)
        
        # 1. Retrieve candidates
        dense_results = retriever.search_dense(request.query, top_n=request.top_dense)
        sparse_results = retriever.search_sparse(request.query, top_n=request.top_sparse)
        
        # 2. Fuse
        fused_candidates = retriever.fuse_rrf(dense_results, sparse_results, k=60)
        
        # 3. Rerank
        final_results = reranker.rerank(request.query, fused_candidates, top_k=request.top_final)
        
        # 4. Generate
        response = generator.generate_answer(request.query, final_results)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
