import time
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
        t_start = time.perf_counter()
        retriever = HybridRetriever(indexer)
        
        # 1. Retrieve candidates
        t0 = time.perf_counter()
        dense_results = retriever.search_dense(request.query, top_n=request.top_dense)
        t_dense = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse_results = retriever.search_sparse(request.query, top_n=request.top_sparse)
        t_sparse = (time.perf_counter() - t0) * 1000
        
        # 2. Fuse
        t0 = time.perf_counter()
        fused_candidates = retriever.fuse_rrf(dense_results, sparse_results, k=request.rrf_k)
        t_fuse = (time.perf_counter() - t0) * 1000
        
        # 3. Rerank
        t0 = time.perf_counter()
        final_results = reranker.rerank(request.query, fused_candidates, top_k=request.top_final)
        t_rerank = (time.perf_counter() - t0) * 1000

        total_t = (time.perf_counter() - t_start) * 1000
        print(f"[TIMING /search] Dense: {t_dense:.1f}ms | Sparse: {t_sparse:.1f}ms | Fuse ({len(fused_candidates)}): {t_fuse:.1f}ms | Rerank ({len(fused_candidates)} candidates): {t_rerank:.1f}ms | Total: {total_t:.1f}ms")
        
        return final_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, indexer: DualIndexerService = Depends(get_indexer), reranker: RerankerService = Depends(get_reranker), generator: GroundedGenerator = Depends(get_generator)):
    try:
        t_start = time.perf_counter()
        retriever = HybridRetriever(indexer)
        
        # 1. Retrieve candidates
        t0 = time.perf_counter()
        dense_results = retriever.search_dense(request.query, top_n=request.top_dense)
        t_dense = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        sparse_results = retriever.search_sparse(request.query, top_n=request.top_sparse)
        t_sparse = (time.perf_counter() - t0) * 1000
        
        # 2. Fuse
        t0 = time.perf_counter()
        fused_candidates = retriever.fuse_rrf(dense_results, sparse_results, k=60)
        t_fuse = (time.perf_counter() - t0) * 1000
        
        # 3. Rerank
        t0 = time.perf_counter()
        final_results = reranker.rerank(request.query, fused_candidates, top_k=request.top_final)
        t_rerank = (time.perf_counter() - t0) * 1000
        
        # 4. Generate
        t0 = time.perf_counter()
        response = generator.generate_answer(request.query, final_results)
        t_gen = (time.perf_counter() - t0) * 1000

        total_t = (time.perf_counter() - t_start) * 1000
        print(f"[TIMING /ask] Dense: {t_dense:.1f}ms | Sparse: {t_sparse:.1f}ms | Fuse: {t_fuse:.1f}ms | Rerank: {t_rerank:.1f}ms | Gen: {t_gen:.1f}ms | Total: {total_t:.1f}ms")

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
