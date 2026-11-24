import time
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import List
from src.models.search import SearchRequest, SearchResultItem
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerService
from src.indexing.indexer import DualIndexerService
from src.models.generation import AskRequest, AskResponse
from src.generation.generator import GroundedGenerator
from src.ingestion.code_chunker import CodeChunker
from src.ingestion.doc_chunker import DocChunker
from src.ingestion.ticket_chunker import TicketChunker

router = APIRouter()

RAW_DATA_DIR = Path("data/raw")
UPLOAD_DIR = RAW_DATA_DIR / "uploads"

# File-type to subdirectory mapping (for display purposes)
SUPPORTED_EXTENSIONS = {".py", ".md", ".pdf", ".json"}

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


# ------------------------------------------------------------------ #
# Knowledge-base file management
# ------------------------------------------------------------------ #

def _scan_files() -> List[dict]:
    """Walk data/raw and return metadata for every ingested file."""
    files = []
    if not RAW_DATA_DIR.exists():
        return files
    for fp in sorted(RAW_DATA_DIR.rglob("*")):
        if not fp.is_file():
            continue
        # Skip helper scripts inside the raw folder (e.g. fetch_tickets.py)
        rel = fp.relative_to(RAW_DATA_DIR)
        category = rel.parts[0] if len(rel.parts) > 1 else "other"
        files.append({
            "name": fp.name,
            "path": str(rel),
            "size": fp.stat().st_size,
            "extension": fp.suffix.lower(),
            "category": category,
        })
    return files


@router.get("/files")
async def list_files():
    """Return every file currently in the knowledge base."""
    files = _scan_files()
    indexer = get_indexer()
    return {
        "files": files,
        "total_files": len(files),
        "qdrant_points": indexer.get_qdrant_count() if indexer else 0,
        "bm25_documents": indexer.get_bm25_count() if indexer else 0,
    }


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    indexer: DualIndexerService = Depends(get_indexer),
):
    """Upload a file, chunk it, and add it to the live indices."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Persist the uploaded file to data/raw/uploads/
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Choose the right chunker
    if ext == ".py":
        chunker = CodeChunker()
    elif ext in (".md", ".pdf"):
        chunker = DocChunker()
    elif ext == ".json":
        chunker = TicketChunker()
    else:
        raise HTTPException(status_code=400, detail="No chunker available for this file type.")

    try:
        chunks = chunker.process(dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {e}")

    if not chunks:
        return {
            "status": "warning",
            "filename": file.filename,
            "chunks_added": 0,
            "message": "File saved but produced 0 chunks (file may be empty or unreadable).",
        }

    # Index into the live Qdrant + BM25 indices
    indexer.index_chunks(chunks)
    indexer.save_bm25()

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_added": len(chunks),
        "qdrant_points": indexer.get_qdrant_count(),
        "bm25_documents": indexer.get_bm25_count(),
    }


@router.delete("/files/{filename}")
async def delete_file(
    filename: str,
    indexer: DualIndexerService = Depends(get_indexer),
):
    """Remove an uploaded file and its chunks from the indices."""
    target = UPLOAD_DIR / filename
    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File '{filename}' not found in uploads. Only uploaded files can be deleted.",
        )

    target.unlink()

    # Rebuild indices from scratch so deleted chunks disappear
    from src.main import app
    new_indexer = _rebuild_indices()
    app.state.indexer = new_indexer

    return {
        "status": "success",
        "filename": filename,
        "qdrant_points": new_indexer.get_qdrant_count(),
        "bm25_documents": new_indexer.get_bm25_count(),
    }


def _rebuild_indices() -> DualIndexerService:
    """Re-ingest everything in data/raw into fresh indices."""
    from src.indexing.indexer import DualIndexerService as DIS
    code_chunker = CodeChunker()
    doc_chunker = DocChunker()
    ticket_chunker = TicketChunker()

    new_indexer = DIS(persist=True)

    all_chunks = []
    for fp in RAW_DATA_DIR.rglob("*"):
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        if ext == ".py":
            all_chunks.extend(code_chunker.process(fp))
        elif ext in (".md", ".pdf"):
            all_chunks.extend(doc_chunker.process(fp))
        elif ext == ".json":
            all_chunks.extend(ticket_chunker.process(fp))

    if all_chunks:
        new_indexer.index_chunks(all_chunks)
        new_indexer.save_bm25()

    return new_indexer
