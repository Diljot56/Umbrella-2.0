from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.indexing.indexer import DualIndexerService, QDRANT_PATH, BM25_PICKLE
from src.retrieval.reranker import RerankerService
from src.generation.generator import GroundedGenerator
from src.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  Project Umbrella — Starting Up")
    print("=" * 60)

    # Check if pre-built indices exist
    qdrant_exists = Path(QDRANT_PATH).exists()
    bm25_exists = BM25_PICKLE.exists()

    if qdrant_exists and bm25_exists:
        print("[1/3] Loading pre-built indices from disk...")
        app.state.indexer = DualIndexerService(persist=True)
        app.state.indexer.load_bm25()
        print(f"       Qdrant points : {app.state.indexer.get_qdrant_count()}")
        print(f"       BM25 documents: {app.state.indexer.get_bm25_count()}")
    else:
        print("[!] No pre-built indices found.")
        print("    Run 'python scripts/run_ingestion.py' first to ingest data.")
        print("    Starting with empty indices for now.")
        app.state.indexer = DualIndexerService(persist=False)

    print("[2/3] Loading reranker model (bge-reranker-large)...")
    app.state.reranker = RerankerService()

    print("[3/3] Initializing generator...")
    app.state.generator = GroundedGenerator()

    print("=" * 60)
    print("  Server ready! Visit http://127.0.0.1:8000")
    print("=" * 60)

    yield

    # Cleanup
    app.state.indexer = None
    app.state.reranker = None
    app.state.generator = None

app = FastAPI(title="Project Umbrella API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

import os
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "indexer_initialized": hasattr(app.state, "indexer") and app.state.indexer is not None,
        "qdrant_count": app.state.indexer.get_qdrant_count() if app.state.indexer else 0,
        "bm25_count": app.state.indexer.get_bm25_count() if app.state.indexer else 0,
        "reranker_initialized": hasattr(app.state, "reranker") and app.state.reranker is not None,
        "generator_initialized": hasattr(app.state, "generator") and app.state.generator is not None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
