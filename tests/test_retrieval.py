import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.chunk import Chunk
from src.models.search import SearchResultItem
from src.indexing.indexer import DualIndexerService
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerService

class MockEncoder:
    def encode(self, texts, normalize_embeddings=False):
        import numpy as np
        return np.random.rand(len(texts), 1024)

class MockCrossEncoder:
    def predict(self, pairs):
        import numpy as np
        return np.random.rand(len(pairs))

@pytest.fixture
def mock_indexer():
    with patch("src.indexing.indexer.SentenceTransformer", return_value=MockEncoder()):
        indexer = DualIndexerService(model_name="mock")
        # index a couple of chunks
        chunks = [
            Chunk(chunk_id="1", source_id="a.md", source_type="document", location="H1", text="Python is a programming language", metadata={}),
            Chunk(chunk_id="2", source_id="b.md", source_type="document", location="H2", text="FastAPI is a modern, fast web framework for building APIs", metadata={}),
            Chunk(chunk_id="3", source_id="c.md", source_type="document", location="H3", text="Qdrant is a vector similarities engine", metadata={})
        ]
        indexer.index_chunks(chunks)
        return indexer

def test_hybrid_retriever_dense_sparse_isolation(mock_indexer):
    retriever = HybridRetriever(mock_indexer)
    # Dense
    dense_res = retriever.search_dense("FastAPI", top_n=2)
    assert len(dense_res) == 2
    for chunk, score, rank in dense_res:
        assert isinstance(chunk, Chunk)
        assert rank in [1, 2] # 1 based

    # Sparse
    sparse_res = retriever.search_sparse("fastapi api framework", top_n=2)
    assert len(sparse_res) > 0
    assert sparse_res[0][0].chunk_id == "2"

def test_rrf_logic(mock_indexer):
    retriever = HybridRetriever(mock_indexer)
    
    # Create fake retrieval sets
    chunk_a = mock_indexer.chunk_store["1"]
    chunk_b = mock_indexer.chunk_store["2"]

    # chunk a is rank 1 in dense, rank 2 in sparse
    dense = [(chunk_a, 0.9, 1), (chunk_b, 0.8, 2)]
    sparse = [(chunk_b, 10.5, 1), (chunk_a, 5.0, 2)]

    fused = retriever.fuse_rrf(dense, sparse, k=60)
    
    assert len(fused) == 2
    # Verify exact math (1-based validation explicitly calculated outside)
    assert round(fused[0].scores["rrf_score"], 4) == round(fused[1].scores["rrf_score"], 4)
    item_a = next(i for i in fused if i.chunk_id == "1")
    assert item_a.scores["dense_rank"] == 1
    assert item_a.scores["bm25_rank"] == 2

@patch("src.retrieval.reranker.CrossEncoder", return_value=MockCrossEncoder())
def test_reranker_sorts(mock_ce):
    reranker = RerankerService(model_name="mock")
    # Reset instance explicitly for tests
    RerankerService._instance.encoder = MockCrossEncoder()
    # Force Mock predict return
    RerankerService._instance.encoder.predict = MagicMock(return_value=[0.1, 0.9])
    
    c1 = SearchResultItem(chunk_id="1", text="Low score", source_id="", source_type="", location="", metadata={})
    c2 = SearchResultItem(chunk_id="2", text="High score", source_id="", source_type="", location="", metadata={})
    
    reranked = reranker.rerank("Query", [c1, c2], top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "2" # Sorts to Top!
    assert reranked[0].scores["cross_encoder_score"] == 0.9
    assert reranked[1].chunk_id == "1"
    assert reranked[1].scores["cross_encoder_score"] == 0.1

@pytest.mark.skip(reason="Downloads real models; meant for staging E2E assertion")
def test_integration_endpoints():
    pass
