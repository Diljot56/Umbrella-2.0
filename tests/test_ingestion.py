import os
import sys
import ast
import json
import pytest
from pathlib import Path
from unittest.mock import patch

# Setup path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.code_chunker import CodeChunker
from src.ingestion.doc_chunker import DocChunker
from src.ingestion.ticket_chunker import TicketChunker
from src.indexing.indexer import DualIndexerService
from src.models.chunk import Chunk

@pytest.fixture
def dummy_data_dir(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    
    # 1. Create a dummy python file
    code_path = raw_dir / "test_code.py"
    code_path.write_text('def sample_func():\n    """Test docstring"""\n    return 42\n', encoding="utf-8")
    
    # 2. Create a dummy markdown file
    md_path = raw_dir / "test_doc.md"
    md_path.write_text('# Header 1\nSome text here.\n## Header 2\nMore text.', encoding="utf-8")
    
    # 3. Create a dummy JSON file
    json_path = raw_dir / "test_ticket.json"
    ticket_data = [{"id": "T-100", "description": "Bug here", "comments": [{"author": "Bob", "body": "Fixed"}]}]
    json_path.write_text(json.dumps(ticket_data), encoding="utf-8")
    
    return raw_dir

class MockEncoder:
    def encode(self, texts, normalize_embeddings=False):
        # bge-large is 1024 dims
        import numpy as np
        # Return random arrays of shape (len(texts), 1024)
        return np.random.rand(len(texts), 1024)

def test_chunking_and_indexing(dummy_data_dir):
    code_chunker = CodeChunker()
    doc_chunker = DocChunker()
    ticket_chunker = TicketChunker()
    
    chunks = []
    for file_path in dummy_data_dir.rglob("*"):
        if file_path.suffix == ".py":
            chunks.extend(code_chunker.process(file_path))
        elif file_path.suffix == ".md":
            chunks.extend(doc_chunker.process(file_path))
        elif file_path.suffix == ".json":
            chunks.extend(ticket_chunker.process(file_path))
            
    assert len(chunks) > 0, "Should have created chunks"
    
    # Verify AST syntax does not break on code chunks
    code_chunks = [c for c in chunks if c.source_type == "code"]
    assert len(code_chunks) == 1
    
    # This should parse successfully if the chunker correctly extracted the whole segment
    parsed_ast = ast.parse(code_chunks[0].text)
    assert isinstance(parsed_ast, ast.Module)
    assert len(parsed_ast.body) == 1
    assert isinstance(parsed_ast.body[0], ast.FunctionDef)
    
    # Mock SentenceTransformer inside DualIndexerService
    with patch("src.indexing.indexer.SentenceTransformer", return_value=MockEncoder()):
        indexer = DualIndexerService(model_name="mocked")
        indexer.index_chunks(chunks)
        
        qdrant_count = indexer.get_qdrant_count()
        bm25_count = indexer.get_bm25_count()
        
        assert qdrant_count == bm25_count, "Indices must be mapped 1:1"
        assert qdrant_count == len(chunks)
