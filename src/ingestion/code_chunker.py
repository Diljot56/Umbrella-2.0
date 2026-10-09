import ast
from typing import List
from pathlib import Path
from src.ingestion.base import BaseChunker
from src.models.chunk import Chunk

class CodeChunker(BaseChunker):
    """Parses Python files and chunks them by top-level classes and functions."""

    def process(self, file_path: Path, **kwargs) -> List[Chunk]:
        chunks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source, filename=file_path.name)
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Extract full source segment, including docstrings
                    segment = ast.get_source_segment(source, node)
                    if not segment:
                        continue
                        
                    location = f"line {node.lineno}: {node.name}"
                    chunk_id = Chunk.generate_id(file_path.name, location)
                    
                    chunk = Chunk(
                        chunk_id=chunk_id,
                        source_id=file_path.name,
                        source_type="code",
                        location=location,
                        text=segment,
                        metadata={
                            "node_type": type(node).__name__,
                            "name": node.name,
                            "lineno": node.lineno
                        }
                    )
                    chunks.append(chunk)

        except Exception as e:
            print(f"Error parsing code file {file_path}: {e}")
        
        return chunks
