import ast
from typing import List
from pathlib import Path
from src.ingestion.base import BaseChunker
from src.models.chunk import Chunk

# Maximum characters for a single chunk; chunks larger than this are split
# to avoid overwhelming the cross-encoder's token window.
MAX_CHUNK_CHARS = 4000

class CodeChunker(BaseChunker):
    """Parses Python files and chunks them by AST nodes.

    Strategy:
    - Top-level functions/async-functions → one chunk each.
    - Top-level ClassDef → walk its body and emit a distinct chunk for
      every method (FunctionDef / AsyncFunctionDef).  The chunk text is
      prefixed with the class signature + docstring so each method chunk
      is self-contained and fully grounded.
    - If a class has no methods (e.g. a pure-data class) it is emitted
      as a single chunk capped at MAX_CHUNK_CHARS.
    """

    def process(self, file_path: Path, **kwargs) -> List[Chunk]:
        chunks: List[Chunk] = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=file_path.name)

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Top-level function → one chunk
                    chunk = self._make_chunk(file_path, source, node,
                                             context_prefix="")
                    if chunk:
                        chunks.append(chunk)

                elif isinstance(node, ast.ClassDef):
                    # Walk into the class body
                    method_chunks = self._chunk_class(file_path, source, node)
                    chunks.extend(method_chunks)

        except Exception as e:
            print(f"Error parsing code file {file_path}: {e}")

        return chunks

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _class_header(self, source: str, class_node: ast.ClassDef) -> str:
        """Return the class signature line + docstring (if present)."""
        header_lines: List[str] = []

        # Signature: "class Foo(Bar, Baz):"
        bases = ", ".join(ast.unparse(b) for b in class_node.bases)
        sig = f"class {class_node.name}" + (f"({bases})" if bases else "") + ":"
        header_lines.append(sig)

        # Docstring
        body = class_node.body
        if body and isinstance(body[0], ast.Expr):
            val = body[0].value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                docstring = ast.get_source_segment(source, body[0]) or ""
                header_lines.append(f"    {docstring.strip()}")

        return "\n".join(header_lines)

    def _chunk_class(self, file_path: Path, source: str,
                     class_node: ast.ClassDef) -> List[Chunk]:
        chunks: List[Chunk] = []
        class_header = self._class_header(source, class_node)

        methods = [
            n for n in class_node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if not methods:
            # No methods → emit the whole class (capped)
            segment = ast.get_source_segment(source, class_node) or ""
            segment = segment[:MAX_CHUNK_CHARS]
            location = f"line {class_node.lineno}: {class_node.name}"
            chunk_id = Chunk.generate_id(file_path.name, location)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_id=file_path.name,
                source_type="code",
                location=location,
                text=segment,
                metadata={
                    "node_type": "ClassDef",
                    "name": class_node.name,
                    "lineno": class_node.lineno,
                }
            ))
            return chunks

        for method in methods:
            chunk = self._make_chunk(
                file_path, source, method,
                context_prefix=f"# Class: {class_node.name}\n{class_header}\n\n"
            )
            if chunk:
                chunks.append(chunk)

        return chunks

    def _make_chunk(self, file_path: Path, source: str, node: ast.AST,
                    context_prefix: str) -> "Chunk | None":
        segment = ast.get_source_segment(source, node)
        if not segment:
            return None

        text = context_prefix + segment
        # Cap to avoid overwhelming the cross-encoder
        text = text[:MAX_CHUNK_CHARS]

        name = getattr(node, "name", "<unknown>")
        lineno = getattr(node, "lineno", 0)
        location = f"line {lineno}: {name}"
        chunk_id = Chunk.generate_id(file_path.name, location)

        return Chunk(
            chunk_id=chunk_id,
            source_id=file_path.name,
            source_type="code",
            location=location,
            text=text,
            metadata={
                "node_type": type(node).__name__,
                "name": name,
                "lineno": lineno,
            }
        )
