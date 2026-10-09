"""
One-time ingestion script.

Run this ONCE to parse raw data, embed it, and persist the indices to disk.
After this, the server loads the pre-built indices instantly on startup.

Usage:
    python scripts/run_ingestion.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ingestion.code_chunker import CodeChunker
from src.ingestion.doc_chunker import DocChunker
from src.ingestion.ticket_chunker import TicketChunker
from src.indexing.indexer import DualIndexerService

def main():
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "data" / "raw"

    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} does not exist.")
        return

    code_chunker = CodeChunker()
    doc_chunker = DocChunker()
    ticket_chunker = TicketChunker()

    # Use persist=True so Qdrant writes to disk
    indexer = DualIndexerService(persist=True)

    all_chunks = []

    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue

        print(f"Processing {file_path}...")

        chunks = []
        if file_path.suffix == ".py":
            chunks = code_chunker.process(file_path)
        elif file_path.suffix in [".md", ".pdf"]:
            chunks = doc_chunker.process(file_path)
        elif file_path.suffix == ".json":
            chunks = ticket_chunker.process(file_path)
        else:
            print(f"  No chunker for file type {file_path.suffix}, skipping.")

        all_chunks.extend(chunks)
        print(f"  Generated {len(chunks)} chunks.")

    print(f"\nTotal chunks generated: {len(all_chunks)}. Indexing...")

    if all_chunks:
        indexer.index_chunks(all_chunks)
        indexer.save_bm25()
        print(f"\n=== Ingestion Complete ===")
        print(f"  Qdrant points: {indexer.get_qdrant_count()}")
        print(f"  BM25 documents: {indexer.get_bm25_count()}")
        print(f"  Indices saved to: data/index/")
    else:
        print("No chunks to index.")

if __name__ == "__main__":
    main()
