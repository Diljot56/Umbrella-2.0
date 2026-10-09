import string
import uuid
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.models.chunk import Chunk

INDEX_DIR = Path("data/index")
BM25_PICKLE = INDEX_DIR / "bm25_state.pkl"
QDRANT_PATH = str(INDEX_DIR / "qdrant_store")

class DualIndexerService:
    """Manages synchronized Qdrant (dense) and BM25 (sparse) indices.
    
    Supports two modes:
      - Ingestion mode: builds indices from scratch and persists to disk.
      - Serving mode:   loads pre-built indices from disk (fast startup).
    """

    def __init__(self, collection_name: str = "enterprise_knowledge",
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 persist: bool = False):
        self.collection_name = collection_name
        self.persist = persist

        # Choose Qdrant backend
        if persist:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            self.qdrant = QdrantClient(path=QDRANT_PATH)
        else:
            self.qdrant = QdrantClient(":memory:")

        self.encoder = SentenceTransformer(model_name)

        # Create collection only if it doesn't already exist
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )

        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.idx_to_chunk_id: Dict[int, str] = {}
        self.chunk_store: Dict[str, Chunk] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation tokenizer for BM25."""
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def index_chunks(self, chunks: List[Chunk], batch_size: int = 64):
        """Embeds chunks, upserts to Qdrant, and builds the BM25 index."""
        if not chunks:
            return

        points = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [c.text for c in batch]
            print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")

            embeddings = self.encoder.encode(texts, normalize_embeddings=True)

            for j, chunk in enumerate(batch):
                self.chunk_store[chunk.chunk_id] = chunk

                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
                points.append(PointStruct(
                    id=point_uuid,
                    vector=embeddings[j].tolist(),
                    payload=chunk.model_dump()
                ))

                idx = len(self.bm25_corpus)
                self.bm25_corpus.append(self._tokenize(chunk.text))
                self.idx_to_chunk_id[idx] = chunk.chunk_id

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )

        self.bm25 = BM25Okapi(self.bm25_corpus)

    def save_bm25(self):
        """Persist BM25 state to disk as a pickle file."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "bm25_corpus": self.bm25_corpus,
            "idx_to_chunk_id": self.idx_to_chunk_id,
            "chunk_store": {cid: c.model_dump() for cid, c in self.chunk_store.items()}
        }
        with open(BM25_PICKLE, "wb") as f:
            pickle.dump(state, f)
        print(f"BM25 state saved to {BM25_PICKLE}")

    def load_bm25(self):
        """Load BM25 state from a previously saved pickle file."""
        if not BM25_PICKLE.exists():
            print(f"WARNING: BM25 pickle not found at {BM25_PICKLE}. Run ingestion first.")
            return
        with open(BM25_PICKLE, "rb") as f:
            state = pickle.load(f)
        self.bm25_corpus = state["bm25_corpus"]
        self.idx_to_chunk_id = state["idx_to_chunk_id"]
        self.chunk_store = {cid: Chunk(**data) for cid, data in state["chunk_store"].items()}
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        print(f"BM25 state loaded: {len(self.bm25_corpus)} documents, {len(self.chunk_store)} chunks in store.")

    def get_qdrant_count(self) -> int:
        return self.qdrant.count(self.collection_name).count

    def get_bm25_count(self) -> int:
        return len(self.bm25_corpus)
