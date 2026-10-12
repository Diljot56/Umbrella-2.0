import sys
import os
sys.path.append(os.getcwd())
os.environ["USE_MOCK_LLM"] = "true"

import json
from src.indexing.indexer import DualIndexerService
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerService
from src.generation.generator import GroundedGenerator

def main():
    indexer = DualIndexerService(persist=True)
    indexer.load_bm25()
    reranker = RerankerService()
    generator = GroundedGenerator()
    retriever = HybridRetriever(indexer)

    with open("data/ground_truth/queries.json", "r") as f:
        queries = json.load(f)

    with open("eval_results.txt", "w", encoding="utf-8") as out:
        for q in queries:
            query_text = q["query"]
            expected_src = q["expected_source"]
            
            dense_results = retriever.search_dense(query_text, top_n=20)
            sparse_results = retriever.search_sparse(query_text, top_n=20)
            
            fused_candidates = retriever.fuse_rrf(dense_results, sparse_results, k=60)
            final_results = reranker.rerank(query_text, fused_candidates, top_k=5)
            
            response = generator.generate_answer(query_text, final_results)
            
            out.write(f"--- QUERY: {query_text} ---\n")
            out.write(f"EXPECTED: {expected_src}\n")
            
            retrieved_sources = [res.source_id for res in final_results]
            out.write(f"RETRIEVED SOURCES: {retrieved_sources}\n")
            out.write(f"ANSWER: {response.answer}\n")
            if hasattr(response, 'citations'):
                out.write(f"CITED CHUNKS: {[c.chunk_id for c in response.citations] if response.citations else []}\n")
            out.write("\n")
            print(f"Processed: {query_text}")

if __name__ == "__main__":
    main()
