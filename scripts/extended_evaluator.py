"""
Extended RAG Evaluation Script — Ragas/RAG-Triad Standards
===========================================================
Replaces the brittle boolean pass/fail with three continuous metrics:

  Context Recall  (CR)  : fraction of expected sources present in Top-K.
  Context Precision (CP): Mean Reciprocal Rank of the first expected-source hit.
  Faithfulness    (F)   : for out-of-corpus queries, 1.0 iff the answer is an
                          explicit refusal; for normal queries, 1.0 iff every
                          cited source is from the expected set.

Pass condition:
  - Out-of-corpus → F == 1.0  (must refuse)
  - Otherwise     → CR >= 0.5 AND CP >= 0.5 AND F == 1.0
    (partial recall allowed; MRR at Rank-2 still passes)
"""
import sys
import os
sys.path.append(os.getcwd())
os.environ["USE_MOCK_LLM"] = "true"

import json
from typing import List, Dict, Any
from src.indexing.indexer import DualIndexerService
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import RerankerService
from src.generation.generator import GroundedGenerator

# ──────────────────────────────────────────────────────────────────────
# Standard "Information not found" token the generator emits on refusal.
# Must match NO_ANSWER in generator.py.
REFUSAL_SENTINEL = "no supported answer could be found in the provided sources"

EXTENDED_QUERIES = [
    # ── A: Cross-source / Ambiguous ──────────────────────────────
    {
        "query_id": "A1",
        "query": "How are HTTP errors handled in the GitLab Python client?",
        "expected_sources": ["gitlab_client.py", "gitlab_exceptions.py"],
        "category": "cross_source",
        "notes": "Answer requires BOTH client (raises GitlabHttpError) and exceptions (defines it). Tests whether pipeline surfaces both sources.",
    },
    {
        "query_id": "A2",
        "query": "What happens when a runner token is compromised?",
        "expected_sources": ["configure_runners.md", "runner_issues.json"],
        "category": "cross_source",
        "notes": "configure_runners.md discusses token rotation/security; runner_issues.json has the prefixed token bug. Both are relevant.",
    },

    # ── B: Out-of-corpus / Negation ──────────────────────────────
    {
        "query_id": "B1",
        "query": "How do I configure GitLab Runner on Kubernetes with Helm?",
        "expected_sources": [],
        "category": "out_of_corpus",
        "notes": "Kubernetes/Helm setup is NOT in the corpus. A faithful system should say 'no supported answer'. Tests hallucination.",
    },
    {
        "query_id": "B2",
        "query": "What is the default Docker image used by GitLab CI?",
        "expected_sources": [],
        "category": "out_of_corpus",
        "notes": "Default Docker image (ruby:2.5 or alpine) is NOT in the corpus. Tests whether the system hallucinates from world knowledge.",
    },
    {
        "query_id": "B3",
        "query": "How to set up GitLab Pages with a custom domain?",
        "expected_sources": [],
        "category": "out_of_corpus",
        "notes": "Pages custom domain setup not in corpus. Only a brief mention of Pages exists. Tests precision of retrieval.",
    },

    # ── C: Multi-hop reasoning ───────────────────────────────────
    {
        "query_id": "C1",
        "query": "If I set GIT_DEPTH to 1, what problem occurs with retried jobs, and which variable controls submodule depth separately?",
        "expected_sources": ["configure_runners.md"],
        "category": "multi_hop",
        "notes": "Requires joining info from 'Shallow cloning' section (GIT_DEPTH=1 issue) with 'Git submodule depth' (GIT_SUBMODULE_DEPTH). Separate chunks.",
    },
    {
        "query_id": "C2",
        "query": "What exception is raised when the server returns 401 and what base class does it inherit from?",
        "expected_sources": ["gitlab_client.py", "gitlab_exceptions.py"],
        "category": "multi_hop",
        "notes": "gitlab_client.py raises GitlabAuthenticationError on 401; gitlab_exceptions.py shows it inherits from GitlabError. Multi-source multi-hop.",
    },

    # ── D: Near-miss lexical traps ───────────────────────────────
    {
        "query_id": "D1",
        "query": "GitlabGetError",
        "expected_sources": ["gitlab_exceptions.py"],
        "category": "lexical_trap",
        "notes": "Similar to Q2 (GitlabHttpError) but a different, less prominent class. BM25 should still find it.",
    },
    {
        "query_id": "D2",
        "query": "on_http_error decorator",
        "expected_sources": ["gitlab_exceptions.py"],
        "category": "lexical_trap",
        "notes": "Decorator function defined at bottom of exceptions file. Tests if chunker splits it into a retrievable unit.",
    },
    {
        "query_id": "D3",
        "query": "FF_USE_FASTZIP",
        "expected_sources": ["configure_runners.md"],
        "category": "lexical_trap",
        "notes": "Exact feature flag name. Appears in artifact/cache settings section. Tests BM25 exact match in docs.",
    },

    # ── E: Ticket-specific retrieval ─────────────────────────────
    {
        "query_id": "E1",
        "query": "What is the IPv6 health check bug in GitLab Runner Docker executor?",
        "expected_sources": ["runner_issues.json"],
        "category": "ticket_retrieval",
        "notes": "GL-39939 describes the IPv6 health check service bug. Tests ticket content retrieval.",
    },
    {
        "query_id": "E2",
        "query": "Runner tokens with instance prefix are not recognized",
        "expected_sources": ["runner_issues.json"],
        "category": "ticket_retrieval",
        "notes": "GL-39938 has this exact title. Tests direct title match in ticket data.",
    },
    {
        "query_id": "E3",
        "query": "What is the FF_DISABLE_UMASK_FOR_DOCKER_EXECUTOR feature flag decision?",
        "expected_sources": ["runner_issues.json"],
        "category": "ticket_retrieval",
        "notes": "GL-39940 deferred decision. Tests retrieval of ticket content with feature flag names.",
    },

    # ── F: Markdown spec retrieval ───────────────────────────────
    {
        "query_id": "F1",
        "query": "How do I create a task list in GitLab Markdown?",
        "expected_sources": ["markdown_spec.md"],
        "category": "markdown_spec",
        "notes": "Task lists section in markdown_spec.md. Tests if the second doc is retrievable.",
    },
    {
        "query_id": "F2",
        "query": "What is GLFM and what features does it support?",
        "expected_sources": ["markdown_spec.md"],
        "category": "markdown_spec",
        "notes": "GLFM (GitLab Flavored Markdown) intro section. Semantic query about the markdown spec.",
    },
    {
        "query_id": "F3",
        "query": "How are heading anchors generated in GitLab?",
        "expected_sources": ["markdown_spec.md"],
        "category": "markdown_spec",
        "notes": "Heading anchors section. Tests semantic retrieval from markdown_spec.",
    },

    # ── G: Very short / very long queries ────────────────────────
    {
        "query_id": "G1",
        "query": "tags",
        "expected_sources": ["configure_runners.md"],
        "category": "short_query",
        "notes": "Extremely vague single-word query. Should match configure_runners tags section, but might get noise from everywhere.",
    },
    {
        "query_id": "G2",
        "query": "error",
        "expected_sources": ["gitlab_exceptions.py"],
        "category": "short_query",
        "notes": "Single word 'error' appears across ALL sources. Tests whether system retrieves the most relevant (exceptions file).",
    },
    {
        "query_id": "G3",
        "query": (
            "I am working on a GitLab CI pipeline and I have a runner that is configured "
            "to build Docker images, but when the runner tries to clone the repository it "
            "fails because my GitLab instance is behind a firewall and the runner cannot "
            "reach the default URL, so I need to know what configuration parameter I should "
            "set in the runner configuration file to specify an alternative URL for cloning "
            "the repository when the default GitLab instance URL is not reachable from the "
            "runner machine"
        ),
        "expected_sources": ["configure_runners.md"],
        "category": "long_query",
        "notes": "Extremely verbose version of Q3. Tests whether long queries dilute embedding quality.",
    },

    # ── H: Code retrieval for non-obvious items ──────────────────
    {
        "query_id": "H1",
        "query": "What is the http_request method signature in the Gitlab client?",
        "expected_sources": ["gitlab_client.py"],
        "category": "code_retrieval",
        "notes": "http_request is a method inside the Gitlab class. Tests per-method chunking.",
    },
    {
        "query_id": "H2",
        "query": "How does the Gitlab class handle SSL verification?",
        "expected_sources": ["gitlab_client.py"],
        "category": "code_retrieval",
        "notes": "ssl_verify is set in __init__. Tests method-level chunking for __init__.",
    },
    {
        "query_id": "H3",
        "query": "What does the from_config classmethod do?",
        "expected_sources": ["gitlab_client.py"],
        "category": "code_retrieval",
        "notes": "from_config is a classmethod inside the Gitlab class. Tests nested method retrieval via code chunker.",
    },
]


# ─────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────

def context_recall(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Fraction of expected sources present in Top-K results.

    Returns None for out-of-corpus queries (no expected sources).
    """
    if not expected_sources:
        return None  # type: ignore[return-value]

    found = {s for s in retrieved_sources if s in expected_sources}
    return len(found) / len(expected_sources)


def context_precision_mrr(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Mean Reciprocal Rank variant of Context Precision.

    Finds the rank of the *first* hit from any expected source and computes
    1/rank. Returns None for out-of-corpus queries.

    Examples:
      Retrieved [expected, ...] → 1/1 = 1.0
      Retrieved [other, expected, ...] → 1/2 = 0.5
      Retrieved [other, other, other, expected, ...] → 1/4 = 0.25
    """
    if not expected_sources:
        return None  # type: ignore[return-value]

    expected_set = set(expected_sources)
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in expected_set:
            return 1.0 / rank
    return 0.0


def faithfulness(
    answer: str,
    cited_sources: List[str],
    expected_sources: List[str],
    is_out_of_corpus: bool,
) -> float:
    """Faithfulness score.

    - Out-of-corpus queries: 1.0 only if the answer exactly matches the
      standardized refusal sentinel.
    - Normal queries: 1.0 if every cited source is in expected_sources
      (ungrounded hallucination = 0.0).  No citations → 1.0 (conservative).
    """
    if is_out_of_corpus:
        return 1.0 if REFUSAL_SENTINEL in answer.lower() else 0.0

    if not cited_sources:
        return 1.0  # no citations → can't be ungrounded

    expected_set = set(expected_sources)
    ungrounded = [s for s in cited_sources if s not in expected_set]
    return 0.0 if ungrounded else 1.0


def compute_pass(
    cr: float | None,
    cp: float | None,
    f: float,
    is_out_of_corpus: bool,
) -> bool:
    """Determine pass/fail under the new continuous metric standard."""
    if is_out_of_corpus:
        return f == 1.0

    # Require at least partial recall (>=0.5), MRR >=0.5 (i.e. in Top-2),
    # and full faithfulness.
    return (cr is not None and cr >= 0.5
            and cp is not None and cp >= 0.5
            and f == 1.0)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EXTENDED RAG EVALUATION — Ragas/RAG-Triad Metrics")
    print("=" * 70)

    # Initialise pipeline
    print("\nLoading indices...")
    indexer = DualIndexerService(persist=True)
    indexer.load_bm25()
    reranker = RerankerService()
    generator = GroundedGenerator()
    retriever = HybridRetriever(indexer)

    print(f"Qdrant count: {indexer.get_qdrant_count()}")
    print(f"BM25 count:   {indexer.get_bm25_count()}")
    print(f"\nRunning {len(EXTENDED_QUERIES)} test cases...\n")

    results = []
    category_stats: Dict[str, Dict[str, Any]] = {}

    with open("extended_eval_results.txt", "w", encoding="utf-8") as out:
        for q in EXTENDED_QUERIES:
            query_text: str = q["query"]
            expected: List[str] = q["expected_sources"]
            category: str = q["category"]
            qid: str = q["query_id"]
            is_ooc = len(expected) == 0

            # ── Retrieval pipeline ─────────────────────────────────────
            dense_results = retriever.search_dense(query_text, top_n=20)
            sparse_results = retriever.search_sparse(query_text, top_n=20)
            fused_candidates = retriever.fuse_rrf(dense_results, sparse_results, k=60)
            final_results = reranker.rerank(query_text, fused_candidates, top_k=5)

            # ── Generation ─────────────────────────────────────────────
            response = generator.generate_answer(query_text, final_results)

            # ── Metric computation ─────────────────────────────────────
            retrieved_sources = [r.source_id for r in final_results]
            cited_sources = [c.source_id for c in response.citations] if response.citations else []
            answer_text = response.answer

            cr = context_recall(retrieved_sources, expected)
            cp = context_precision_mrr(retrieved_sources, expected)
            f  = faithfulness(answer_text, cited_sources, expected, is_ooc)
            passed = compute_pass(cr, cp, f, is_ooc)

            result_entry = {
                "query_id": qid,
                "query": query_text[:80],
                "category": category,
                "expected_sources": expected,
                "passed": passed,
                "metrics": {
                    "context_recall": cr,
                    "context_precision_mrr": cp,
                    "faithfulness": f,
                },
                "retrieval": {
                    "retrieved_sources": retrieved_sources,
                },
                "faithfulness_detail": {
                    "answer": answer_text[:200],
                    "cited_sources": cited_sources,
                    "is_out_of_corpus": is_ooc,
                    "is_refusal": REFUSAL_SENTINEL in answer_text.lower(),
                },
                "notes": q["notes"],
            }
            results.append(result_entry)

            # Category stats
            if category not in category_stats:
                category_stats[category] = {
                    "total": 0, "passed": 0, "failed": 0,
                    "sum_cr": 0.0, "sum_cp": 0.0, "sum_f": 0.0, "n_metric": 0,
                }
            cs = category_stats[category]
            cs["total"] += 1
            if passed:
                cs["passed"] += 1
            else:
                cs["failed"] += 1
            if cr is not None:
                cs["sum_cr"] += cr
                cs["sum_cp"] += cp
                cs["n_metric"] += 1
            cs["sum_f"] += f

            # ── Per-query output ───────────────────────────────────────
            status = "PASS" if passed else "FAIL"
            out.write(f"{'=' * 60}\n")
            out.write(f"[{status}] {qid} ({category}): {query_text[:60]}...\n")
            out.write(f"Expected sources : {expected}\n")
            out.write(f"Retrieved sources: {retrieved_sources}\n")
            if cr is not None:
                out.write(
                    f"Metrics          : CR={cr:.2f} | MRR={cp:.2f} | Faith={f:.2f}\n"
                )
            else:
                out.write(
                    f"Metrics          : CR=N/A (OOC) | MRR=N/A | Faith={f:.2f}\n"
                )
            out.write(f"Answer           : {answer_text[:150]}\n")
            out.write(f"Cited sources    : {cited_sources}\n")
            out.write(f"Notes            : {q['notes']}\n\n")

            cr_str = f"{cr:.2f}" if cr is not None else "N/A "
            cp_str = f"{cp:.2f}" if cp is not None else "N/A "
            print(f"  [{status}] {qid}: CR={cr_str} MRR={cp_str} F={f:.2f}  — {query_text[:45]}...")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total   = len(results)
    passed_total = sum(1 for r in results if r["passed"])
    failed_total = total - passed_total

    print(f"\nTotal: {total} | Passed: {passed_total} | Failed: {failed_total}")
    print(f"Overall Pass Rate: {passed_total / total * 100:.1f}%\n")

    print(f"{'Category':<22} {'P/T':>6} {'%':>6} {'Avg CR':>8} {'Avg MRR':>9} {'Avg F':>7}")
    print("-" * 62)
    for cat, cs in sorted(category_stats.items()):
        pct = cs["passed"] / cs["total"] * 100
        avg_cr  = cs["sum_cr"]  / cs["n_metric"] if cs["n_metric"] else float("nan")
        avg_cp  = cs["sum_cp"]  / cs["n_metric"] if cs["n_metric"] else float("nan")
        avg_f   = cs["sum_f"]   / cs["total"]
        print(
            f"  {cat:<20} {cs['passed']}/{cs['total']:>1} {pct:>6.0f}%"
            f" {avg_cr:>8.2f} {avg_cp:>9.2f} {avg_f:>7.2f}"
        )

    # Failures detail
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n{'=' * 70}")
        print(f"FAILURE DETAILS ({len(failures)} cases)")
        print(f"{'=' * 70}")
        for f_entry in failures:
            m = f_entry["metrics"]
            cr_str = f"{m['context_recall']:.2f}" if m["context_recall"] is not None else "N/A"
            cp_str = f"{m['context_precision_mrr']:.2f}" if m["context_precision_mrr"] is not None else "N/A"
            print(
                f"\n  [{f_entry['query_id']}] {f_entry['category']}: {f_entry['query']}"
            )
            print(f"    Expected : {f_entry['expected_sources']}")
            print(f"    Retrieved: {f_entry['retrieval']['retrieved_sources']}")
            print(f"    CR={cr_str}  MRR={cp_str}  F={m['faithfulness']:.2f}")
            print(f"    Reason   : {f_entry['notes']}")

    # Save JSON results
    with open("extended_eval_results.json", "w", encoding="utf-8") as jf:
        json.dump(
            {
                "summary": {
                    "total": total,
                    "passed": passed_total,
                    "failed": failed_total,
                    "pass_rate": passed_total / total,
                    "metric_standard": "Ragas/RAG-Triad (CR>=0.5, MRR>=0.5, Faithfulness=1.0)",
                    "category_stats": category_stats,
                },
                "results": results,
            },
            jf,
            indent=2,
            default=str,
        )

    print(f"\nResults saved to extended_eval_results.txt and extended_eval_results.json")


if __name__ == "__main__":
    main()
