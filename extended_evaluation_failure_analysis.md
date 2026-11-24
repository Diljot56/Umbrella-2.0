# Extended RAG Evaluation — Failure Analysis Report

This report documents the results of an **extended stress-test** of the Project Umbrella RAG pipeline. The original `evaluator.py` tested only 3 carefully curated queries and reported **100% recall, precision, and faithfulness**. This extended evaluation (`scripts/extended_evaluator.py`) added **22 adversarial and edge-case queries** across 8 categories to discover where the pipeline actually breaks.

---

## Executive Summary

| Metric | Original (3 queries) | Extended (22 queries) |
|--------|----------------------|-----------------------|
| **Pass Rate** | 100% (3/3) | **59.1% (13/22)** |
| **Failures** | 0 | **9** |

The pipeline fails on **9 out of 22** test cases spanning four distinct failure categories. Three categories scored **0% pass rate**: cross-source retrieval, out-of-corpus hallucination detection, and nested code retrieval.

---

## Category Breakdown

| Category | Passed | Failed | Pass Rate | Verdict |
|----------|--------|--------|-----------|---------|
| Lexical trap (exact tokens) | 3/3 | 0 | **100%** | ✅ Strong |
| Multi-hop reasoning | 2/2 | 0 | **100%** | ✅ Strong |
| Markdown spec retrieval | 3/3 | 0 | **100%** | ✅ Strong |
| Short query | 2/2 | 0 | **100%** | ✅ Strong |
| Long query | 1/1 | 0 | **100%** | ✅ Strong |
| Ticket retrieval | 2/3 | 1 | 67% | ⚠️ Partial |
| Cross-source retrieval | 0/2 | 2 | **0%** | ❌ Broken |
| Out-of-corpus (hallucination) | 0/3 | 3 | **0%** | ❌ Broken |
| Nested code retrieval | 0/3 | 3 | **0%** | ❌ Broken |

---

## Detailed Failure Analysis

### Failure Category 1: Out-of-Corpus Queries — No Abstention Mechanism (B1, B2, B3)

**Queries that failed:**
- **B1**: *"How do I configure GitLab Runner on Kubernetes with Helm?"*
- **B2**: *"What is the default Docker image used by GitLab CI?"*
- **B3**: *"How to set up GitLab Pages with a custom domain?"*

**What happened:**
All three queries ask about topics that **do not exist in the corpus**. Kubernetes/Helm setup, default Docker images, and GitLab Pages custom domain configuration are not covered in any of the indexed documents. Yet the system retrieved `configure_runners.md` chunks (which are tangentially related to "runners" and "GitLab") and the MockLLM generated a confident answer citing those irrelevant chunks.

**Root cause — Two compounding architectural flaws:**

1. **No relevance threshold in retrieval.** The `HybridRetriever.fuse_rrf()` method always returns the top-k candidates by RRF score — it never checks whether those candidates are actually relevant. In a vector-based retrieval system, *every* query has a nearest neighbor, even if that neighbor is semantically distant. There is no minimum similarity threshold, no confidence gate, and no mechanism to say "nothing relevant was found."

2. **MockLLM cannot refuse.** The `MockLLMClient.generate()` only checks whether `"Content:"` and `"Source:"` are present in the system prompt. Since the retriever always returns chunks, the context always contains those strings, so the mock always produces a confident-sounding answer. Even with Gemini enabled, the generator's system prompt does instruct the LLM to refuse when context is insufficient — but the retriever always provides _something_, making it hard for the LLM to judge insufficiency.

**Why the original evaluation missed this:** The original 3 queries all had answers that were present in the corpus. No out-of-corpus query was tested.

**Recommended fix:** Add a minimum RRF or cross-encoder score threshold. When the best candidate's cross-encoder score falls below a configurable threshold (e.g., 0.3), the system should respond with "No supported answer could be found in the provided sources" instead of fabricating an answer from tangentially related chunks.

---

### Failure Category 2: Cross-Source Retrieval — Source Diversity Collapse (A1, A2)

**Queries that failed:**
- **A1**: *"How are HTTP errors handled in the GitLab Python client?"* — Retrieved: `gitlab_exceptions.py` ×3 + `configure_runners.md` + `markdown_spec.md`. **Missing: `gitlab_client.py`** (recall: 50%)
- **A2**: *"What happens when a runner token is compromised?"* — Retrieved: `configure_runners.md` ×5. **Missing: `runner_issues.json`** (recall: 50%)

**What happened:**
When a query is relevant to **multiple source documents**, the pipeline's top-5 slots are monopolized by a single dominant source. For A1, `gitlab_exceptions.py` (which defines `GitlabHttpError` and many error classes) drowned out `gitlab_client.py` (which *uses* those errors in `http_request`). For A2, `configure_runners.md` (which discusses token security at length across multiple sections) filled all 5 slots, completely excluding the ticket `runner_issues.json` (which documents a real-world prefixed-token bug).

**Root cause — No source diversification in fusion or reranking:**

1. **RRF fusion has no deduplication or diversity constraint.** When chunks from the same source all score highly in both dense and sparse retrievers, they accumulate RRF scores and dominate the fused candidates list. The `fuse_rrf()` method sorts purely by score with no mechanism to ensure diversity across source documents.

2. **Cross-encoder reranker amplifies the bias.** The `RerankerService.rerank()` takes only the top 5 candidates from the RRF-fused list (`max_rerank_pool=5`) and re-sorts them. If those top 5 are already from the same source, the reranker has no opportunity to surface chunks from the missing source.

3. **Document-level chunking creates many chunks per large source.** `configure_runners.md` (1,197 lines) produces ~50+ chunks split by H1/H2/H3 headers. `runner_issues.json` produces only 5 chunks (one per ticket). The large document has a massive numerical advantage in RRF simply because it contributes more candidates.

**Why the original evaluation missed this:** All 3 original queries targeted a single expected source. No query required retrieving from multiple sources simultaneously.

**Recommended fix:** Implement a **Source-Aware Maximal Marginal Relevance (MMR)** strategy: after RRF fusion, enforce that the top-k results contain at most N chunks from any single source (e.g., max 2–3 per source), allowing less-dominant but relevant sources to surface in the remaining slots.

---

### Failure Category 3: Nested Code Retrieval — CodeChunker Architectural Limitation (H1, H2, H3)

**Queries that failed:**
- **H1**: *"What is the http_request method signature in the Gitlab client?"* — Retrieved `gitlab_client.py` at rank 2 but cited `gitlab_exceptions.py` at rank 1. Precision@1 = 0%.
- **H2**: *"How does the Gitlab class handle SSL verification?"* — **Never retrieved `gitlab_client.py` at all.** Recall = 0%, Precision@1 = 0%.
- **H3**: *"What does the from_config classmethod do?"* — Retrieved `gitlab_client.py` at rank 2 and 3, but cited `gitlab_exceptions.py` at rank 1. Precision@1 = 0%.

**What happened:**
The `CodeChunker` uses Python's `ast` module and only extracts **top-level** AST nodes (`ast.iter_child_nodes(tree)`). In `gitlab_client.py`, the class `Gitlab` is a top-level node and gets chunked as a single enormous block (spanning ~1,400 lines). The individual methods inside it — `http_request`, `_set_auth_info`, `from_config`, `_get_session_opts`, etc. — are **nested inside the class** and are never extracted as separate chunks.

This creates two problems:

1. **The Gitlab class chunk is too large for the embedding model.** A single chunk containing ~1,400 lines far exceeds the embedding model's effective context window. The `BAAI/bge-large-en-v1.5` encoder produces a single 1024-dimensional vector for this massive text. Critical details like `ssl_verify` or `http_request` are diluted in the averaged embedding, making the chunk semantically weak for specific method-level queries. This is why H2 ("SSL verification") returns **0% recall** — the diluted embedding doesn't match the query semantically.

2. **The cross-encoder's `max_length=192` truncates the chunk.** The `RerankerService` calls `encoder.predict(pairs, max_length=192)`. For the massive Gitlab class chunk, the cross-encoder only sees the first ~192 tokens (roughly the `__init__` docstring and the first few constructor parameters). Methods defined later in the class (like `http_request` at line 636 or `from_config` at line 258) are never seen by the reranker, so their relevance is never scored.

3. **Smaller but irrelevant chunks from `gitlab_exceptions.py` rank higher.** The code chunker correctly splits `gitlab_exceptions.py` into ~50 small, focused chunks (one per exception class). These small chunks embed well and score highly for queries containing words like "http", "error", "client" — outranking the diluted `gitlab_client.py` mega-chunk.

**Why the original evaluation missed this:** The original Q2 ("GitlabHttpError") targeted `gitlab_exceptions.py`, which the chunker handles well because each short class definition is a clean top-level AST node. No original query targeted methods *inside* a class.

**Recommended fix:** Modify `CodeChunker` to recursively walk class bodies and extract individual methods/classmethods as separate chunks. For each `ast.ClassDef`, iterate over its `body` and emit a chunk per method:

```python
for node in ast.iter_child_nodes(tree):
    if isinstance(node, ast.ClassDef):
        # Chunk the class docstring + signature separately
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Emit as: "class Gitlab → method http_request"
                ...
```

Additionally, increase `max_length` in the cross-encoder from 192 to at least 512 for code chunks to reduce truncation.

---

### Failure Category 4: Ticket Precision — Reranker Demotes Tickets (E1)

**Query that failed:**
- **E1**: *"What is the IPv6 health check bug in GitLab Runner Docker executor?"* — Ticket `runner_issues.json` was retrieved at **rank 2**, but `configure_runners.md` occupied rank 1. Precision@1 = 0%.

**What happened:**
The ticket text for GL-39939 (IPv6 health check bug) is a densely technical description buried inside a large JSON-derived text block. The retriever found it (recall = 100%), but the cross-encoder promoted a `configure_runners.md` chunk to rank 1 because that documentation chunk contains common terms like "Docker", "runner", and "executor" in well-structured prose — exactly the terms in the query.

**Root cause:**

1. **Ticket text is structurally noisy.** The `TicketChunker` concatenates title, description, and all comments into a single chunk, prefixed with synthetic headers like `"Ticket: ..."` and `"Comment 1 (davux): ..."`. This formatting creates a long, noisy text block that embeds and reranks poorly compared to clean documentation paragraphs.

2. **The cross-encoder favors well-formed prose.** Cross-encoders like `BAAI/bge-reranker-large` are trained primarily on natural language question-answer pairs. They tend to score clean documentation prose higher than ticket-style conversational text with usernames, Git commit references, code blocks, and inline metadata.

3. **The `max_rerank_pool=5` cap may also limit exploration.** If the ticket chunk was at position 6+ in the RRF list, it would never reach the reranker and would get the sentinel score `-999.0`.

**Why the original evaluation missed this:** No original query targeted ticket content. All 3 original queries targeted docs or code.

**Recommended fix:** Apply source-type boosting in the reranker for query-source alignment. If the query mentions "bug", "issue", or "error" in a contextual way, slightly boost ticket-sourced candidates. Alternatively, splitting tickets into separate chunks for title/description vs. comments would give the title chunk a cleaner embedding.

---

## Why the Original Evaluation Showed 100%

The original evaluator (`scripts/evaluator.py`) tested exactly **3 queries** that were specifically crafted to align with the pipeline's strengths:

| Property | Original Queries | Extended Queries |
|----------|-----------------|-----------------|
| Sources per query | Always 1 | 1 or 2 (cross-source) |
| Answers in corpus? | Always yes | 3 queries intentionally out-of-corpus |
| Code target | Top-level class (exceptions) | Nested methods inside class |
| Ticket queries | None | 3 queries |
| Query length | Moderate (6–12 words) | 1 word to 50+ words |

Furthermore, the mock LLM always returns a canned answer with `[1]` citation, making faithfulness trivially pass whenever the top-1 retrieved chunk is from the expected source. The original evaluation was a **best-case confirmation test**, not a rigorous stress test.

---

## Summary of Root Causes

| # | Root Cause | Failed Queries | Impact |
|---|-----------|---------------|--------|
| 1 | **No relevance threshold** — retriever always returns top-k regardless of actual relevance | B1, B2, B3 | System hallucinates answers to questions not in corpus |
| 2 | **No source diversification** — RRF fusion and reranker allow one dominant source to monopolize all slots | A1, A2 | Incomplete answers that miss relevant secondary sources |
| 3 | **CodeChunker only extracts top-level AST nodes** — class methods are not individually chunked | H1, H2, H3 | Cannot retrieve or rank specific methods inside large Python classes |
| 4 | **Cross-encoder `max_length=192` truncation** — large code chunks are truncated to ~192 tokens | H1, H2, H3 | Reranker cannot assess relevance of content beyond the first ~40 lines |
| 5 | **MockLLM cannot detect irrelevance** — always answers confidently if context is provided | B1, B2, B3 | All out-of-corpus tests fail faithfulness |
| 6 | **Ticket text is structurally noisy** — concatenated author/body format reranks poorly against clean docs | E1 | Ticket content loses precision@1 to documentation chunks |

---

## Recommended Priority Fixes

1. **🔴 Critical — Add relevance thresholding.** Gate the generation pipeline with a minimum cross-encoder score. If no candidate passes the threshold, return "No supported answer found."

2. **🔴 Critical — Fix CodeChunker to extract class methods.** Walk into `ast.ClassDef.body` and emit method-level chunks. This is a ~15-line code change with massive retrieval improvement.

3. **🟡 High — Add source diversity to RRF fusion.** Cap chunks per source in the top-k, or implement MMR (Maximal Marginal Relevance) to balance relevance with diversity.

4. **🟡 High — Increase cross-encoder `max_length`.** Raise from 192 to 512 tokens for code-type chunks to reduce truncation.

5. **🟢 Medium — Improve ticket chunking.** Split tickets into title+description vs. comments as separate chunks for cleaner embeddings.

6. **🟢 Medium — Expand ground truth test suite.** Replace the 3-query evaluation with the 22-query extended suite to prevent regression.
