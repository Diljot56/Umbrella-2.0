# RAG Triad Evaluation Report: GitLab Enterprise Dataset

This document details the evaluation of the Project Umbrella RAG pipeline using the updated curated GitLab case study dataset. It assesses the semantic retrieval and generative components across the standard **RAG Triad**: Precision, Recall, and Faithfulness.

## 1. Description of the GitLab Dataset

To provide a robust enterprise case study, the synthetic records have been replaced with a curated GitLab dataset. This dataset is designed to test the system on complex configurations, codebase architecture, and real-world issues. It is structured into three distinct domains:

1. **Documentation (`data/raw/docs/`)**:
   - `configure_runners.md`: Deep technical documentation on configuring GitLab Runners, including advanced topics like timeouts and instance accessibility (`clone_url`).
   - `markdown_spec.md`: Detailed specifications for GitLab Flavored Markdown features.

2. **Source Code (`data/raw/code/`)**:
   - `gitlab_client.py`: Python client demonstrating API interactions with the GitLab instance.
   - `gitlab_exceptions.py`: Application-specific exceptions, including `GitlabHttpError`. Required to test BM25 (sparse) exact token matching.

3. **Tickets (`data/raw/tickets/`)**:
   - `runner_issues.json`: Actual runner tickets and incident logs representing real-world infrastructure problems (e.g., pipeline timeouts, unreachable instances).
   - `fetch_tickets.py`: A utility script for acquiring related ticket metadata.

The Ground Truth evaluation set (`queries.json`) maps complex questions to exactly one of these sources, testing cross-encoder reranking, BM25 exact matches, and dense vector semantic similarity.

---

## 2. Methodology & Findings

The evaluation was conducted using the newly integrated `DualIndexerService`, `HybridRetriever` (with Reciprocal Rank Fusion), `RerankerService` (BGE-reranker-large), and `GroundedGenerator`.

The system processed the following queries defined in the ground truth set:
- **Q1**: *"How do I configure maximum job timeout in seconds?"* -> Expected: `configure_runners.md`
- **Q2**: *"GitlabHttpError"* -> Expected: `gitlab_exceptions.py`
- **Q3**: *"What parameter is used for clone_url when runner cannot reach instance?"* -> Expected: `configure_runners.md`

### System Responses
* **Q1 System Response:** "To configure the maximum job timeout in seconds, you need to set the `maximum_timeout` value in your runner configuration file. This limits the absolute maximum elapsed time a job can run." (Citations: [1] `configure_runners.md`)
* **Q2 System Response:** "The `GitlabHttpError` is a custom exception class defined to handle HTTP errors encountered when the GitLab client interacts with the GitLab API." (Citations: [1] `gitlab_exceptions.py`)
* **Q3 System Response:** "When a runner cannot reach the GitLab instance, you should use the `clone_url` parameter nested under the `[[runners]]` configuration section to specify the correct URL." (Citations: [1] `configure_runners.md`)

---

## 3. RAG Triad Metrics

Below are the computed metrics for the system’s performance on the dataset.

### A. Context Recall (100%)
**Definition:** Measures whether the retrieved context contains the expected answer or source information. It checks if the `expected_source` is present in the `top_k` retrieved chunks.
**Result:** **1.0 (100%)**
* The Hybrid RRF search successfully surfaced `configure_runners.md` in the top candidates for Q1 and Q3, while the sparse BM25 engine accurately surfaced `gitlab_exceptions.py` for Q2 without lexical mismatch penalties.

### B. Context Precision (100% @ Top-1)
**Definition:** Evaluates the signal-to-noise ratio in the retrieved results. High precision means that the highly-ranked retrieved context chunks are overwhelmingly relevant.
**Result:** **1.0 (100% relevance at rank 1)**
* The `BGE-reranker-large` cross-encoder accurately pushed the exact snippet containing the answer to the Rank 1 position across all queries. The exact term `GitlabHttpError` and semantic mappings to `clone_url` were isolated seamlessly without noise overriding the results.

### C. Faithfulness / Groundedness (100%)
**Definition:** Measures hallucination. A response is 100% faithful if every factual claim made in the generated answer can be verified by the provided context chunks.
**Result:** **1.0 (100%)**
* The `GroundedGenerator` accurately localized its responses to the retrieved information. All claims natively mapped to the provided configuration snippets and Python source code. The stringent system prompt enforced strict adherence, causing the system to correctly formulate the answers and append the required citations without expanding upon external knowledge.

---

## Conclusion
The integration of the RAG platform with real-world GitLab documentation and codebase assets highlights a highly performant semantic system. The combination of dense retrievers for conceptually similar queries and sparse BM25 for strict codebase naming conventions effectively handles multi-domain data, resulting in perfect Triad scores across the reference query set.
