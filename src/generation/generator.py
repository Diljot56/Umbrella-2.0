import os
import re
from typing import List, Tuple, Dict
from src.models.search import SearchResultItem
from src.models.generation import AskResponse, Citation

NO_ANSWER = "No supported answer could be found in the provided sources."

SYSTEM_INSTRUCTION = (
    "You are a verifiable enterprise assistant. Answer the user query strictly "
    "using only the supplied context passages. For every factual claim, append "
    "the citation marker corresponding to the source passage (e.g. [1], [2]). "
    "Never use external knowledge or extrapolate. If the context does not "
    "contain sufficient facts to answer the question, state exactly: "
    f"'{NO_ANSWER}'"
)

class MockLLMClient:
    """Fallback client for offline / unit-testing scenarios."""

    def generate(self, system: str, user: str) -> str:
        # Simulate refusal when the context signals low confidence or is absent
        if "__LOW_CONFIDENCE__" in system:
            return NO_ANSWER
        if "Content:" in system and "Source:" in system:
            return (
                "This is a detailed and realistic answer based on the context. "
                "It uses source [1] to verify claims."
            )
        return NO_ANSWER


class GeminiClient:
    """Lightweight wrapper around the Google Generative AI SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, system: str, user: str) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
            ),
            contents=user,
        )
        return response.text


class GroundedGenerator:
    """Formats context, calls the LLM, parses citations."""

    DEFAULT_API_KEY = "AQ.Ab8RN6If6hzkVOc7F_rtx9B81Bn80Vlx2882TnOaF_MPxRFpGAi"

    def __init__(self):
        api_key = os.getenv("LLM_API_KEY", "") or self.DEFAULT_API_KEY
        use_mock = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

        if api_key and not use_mock:
            try:
                self.llm = GeminiClient(api_key=api_key)
                self.is_mock = False
                print("GroundedGenerator: Using Gemini API")
            except Exception as e:
                print(f"GroundedGenerator: Gemini init failed ({e}), using MockLLMClient")
                self.llm = MockLLMClient()
                self.is_mock = True
        else:
            self.llm = MockLLMClient()
            self.is_mock = True
            print("GroundedGenerator: Using MockLLMClient (set LLM_API_KEY to enable Gemini)")

    def format_context(
        self, candidates: List[SearchResultItem]
    ) -> Tuple[str, Dict[str, SearchResultItem]]:
        if not candidates:
            return "No provided context.", {}

        context_blocks = []
        marker_map: Dict[str, SearchResultItem] = {}
        for idx, item in enumerate(candidates):
            marker = f"[{idx + 1}]"
            marker_map[marker] = item
            block = (
                f"{marker} Source: {item.source_id} | Location: {item.location}\n"
                f"Content: {item.text}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks), marker_map

    def generate_answer(
        self, query: str, candidates: List[SearchResultItem]
    ) -> AskResponse:
        # Check whether the reranker flagged these results as low-confidence
        low_confidence = bool(
            candidates
            and candidates[0].metadata
            and candidates[0].metadata.get("low_confidence", False)
        )

        formatted_context, marker_map = self.format_context(candidates)

        confidence_tag = "\n__LOW_CONFIDENCE__" if low_confidence else ""
        system_prompt = f"{SYSTEM_INSTRUCTION}{confidence_tag}\n\nContext:\n{formatted_context}"

        try:
            raw_answer = self.llm.generate(system=system_prompt, user=query)
        except Exception as e:
            print(f"LLM generation failed ({e}), using fallback.")
            # Fallback: generate a basic answer from the top chunks
            raw_answer = self._build_fallback_answer(candidates)

        # Parse and deduplicate citations
        citations: List[Citation] = []
        seen_markers = set()

        for m in re.findall(r"\[(\d+)\]", raw_answer):
            marker_key = f"[{m}]"
            if marker_key in marker_map and marker_key not in seen_markers:
                seen_markers.add(marker_key)
                item = marker_map[marker_key]

                snippet = item.text[:200]
                if len(item.text) > 200:
                    snippet += "..."

                citations.append(
                    Citation(
                        marker=marker_key,
                        chunk_id=item.chunk_id,
                        source_id=item.source_id,
                        source_type=item.source_type,
                        location=item.location,
                        snippet=snippet,
                    )
                )

        return AskResponse(
            query=query,
            answer=raw_answer,
            citations=citations,
            retrieved_chunks=candidates,
        )

    def _build_fallback_answer(self, candidates: List[SearchResultItem]) -> str:
        """Build a fallback answer from retrieved chunks when LLM fails."""
        if not candidates:
            return "No supported answer could be found in the provided sources."

        parts = ["**Based on the retrieved sources:**\n"]
        for i, c in enumerate(candidates[:5]):
            snippet = c.text[:300].strip()
            if len(c.text) > 300:
                snippet += "..."
            parts.append(
                f"**[{i+1}]** *{c.source_id}* — {c.location}\n\n{snippet}\n"
            )

        parts.append(
            "\n> *Note: The AI summarizer was unavailable. "
            "Showing raw retrieved passages with scores in the diagnostics panel.*"
        )
        return "\n".join(parts)
