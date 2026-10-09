import os
import sys
import pytest
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.search import SearchResultItem
from src.generation.generator import GroundedGenerator

@pytest.fixture
def mock_generator():
    os.environ["USE_MOCK_LLM"] = "true"
    generator = GroundedGenerator()
    return generator

def test_format_context(mock_generator):
    candidates = [
        SearchResultItem(chunk_id="c1", text="Text 1", source_id="a.md", source_type="document", location="L1", metadata={}),
        SearchResultItem(chunk_id="c2", text="Text 2", source_id="b.py", source_type="code", location="L2", metadata={})
    ]
    
    formatted, marker_map = mock_generator.format_context(candidates)
    
    assert "[1] Source: a.md" in formatted
    assert "[2] Source: b.py" in formatted
    assert "[1]" in marker_map
    assert "[2]" in marker_map
    assert marker_map["[1]"].chunk_id == "c1"

def test_generate_answer_with_candidates(mock_generator):
    candidates = [
        SearchResultItem(chunk_id="c1", text="A very long text to test snippet truncation so we need at least 200 characters of dummy content here in this text property just to be absolutely sure that we trigger the truncation logic properly in our parsing routine natively and flawlessly without fail yes indeed.", source_id="a.md", source_type="doc", location="L1", metadata={})
    ]
    
    response = mock_generator.generate_answer("query?", candidates)
    
    assert "uses source [1]" in response.answer
    assert len(response.citations) == 1
    
    citation = response.citations[0]
    assert citation.marker == "[1]"
    assert citation.chunk_id == "c1"
    # Verify snippet truncation to 200 + '...'
    assert len(citation.snippet) == 203
    assert citation.snippet.endswith("...")

def test_generate_answer_deduplication(mock_generator):
    generator = mock_generator
    # force raw answer logic
    def mock_completion(messages, **kwargs):
         return "This is a fact [1]. This is another fact [1]. And [2]."
    
    generator.client.chat_completions_create = mock_completion
    
    candidates = [
        SearchResultItem(chunk_id="c1", text="Text 1", source_id="a", source_type="b", location="c", metadata={}),
        SearchResultItem(chunk_id="c2", text="Text 2", source_id="a", source_type="b", location="c", metadata={})
    ]
    
    response = generator.generate_answer("query", candidates)
    
    assert len(response.citations) == 2 # Only one [1] and one [2]
    markers = [c.marker for c in response.citations]
    assert "[1]" in markers
    assert "[2]" in markers

def test_fallback_behavior_empty_candidates(mock_generator):
    response = mock_generator.generate_answer("What is Umbrella?", [])
    
    assert "No supported answer could be found in the provided sources." in response.answer
    assert len(response.citations) == 0
