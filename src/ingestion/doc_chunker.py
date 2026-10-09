import re
from typing import List
from pathlib import Path
from src.ingestion.base import BaseChunker
from src.models.chunk import Chunk

try:
    import fitz # PyMuPDF
except ImportError:
    fitz = None

class DocChunker(BaseChunker):
    def process(self, file_path: Path, **kwargs) -> List[Chunk]:
        if file_path.suffix.lower() == '.md':
            return self._process_markdown(file_path)
        elif file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        else:
            return []

    def _process_markdown(self, file_path: Path) -> List[Chunk]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split on headers H1, H2, H3
        header_pattern = re.compile(r'^(#{1,3})\s+(.*)$', re.MULTILINE)
        
        chunks = []
        matches = list(header_pattern.finditer(content))
        
        if not matches:
            if content.strip():
                location = "Entire Document"
                chunks.append(Chunk(
                    chunk_id=Chunk.generate_id(file_path.name, location),
                    source_id=file_path.name,
                    source_type="document",
                    location=location,
                    text=content.strip(),
                    metadata={}
                ))
            return chunks

        first_match = matches[0]
        if first_match.start() > 0:
            pre_text = content[:first_match.start()].strip()
            if pre_text:
                location = "Intro"
                chunks.append(Chunk(
                    chunk_id=Chunk.generate_id(file_path.name, location),
                    source_id=file_path.name,
                    source_type="document",
                    location=location,
                    text=pre_text,
                    metadata={}
                ))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            
            start_pos = match.end()
            end_pos = matches[i+1].start() if i + 1 < len(matches) else len(content)
            
            body = content[start_pos:end_pos].strip()
            full_chunk_text = f"{match.group(0)}\n\n{body}" if body else match.group(0)
            
            location = f"H{level} - {title}"
            
            chunks.append(Chunk(
                chunk_id=Chunk.generate_id(file_path.name, location),
                source_id=file_path.name,
                source_type="document",
                location=location,
                text=full_chunk_text,
                metadata={"level": level, "title": title}
            ))
            
        return chunks

    def _process_pdf(self, file_path: Path) -> List[Chunk]:
        chunks = []
        if not fitz:
            print("PyMuPDF not installed, cannot parse PDF")
            return chunks
            
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text").strip()
                if not text:
                    continue
                location = f"Page {page_num + 1}"
                chunks.append(Chunk(
                    chunk_id=Chunk.generate_id(file_path.name, location),
                    source_id=file_path.name,
                    source_type="document",
                    location=location,
                    text=text,
                    metadata={"page": page_num + 1}
                ))
            doc.close()
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            
        return chunks
