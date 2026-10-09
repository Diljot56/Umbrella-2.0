import json
from typing import List
from pathlib import Path
from src.ingestion.base import BaseChunker
from src.models.chunk import Chunk

class TicketChunker(BaseChunker):
    """Parses JSON ticket arrays, grouping the issue description and its comment thread together as a single conversational unit."""
    
    def process(self, file_path: Path, **kwargs) -> List[Chunk]:
        chunks = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                if isinstance(data, dict):
                    data = [data]
                else:
                    return chunks
                    
            for item in data:
                ticket_id = item.get("id", item.get("ticket_id", "UnknownID"))
                description = item.get("description", "")
                title = item.get("title", f"Ticket {ticket_id}")
                
                text_parts = [f"Ticket: {title}\nDescription: {description}"]
                
                comments = item.get("comments", [])
                for idx, c in enumerate(comments):
                    author = c.get("author", "UnknownAuthor")
                    body = c.get("body", "")
                    text_parts.append(f"Comment {idx+1} ({author}): {body}")
                    
                full_text = "\n\n".join(text_parts).strip()
                
                if full_text:
                    location = f"Ticket-{ticket_id}"
                    chunks.append(Chunk(
                        chunk_id=Chunk.generate_id(file_path.name, location),
                        source_id=file_path.name,
                        source_type="ticket",
                        location=location,
                        text=full_text,
                        metadata={"ticket_id": ticket_id, "num_comments": len(comments), "title": title}
                    ))
                    
        except Exception as e:
            print(f"Error parsing JSON ticket {file_path}: {e}")
            
        return chunks
