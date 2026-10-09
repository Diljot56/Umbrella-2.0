from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from src.models.chunk import Chunk

class BaseChunker(ABC):
    @abstractmethod
    def process(self, file_path: Path, **kwargs) -> List[Chunk]:
        """Process a file and return a list of Chunks."""
        pass
