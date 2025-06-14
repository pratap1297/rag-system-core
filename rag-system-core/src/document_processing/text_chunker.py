from typing import List, Any, Dict, Optional
from enum import Enum

class ChunkBoundary:
    pass

class ChunkQuality:
    pass

class DocumentType(Enum):
    GENERAL = "general"

class BoundaryType(Enum):
    SIZE_LIMIT = "size_limit"

class TokenCounter:
    pass

class BoundaryDetector:
    pass

class QualityAssessor:
    pass

class TextChunk:
    pass

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.overlap
        return chunks

def create_text_chunker(config: Optional[Dict[str, Any]] = None) -> TextChunker:
    return TextChunker() 