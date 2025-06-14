"""
Advanced Chunking Strategies for RAG System
Provides multiple text chunking approaches for optimal retrieval
"""

import re
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from core.logging_system import get_logger
from core.monitoring import get_performance_monitor

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    overlap_with_previous: int = 0
    overlap_with_next: int = 0

class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("chunking")
        self.monitor = get_performance_monitor()
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Chunk text into segments"""
        pass
    
    def _generate_chunk_id(self, base_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        return f"{base_id}_chunk_{chunk_index:04d}"
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

class FixedSizeChunker(ChunkingStrategy):
    """Fixed-size chunking with overlap"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200, **kwargs):
        super().__init__(kwargs.get('config'))
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.logger.info(f"Fixed-size chunker initialized: size={chunk_size}, overlap={overlap}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Chunk text into fixed-size segments with overlap"""
        
        if not text.strip():
            return []
        
        text = self._clean_text(text)
        chunks = []
        base_id = metadata.get('document_id', 'doc') if metadata else 'doc'
        
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Try to break at word boundary if not at end of text
            if end < len(text):
                # Look for last space within reasonable distance
                last_space = text.rfind(' ', start, end)
                if last_space > start + self.chunk_size * 0.8:  # At least 80% of chunk size
                    end = last_space
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_metadata = {
                    'chunk_index': chunk_index,
                    'chunk_method': 'fixed_size',
                    'chunk_size': len(chunk_text),
                    'original_start': start,
                    'original_end': end
                }
                
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunk = TextChunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    chunk_id=self._generate_chunk_id(base_id, chunk_index),
                    metadata=chunk_metadata,
                    overlap_with_previous=self.overlap if chunk_index > 0 else 0
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap)
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        self.logger.info(f"Fixed-size chunking completed: {len(chunks)} chunks")
        return chunks

class SentenceChunker(ChunkingStrategy):
    """Sentence-based chunking"""
    
    def __init__(self, max_sentences: int = 5, max_chars: int = 1500, **kwargs):
        super().__init__(kwargs.get('config'))
        self.max_sentences = max_sentences
        self.max_chars = max_chars
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        
        self.logger.info(f"Sentence chunker initialized: max_sentences={max_sentences}, max_chars={max_chars}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Chunk text by sentences"""
        
        if not text.strip():
            return []
        
        text = self._clean_text(text)
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        base_id = metadata.get('document_id', 'doc') if metadata else 'doc'
        
        current_chunk = []
        current_length = 0
        chunk_index = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed limits
            if (len(current_chunk) >= self.max_sentences or 
                current_length + sentence_length > self.max_chars) and current_chunk:
                
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                
                chunk_metadata = {
                    'chunk_index': chunk_index,
                    'chunk_method': 'sentence',
                    'sentence_count': len(current_chunk),
                    'chunk_size': len(chunk_text)
                }
                
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunk = TextChunk(
                    text=chunk_text,
                    start_index=start_pos,
                    end_index=end_pos,
                    chunk_id=self._generate_chunk_id(base_id, chunk_index),
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
                
                # Reset for next chunk
                start_pos = end_pos
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            
            chunk_metadata = {
                'chunk_index': chunk_index,
                'chunk_method': 'sentence',
                'sentence_count': len(current_chunk),
                'chunk_size': len(chunk_text)
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            chunk = TextChunk(
                text=chunk_text,
                start_index=start_pos,
                end_index=start_pos + len(chunk_text),
                chunk_id=self._generate_chunk_id(base_id, chunk_index),
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
        
        self.logger.info(f"Sentence chunking completed: {len(chunks)} chunks")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        
        # Split by sentence endings
        sentences = self.sentence_endings.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

class SemanticChunker(ChunkingStrategy):
    """Semantic chunking using embeddings (placeholder for now)"""
    
    def __init__(self, similarity_threshold: float = 0.8, max_chunk_size: int = 1500, **kwargs):
        super().__init__(kwargs.get('config'))
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        
        self.logger.info(f"Semantic chunker initialized: threshold={similarity_threshold}")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Semantic chunking (simplified implementation)"""
        
        # For now, fall back to sentence-based chunking
        # In a full implementation, this would use embeddings to group semantically similar sentences
        
        sentence_chunker = SentenceChunker(max_chars=self.max_chunk_size)
        chunks = sentence_chunker.chunk_text(text, metadata)
        
        # Update metadata to indicate semantic method
        for chunk in chunks:
            chunk.metadata['chunk_method'] = 'semantic_fallback'
        
        self.logger.info(f"Semantic chunking completed: {len(chunks)} chunks (fallback mode)")
        return chunks

class HierarchicalChunker(ChunkingStrategy):
    """Hierarchical chunking with multiple levels"""
    
    def __init__(self, levels: List[Dict[str, Any]] = None, **kwargs):
        super().__init__(kwargs.get('config'))
        
        # Default hierarchical levels
        self.levels = levels or [
            {'name': 'paragraph', 'max_size': 500, 'separator': '\n\n'},
            {'name': 'sentence', 'max_size': 150, 'separator': '. '},
            {'name': 'phrase', 'max_size': 50, 'separator': ', '}
        ]
        
        self.logger.info(f"Hierarchical chunker initialized with {len(self.levels)} levels")
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """Hierarchical chunking with multiple levels"""
        
        if not text.strip():
            return []
        
        text = self._clean_text(text)
        chunks = []
        base_id = metadata.get('document_id', 'doc') if metadata else 'doc'
        
        # Start with the full text as level 0
        current_segments = [{'text': text, 'start': 0, 'end': len(text), 'level': 0}]
        chunk_index = 0
        
        for level_idx, level_config in enumerate(self.levels):
            new_segments = []
            
            for segment in current_segments:
                if len(segment['text']) <= level_config['max_size']:
                    # Segment is small enough, keep as is
                    new_segments.append(segment)
                else:
                    # Split segment
                    sub_segments = self._split_segment(
                        segment['text'], 
                        level_config['separator'],
                        level_config['max_size'],
                        segment['start'],
                        level_idx + 1
                    )
                    new_segments.extend(sub_segments)
            
            current_segments = new_segments
        
        # Convert segments to chunks
        for segment in current_segments:
            if segment['text'].strip():
                chunk_metadata = {
                    'chunk_index': chunk_index,
                    'chunk_method': 'hierarchical',
                    'hierarchy_level': segment['level'],
                    'chunk_size': len(segment['text'])
                }
                
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunk = TextChunk(
                    text=segment['text'],
                    start_index=segment['start'],
                    end_index=segment['end'],
                    chunk_id=self._generate_chunk_id(base_id, chunk_index),
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        self.logger.info(f"Hierarchical chunking completed: {len(chunks)} chunks")
        return chunks
    
    def _split_segment(self, text: str, separator: str, max_size: int, start_offset: int, level: int) -> List[Dict[str, Any]]:
        """Split a segment by separator"""
        
        if separator not in text:
            return [{'text': text, 'start': start_offset, 'end': start_offset + len(text), 'level': level}]
        
        parts = text.split(separator)
        segments = []
        current_pos = start_offset
        
        for part in parts:
            part = part.strip()
            if part:
                segments.append({
                    'text': part,
                    'start': current_pos,
                    'end': current_pos + len(part),
                    'level': level
                })
                current_pos += len(part) + len(separator)
        
        return segments

class ChunkingFactory:
    """Factory for creating chunking strategies"""
    
    @staticmethod
    def create_chunker(strategy: str, **kwargs) -> ChunkingStrategy:
        """Create a chunking strategy"""
        
        strategies = {
            'fixed_size': FixedSizeChunker,
            'sentence': SentenceChunker,
            'semantic': SemanticChunker,
            'hierarchical': HierarchicalChunker
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        
        return strategies[strategy](**kwargs)
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available chunking strategies"""
        return ['fixed_size', 'sentence', 'semantic', 'hierarchical'] 