"""
Semantic Chunker
Advanced chunking based on semantic similarity and document structure
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Semantic chunking will use fallback.")

try:
    from ..core.error_handling import ChunkingError
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.error_handling import ChunkingError

@dataclass
class ChunkBoundary:
    """Represents a potential chunk boundary with its score"""
    position: int
    score: float
    sentence_text: str
    boundary_type: str  # 'paragraph', 'sentence', 'semantic'

class SemanticChunker:
    """Advanced chunker that uses semantic similarity to determine optimal boundaries"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 similarity_threshold: float = 0.5,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.model = None
        self.enabled = SENTENCE_TRANSFORMERS_AVAILABLE
        
        if self.enabled:
            self._initialize_model()
        else:
            logging.warning("Semantic chunking disabled - using fallback chunking")
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logging.info(f"Semantic chunker initialized with model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize semantic chunker: {e}")
            self.enabled = False
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text using semantic similarity analysis
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        if not self.enabled:
            return self._fallback_chunking(text, metadata)
        
        try:
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            # Split into sentences
            sentences = self._split_into_sentences(cleaned_text)
            if len(sentences) <= 1:
                return self._create_single_chunk(cleaned_text, metadata)
            
            # Find semantic boundaries
            boundaries = self._find_semantic_boundaries(sentences)
            
            # Create chunks based on boundaries
            chunks = self._create_chunks_from_boundaries(sentences, boundaries, metadata)
            
            logging.info(f"Semantic chunking created {len(chunks)} chunks from {len(sentences)} sentences")
            return chunks
            
        except Exception as e:
            logging.error(f"Semantic chunking failed: {e}, falling back to simple chunking")
            return self._fallback_chunking(text, metadata)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved detection"""
        # Enhanced sentence splitting patterns
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Split by sentence endings
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _find_semantic_boundaries(self, sentences: List[str]) -> List[ChunkBoundary]:
        """Find optimal chunk boundaries based on semantic similarity"""
        if len(sentences) <= 2:
            return []
        
        try:
            # Generate embeddings for all sentences
            embeddings = self.model.encode(sentences)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(similarity)
            
            # Find boundaries where similarity drops significantly
            boundaries = []
            for i, similarity in enumerate(similarities):
                if similarity < self.similarity_threshold:
                    boundary = ChunkBoundary(
                        position=i + 1,  # Position after current sentence
                        score=1.0 - similarity,  # Higher score = better boundary
                        sentence_text=sentences[i + 1] if i + 1 < len(sentences) else "",
                        boundary_type='semantic'
                    )
                    boundaries.append(boundary)
            
            # Add paragraph boundaries (double newlines in original text)
            paragraph_boundaries = self._find_paragraph_boundaries(sentences)
            boundaries.extend(paragraph_boundaries)
            
            # Sort boundaries by position
            boundaries.sort(key=lambda x: x.position)
            
            return boundaries
            
        except Exception as e:
            logging.error(f"Failed to find semantic boundaries: {e}")
            return []
    
    def _find_paragraph_boundaries(self, sentences: List[str]) -> List[ChunkBoundary]:
        """Find paragraph boundaries in the text"""
        boundaries = []
        
        for i, sentence in enumerate(sentences):
            # Look for sentences that start with common paragraph indicators
            if (sentence.strip().startswith(('•', '-', '*', '1.', '2.', '3.')) or
                re.match(r'^\d+\.', sentence.strip()) or
                sentence.strip().startswith(('Chapter', 'Section', 'Part'))):
                
                boundary = ChunkBoundary(
                    position=i,
                    score=0.8,  # High score for structural boundaries
                    sentence_text=sentence,
                    boundary_type='paragraph'
                )
                boundaries.append(boundary)
        
        return boundaries
    
    def _create_chunks_from_boundaries(self, sentences: List[str], 
                                     boundaries: List[ChunkBoundary],
                                     metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Create chunks based on identified boundaries"""
        if not boundaries:
            # No good boundaries found, use size-based chunking
            return self._create_size_based_chunks(sentences, metadata)
        
        chunks = []
        current_chunk_start = 0
        
        for boundary in boundaries:
            # Check if chunk would be too large
            chunk_text = ' '.join(sentences[current_chunk_start:boundary.position])
            
            if len(chunk_text) >= self.chunk_size:
                # Create chunk up to this boundary
                chunk = self._create_chunk_object(
                    text=chunk_text,
                    sentences=sentences[current_chunk_start:boundary.position],
                    chunk_index=len(chunks),
                    boundary_info=boundary,
                    metadata=metadata
                )
                chunks.append(chunk)
                current_chunk_start = max(0, boundary.position - self._calculate_overlap_sentences(sentences, boundary.position))
        
        # Create final chunk if there are remaining sentences
        if current_chunk_start < len(sentences):
            final_text = ' '.join(sentences[current_chunk_start:])
            if final_text.strip():
                chunk = self._create_chunk_object(
                    text=final_text,
                    sentences=sentences[current_chunk_start:],
                    chunk_index=len(chunks),
                    boundary_info=None,
                    metadata=metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_size_based_chunks(self, sentences: List[str], 
                                metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fallback to size-based chunking when no good boundaries found"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk_object(
                    text=chunk_text,
                    sentences=current_chunk,
                    chunk_index=len(chunks),
                    boundary_info=None,
                    metadata=metadata
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = self._create_chunk_object(
                text=chunk_text,
                sentences=current_chunk,
                chunk_index=len(chunks),
                boundary_info=None,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _calculate_overlap_sentences(self, sentences: List[str], position: int) -> int:
        """Calculate how many sentences to include for overlap"""
        overlap_chars = 0
        overlap_sentences = 0
        
        for i in range(position - 1, -1, -1):
            sentence_length = len(sentences[i])
            if overlap_chars + sentence_length <= self.chunk_overlap:
                overlap_chars += sentence_length
                overlap_sentences += 1
            else:
                break
        
        return overlap_sentences
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap from the end of current chunk"""
        overlap_chars = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            if overlap_chars + len(sentence) <= self.chunk_overlap:
                overlap_chars += len(sentence)
                overlap_sentences.insert(0, sentence)
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk_object(self, text: str, sentences: List[str], 
                           chunk_index: int, boundary_info: Optional[ChunkBoundary],
                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a standardized chunk object"""
        chunk = {
            'text': text.strip(),
            'chunk_index': chunk_index,
            'chunk_size': len(text),
            'sentence_count': len(sentences),
            'chunking_method': 'semantic',
            'metadata': metadata or {}
        }
        
        if boundary_info:
            chunk['boundary_type'] = boundary_info.boundary_type
            chunk['boundary_score'] = boundary_info.score
        
        return chunk
    
    def _create_single_chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Create a single chunk when text is short"""
        return [{
            'text': text.strip(),
            'chunk_index': 0,
            'chunk_size': len(text),
            'sentence_count': 1,
            'chunking_method': 'single',
            'total_chunks': 1,
            'metadata': metadata or {}
        }]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        return text.strip()
    
    def _fallback_chunking(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fallback to simple character-based chunking"""
        from .chunker import Chunker
        
        fallback_chunker = Chunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        chunks = fallback_chunker.chunk_text(text, metadata)
        
        # Update chunking method
        for chunk in chunks:
            chunk['chunking_method'] = 'fallback'
        
        return chunks
    
    def get_chunker_info(self) -> Dict[str, Any]:
        """Get information about the chunker configuration"""
        return {
            'chunker_type': 'semantic',
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'similarity_threshold': self.similarity_threshold,
            'enabled': self.enabled,
            'model_available': SENTENCE_TRANSFORMERS_AVAILABLE
        }

def create_semantic_chunker(config_manager) -> SemanticChunker:
    """Factory function to create semantic chunker based on configuration"""
    config = config_manager.get_config()
    
    return SemanticChunker(
        chunk_size=config.ingestion.chunk_size,
        chunk_overlap=config.ingestion.chunk_overlap,
        similarity_threshold=0.5,  # Can be made configurable
        model_name="all-MiniLM-L6-v2"  # Lightweight model for chunking
    ) 