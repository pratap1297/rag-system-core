"""
Text Chunker
Split documents into chunks for processing
"""
import re
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..core.error_handling import ChunkingError

class Chunker:
    """Text chunking with overlap and metadata preservation"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 use_semantic: bool = False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize semantic chunker if requested
        self.semantic_chunker = None
        if use_semantic:
            try:
                from .semantic_chunker import SemanticChunker
                self.semantic_chunker = SemanticChunker(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                logging.info(f"Chunker initialized with semantic chunking: size={chunk_size}, overlap={chunk_overlap}")
            except Exception as e:
                logging.warning(f"Failed to initialize semantic chunker: {e}, using regular chunking")
                self.use_semantic = False
        else:
            logging.info(f"Chunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata"""
        if not text.strip():
            return []
        
        try:
            # Use semantic chunking if enabled and available
            if self.use_semantic and self.semantic_chunker:
                return self.semantic_chunker.chunk_text(text, metadata)
            
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk in enumerate(chunks):
                chunk_obj = {
                    'text': chunk,
                    'chunk_index': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks),
                    'chunking_method': 'recursive',
                    'metadata': metadata or {}
                }
                chunk_objects.append(chunk_obj)
            
            logging.info(f"Created {len(chunks)} chunks from text")
            return chunk_objects
            
        except Exception as e:
            raise ChunkingError(f"Failed to chunk text: {e}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        
        return text.strip()
    
    def chunk_by_sentences(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text by sentences while respecting size limits"""
        if not text.strip():
            return []
        
        try:
            # Split into sentences
            sentences = self._split_sentences(text)
            
            chunks = []
            current_chunk = ""
            chunk_index = 0
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_obj = {
                        'text': current_chunk.strip(),
                        'chunk_index': chunk_index,
                        'chunk_size': len(current_chunk),
                        'metadata': metadata or {}
                    }
                    chunks.append(chunk_obj)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + sentence
                    chunk_index += 1
                else:
                    current_chunk += sentence
            
            # Add final chunk
            if current_chunk.strip():
                chunk_obj = {
                    'text': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'chunk_size': len(current_chunk),
                    'metadata': metadata or {}
                }
                chunks.append(chunk_obj)
            
            # Update total chunks count
            for chunk in chunks:
                chunk['total_chunks'] = len(chunks)
            
            logging.info(f"Created {len(chunks)} sentence-based chunks")
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to chunk by sentences: {e}")
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - could be improved with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s + ' ' for s in sentences if s.strip()]
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to find a good breaking point (word boundary)
        overlap_text = text[-self.chunk_overlap:]
        space_index = overlap_text.find(' ')
        
        if space_index > 0:
            return overlap_text[space_index:]
        
        return overlap_text 