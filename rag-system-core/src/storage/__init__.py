"""
Storage Module for RAG System
Provides vector storage and similarity search capabilities
"""

# Phase 4.1: FAISS Vector Storage Module
from .faiss_vector_storage import (
    FAISSVectorStorage,
    VectorType,
    IndexType,
    SearchMode,
    SearchResult,
    IndexStats,
    VectorMetadata,
    VectorCache,
    IndexConfig,
    create_vector_storage
)

# Legacy FAISS store (for backward compatibility)
try:
    from .faiss_store import FAISSStore
except ImportError:
    FAISSStore = None

# Metadata storage
try:
    from .metadata_store import MetadataStore
    from .persistent_metadata_store import PersistentMetadataStore
except ImportError:
    MetadataStore = None
    PersistentMetadataStore = None

__all__ = [
    # Phase 4.1: FAISS Vector Storage
    'FAISSVectorStorage',
    'VectorType',
    'IndexType', 
    'SearchMode',
    'SearchResult',
    'IndexStats',
    'VectorMetadata',
    'VectorCache',
    'IndexConfig',
    'create_vector_storage',
    
    # Legacy support
    'FAISSStore',
    'MetadataStore',
    'PersistentMetadataStore',
] 