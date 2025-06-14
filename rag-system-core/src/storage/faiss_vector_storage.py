"""
FAISS Vector Storage Module for RAG System - Phase 4.1
High-performance vector storage and similarity search with enterprise features
"""

import asyncio
import faiss
import hashlib
import json
import numpy as np
import pickle
import threading
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging

# Core imports with fallback
try:
    from ..core.logging_system import get_logger
    from ..core.exceptions import ProcessingError, ConfigurationError
    from ..core.error_handler import with_error_handling
    from ..core.monitoring import get_performance_monitor
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)
    
    class ProcessingError(Exception):
        pass
    
    class ConfigurationError(Exception):
        pass
    
    def with_error_handling(module, operation):
        def decorator(func):
            return func
        return decorator
    
    def get_performance_monitor():
        return None


class IndexType(Enum):
    """FAISS index types for different use cases"""
    FLAT = "flat"                    # Exact search, small datasets
    IVF_FLAT = "ivf_flat"           # Inverted file with flat quantizer
    IVF_PQ = "ivf_pq"               # Inverted file with product quantizer
    HNSW = "hnsw"                   # Hierarchical Navigable Small World
    IVF_HNSW = "ivf_hnsw"           # Combined IVF + HNSW


class VectorType(Enum):
    """Vector types for dual index management"""
    CHUNK_EMBEDDING = "chunk_embedding"      # 1024-dim document chunks
    SITE_EMBEDDING = "site_embedding"        # 384-dim site-level embeddings


class SearchMode(Enum):
    """Search operation modes"""
    EXACT = "exact"                 # Exact similarity search
    APPROXIMATE = "approximate"     # Fast approximate search
    HYBRID = "hybrid"               # Combination of exact and approximate


@dataclass
class IndexConfig:
    """Configuration for FAISS index"""
    index_type: IndexType = IndexType.IVF_FLAT
    dimension: int = 1024
    nlist: int = 100                # Number of clusters for IVF
    nprobe: int = 10               # Number of clusters to search
    m: int = 8                     # Number of subquantizers for PQ
    nbits: int = 8                 # Bits per subquantizer
    hnsw_m: int = 16               # HNSW connections per node
    efConstruction: int = 200      # HNSW construction parameter
    efSearch: int = 64             # HNSW search parameter
    metric: str = "IP"             # Inner Product (for cosine similarity)
    train_size: int = 10000        # Minimum vectors needed for training


@dataclass
class SearchResult:
    """Search result with metadata"""
    vector_id: str
    similarity_score: float
    metadata: Dict[str, Any]
    vector_data: Optional[np.ndarray] = None
    search_time: float = 0.0
    index_used: str = ""


@dataclass
class IndexStats:
    """Index statistics and health metrics"""
    total_vectors: int = 0
    index_size_mb: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    search_count: int = 0
    avg_search_time: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    is_trained: bool = False
    training_time: float = 0.0


@dataclass
class VectorMetadata:
    """Metadata for stored vectors"""
    vector_id: str
    vector_type: VectorType
    source_document: str = ""
    chunk_index: int = -1
    section_headers: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    document_type: str = ""
    quality_score: float = 0.0
    embedding_provider: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


class VectorCache:
    """LRU cache for search results with TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
        self.access_order: List[str] = []
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, query_vector: np.ndarray, k: int, filters: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        vector_hash = hashlib.sha256(query_vector.tobytes()).hexdigest()[:16]
        filter_hash = hashlib.sha256(json.dumps(filters, sort_keys=True).encode()).hexdigest()[:8]
        return f"{vector_hash}_{k}_{filter_hash}"
    
    def get(self, query_vector: np.ndarray, k: int, filters: Dict[str, Any]) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        key = self._generate_key(query_vector, k, filters)
        
        with self.lock:
            if key in self.cache:
                results, timestamp = self.cache[key]
                
                # Check TTL
                if datetime.now() - timestamp < self.ttl:
                    # Update access order
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    self.hits += 1
                    return results.copy()
                else:
                    # Expired
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            
            self.misses += 1
            return None
    
    def put(self, query_vector: np.ndarray, k: int, filters: Dict[str, Any], results: List[SearchResult]):
        """Cache search results"""
        key = self._generate_key(query_vector, k, filters)
        
        with self.lock:
            # Evict if necessary
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = (results.copy(), datetime.now())
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached results"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'ttl_hours': self.ttl.total_seconds() / 3600
        }


class FAISSVectorStorage:
    """Enterprise FAISS Vector Storage with dual index management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("faiss_vector_storage")
        self.monitor = get_performance_monitor()
        
        # Storage paths
        self.storage_path = Path(config.get('storage_path', 'data/vectors'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Dual index configuration
        self.chunk_config = IndexConfig(
            index_type=IndexType(config.get('chunk_index_type', 'ivf_flat')),
            dimension=config.get('chunk_dimension', 1024),
            nlist=config.get('chunk_nlist', 100),
            nprobe=config.get('chunk_nprobe', 10)
        )
        
        self.site_config = IndexConfig(
            index_type=IndexType(config.get('site_index_type', 'flat')),
            dimension=config.get('site_dimension', 384),
            nlist=config.get('site_nlist', 50),
            nprobe=config.get('site_nprobe', 5)
        )
        
        # Initialize indices
        self.chunk_index: Optional[faiss.Index] = None
        self.site_index: Optional[faiss.Index] = None
        
        # Metadata storage
        self.chunk_metadata: Dict[int, VectorMetadata] = {}
        self.site_metadata: Dict[int, VectorMetadata] = {}
        
        # ID mapping (FAISS index -> our vector ID)
        self.chunk_id_mapping: Dict[int, str] = {}
        self.site_id_mapping: Dict[int, str] = {}
        
        # Reverse mapping (our vector ID -> FAISS index)
        self.chunk_reverse_mapping: Dict[str, int] = {}
        self.site_reverse_mapping: Dict[str, int] = {}
        
        # Statistics
        self.chunk_stats = IndexStats()
        self.site_stats = IndexStats()
        
        # Caching
        cache_config = config.get('cache', {})
        self.enable_caching = cache_config.get('enabled', True)
        self.cache = VectorCache(
            max_size=cache_config.get('max_size', 1000),
            ttl_hours=cache_config.get('ttl_hours', 24)
        ) if self.enable_caching else None
        
        # Concurrency
        self.max_workers = config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Thread safety
        self.chunk_lock = threading.RLock()
        self.site_lock = threading.RLock()
        
        # Initialize indices
        self._initialize_indices()
        
        self.logger.info("FAISS Vector Storage initialized")
    
    def _initialize_indices(self):
        """Initialize or load existing FAISS indices"""
        # Initialize chunk index
        chunk_index_path = self.storage_path / "chunk_index.faiss"
        
        if chunk_index_path.exists():
            try:
                self.chunk_index = faiss.read_index(str(chunk_index_path))
                self._load_metadata(VectorType.CHUNK_EMBEDDING)
                self.chunk_stats.total_vectors = self.chunk_index.ntotal
                self.chunk_stats.is_trained = self.chunk_index.is_trained
                self.logger.info(f"Loaded chunk index with {self.chunk_stats.total_vectors} vectors")
            except Exception as e:
                self.logger.warning(f"Failed to load chunk index: {e}")
                self._create_index(VectorType.CHUNK_EMBEDDING)
        else:
            self._create_index(VectorType.CHUNK_EMBEDDING)
        
        # Initialize site index
        site_index_path = self.storage_path / "site_index.faiss"
        
        if site_index_path.exists():
            try:
                self.site_index = faiss.read_index(str(site_index_path))
                self._load_metadata(VectorType.SITE_EMBEDDING)
                self.site_stats.total_vectors = self.site_index.ntotal
                self.site_stats.is_trained = self.site_index.is_trained
                self.logger.info(f"Loaded site index with {self.site_stats.total_vectors} vectors")
            except Exception as e:
                self.logger.warning(f"Failed to load site index: {e}")
                self._create_index(VectorType.SITE_EMBEDDING)
        else:
            self._create_index(VectorType.SITE_EMBEDDING)
    
    def _create_index(self, vector_type: VectorType):
        """Create a new FAISS index"""
        config = self.chunk_config if vector_type == VectorType.CHUNK_EMBEDDING else self.site_config
        
        if config.index_type == IndexType.FLAT:
            if config.metric == "IP":
                index = faiss.IndexFlatIP(config.dimension)
            else:
                index = faiss.IndexFlatL2(config.dimension)
        
        elif config.index_type == IndexType.IVF_FLAT:
            quantizer = faiss.IndexFlatIP(config.dimension) if config.metric == "IP" else faiss.IndexFlatL2(config.dimension)
            index = faiss.IndexIVFFlat(quantizer, config.dimension, config.nlist)
            index.nprobe = config.nprobe
        
        elif config.index_type == IndexType.IVF_PQ:
            quantizer = faiss.IndexFlatIP(config.dimension) if config.metric == "IP" else faiss.IndexFlatL2(config.dimension)
            index = faiss.IndexIVFPQ(quantizer, config.dimension, config.nlist, config.m, config.nbits)
            index.nprobe = config.nprobe
        
        elif config.index_type == IndexType.HNSW:
            index = faiss.IndexHNSWFlat(config.dimension, config.hnsw_m)
            index.hnsw.efConstruction = config.efConstruction
            index.hnsw.efSearch = config.efSearch
        
        else:
            # Default to flat index
            index = faiss.IndexFlatIP(config.dimension)
        
        if vector_type == VectorType.CHUNK_EMBEDDING:
            self.chunk_index = index
            self.chunk_metadata = {}
            self.chunk_id_mapping = {}
            self.chunk_reverse_mapping = {}
        else:
            self.site_index = index
            self.site_metadata = {}
            self.site_id_mapping = {}
            self.site_reverse_mapping = {}
        
        self.logger.info(f"Created new {config.index_type.value} index for {vector_type.value}")
    
    def _load_metadata(self, vector_type: VectorType):
        """Load vector metadata from disk"""
        metadata_path = self.storage_path / f"{vector_type.value}_metadata.pkl"
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                
                if vector_type == VectorType.CHUNK_EMBEDDING:
                    self.chunk_metadata = data.get('metadata', {})
                    self.chunk_id_mapping = data.get('id_mapping', {})
                    self.chunk_reverse_mapping = data.get('reverse_mapping', {})
                else:
                    self.site_metadata = data.get('metadata', {})
                    self.site_id_mapping = data.get('id_mapping', {})
                    self.site_reverse_mapping = data.get('reverse_mapping', {})
                
                self.logger.info(f"Loaded metadata for {vector_type.value}")
            except Exception as e:
                self.logger.warning(f"Failed to load metadata for {vector_type.value}: {e}")
    
    def _save_metadata(self, vector_type: VectorType):
        """Save vector metadata to disk"""
        metadata_path = self.storage_path / f"{vector_type.value}_metadata.pkl"
        
        try:
            if vector_type == VectorType.CHUNK_EMBEDDING:
                data = {
                    'metadata': self.chunk_metadata,
                    'id_mapping': self.chunk_id_mapping,
                    'reverse_mapping': self.chunk_reverse_mapping,
                    'saved_at': datetime.now().isoformat()
                }
            else:
                data = {
                    'metadata': self.site_metadata,
                    'id_mapping': self.site_id_mapping,
                    'reverse_mapping': self.site_reverse_mapping,
                    'saved_at': datetime.now().isoformat()
                }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(data, f)
            
        except Exception as e:
            raise ProcessingError(f"Failed to save metadata for {vector_type.value}: {e}")
    
    def _save_index(self, vector_type: VectorType):
        """Save FAISS index to disk"""
        index_path = self.storage_path / f"{vector_type.value.replace('_embedding', '')}_index.faiss"
        
        try:
            if vector_type == VectorType.CHUNK_EMBEDDING:
                faiss.write_index(self.chunk_index, str(index_path))
            else:
                faiss.write_index(self.site_index, str(index_path))
            
            self.logger.info(f"Saved {vector_type.value} index to {index_path}")
        except Exception as e:
            raise ProcessingError(f"Failed to save {vector_type.value} index: {e}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def _train_index_if_needed(self, vector_type: VectorType, vectors: np.ndarray):
        """Train index if it requires training and has enough data"""
        index = self.chunk_index if vector_type == VectorType.CHUNK_EMBEDDING else self.site_index
        config = self.chunk_config if vector_type == VectorType.CHUNK_EMBEDDING else self.site_config
        
        if not index.is_trained and vectors.shape[0] >= config.train_size:
            start_time = time.time()
            
            # Use a subset for training if we have too many vectors
            train_vectors = vectors[:config.train_size] if vectors.shape[0] > config.train_size else vectors
            
            self.logger.info(f"Training {vector_type.value} index with {train_vectors.shape[0]} vectors")
            index.train(train_vectors)
            
            training_time = time.time() - start_time
            
            if vector_type == VectorType.CHUNK_EMBEDDING:
                self.chunk_stats.is_trained = True
                self.chunk_stats.training_time = training_time
            else:
                self.site_stats.is_trained = True
                self.site_stats.training_time = training_time
            
            self.logger.info(f"Index training completed in {training_time:.2f}s")
    
    @with_error_handling("faiss_vector_storage", "add_vectors")
    async def add_vectors(self, vectors: List[np.ndarray], metadata_list: List[VectorMetadata]) -> List[str]:
        """Add vectors to the appropriate index"""
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        if not vectors:
            return []
        
        # Group by vector type
        chunk_vectors = []
        chunk_metadata = []
        site_vectors = []
        site_metadata = []
        
        for vector, metadata in zip(vectors, metadata_list):
            if metadata.vector_type == VectorType.CHUNK_EMBEDDING:
                chunk_vectors.append(vector)
                chunk_metadata.append(metadata)
            else:
                site_vectors.append(vector)
                site_metadata.append(metadata)
        
        vector_ids = []
        
        # Add chunk vectors
        if chunk_vectors:
            chunk_ids = await self._add_vectors_to_index(
                VectorType.CHUNK_EMBEDDING, 
                chunk_vectors, 
                chunk_metadata
            )
            vector_ids.extend(chunk_ids)
        
        # Add site vectors
        if site_vectors:
            site_ids = await self._add_vectors_to_index(
                VectorType.SITE_EMBEDDING, 
                site_vectors, 
                site_metadata
            )
            vector_ids.extend(site_ids)
        
        return vector_ids
    
    async def _add_vectors_to_index(self, vector_type: VectorType, vectors: List[np.ndarray], 
                                   metadata_list: List[VectorMetadata]) -> List[str]:
        """Add vectors to a specific index"""
        
        def _add_sync():
            lock = self.chunk_lock if vector_type == VectorType.CHUNK_EMBEDDING else self.site_lock
            index = self.chunk_index if vector_type == VectorType.CHUNK_EMBEDDING else self.site_index
            
            with lock:
                # Convert to numpy array
                vector_array = np.vstack(vectors).astype(np.float32)
                
                # Normalize for cosine similarity
                normalized_vectors = self._normalize_vectors(vector_array)
                
                # Train index if needed
                self._train_index_if_needed(vector_type, normalized_vectors)
                
                # Get current index size
                current_size = index.ntotal
                
                # Add vectors to index
                index.add(normalized_vectors)
                
                # Store metadata and mappings
                vector_ids = []
                for i, metadata in enumerate(metadata_list):
                    faiss_idx = current_size + i
                    vector_id = metadata.vector_id
                    
                    if vector_type == VectorType.CHUNK_EMBEDDING:
                        self.chunk_metadata[faiss_idx] = metadata
                        self.chunk_id_mapping[faiss_idx] = vector_id
                        self.chunk_reverse_mapping[vector_id] = faiss_idx
                        self.chunk_stats.total_vectors += 1
                    else:
                        self.site_metadata[faiss_idx] = metadata
                        self.site_id_mapping[faiss_idx] = vector_id
                        self.site_reverse_mapping[vector_id] = faiss_idx
                        self.site_stats.total_vectors += 1
                    
                    vector_ids.append(vector_id)
                
                # Save to disk
                self._save_metadata(vector_type)
                self._save_index(vector_type)
                
                return vector_ids
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _add_sync)
    
    @with_error_handling("faiss_vector_storage", "search_vectors")
    async def search_vectors(self, query_vector: np.ndarray, vector_type: VectorType, 
                           k: int = 5, search_mode: SearchMode = SearchMode.APPROXIMATE,
                           filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for similar vectors"""
        
        if filters is None:
            filters = {}
        
        # Check cache first
        if self.enable_caching and self.cache:
            cached_results = self.cache.get(query_vector, k, filters)
            if cached_results:
                return cached_results
        
        def _search_sync():
            start_time = time.time()
            
            lock = self.chunk_lock if vector_type == VectorType.CHUNK_EMBEDDING else self.site_lock
            index = self.chunk_index if vector_type == VectorType.CHUNK_EMBEDDING else self.site_index
            metadata_dict = self.chunk_metadata if vector_type == VectorType.CHUNK_EMBEDDING else self.site_metadata
            id_mapping = self.chunk_id_mapping if vector_type == VectorType.CHUNK_EMBEDDING else self.site_id_mapping
            
            with lock:
                if index.ntotal == 0:
                    return []
                
                # Normalize query vector
                query_array = query_vector.reshape(1, -1).astype(np.float32)
                normalized_query = self._normalize_vectors(query_array)
                
                # Adjust search parameters based on mode
                search_k = k
                if search_mode == SearchMode.EXACT:
                    search_k = min(k * 3, index.ntotal)
                elif search_mode == SearchMode.HYBRID:
                    search_k = min(k * 2, index.ntotal)
                
                # Perform search
                scores, indices = index.search(normalized_query, search_k)
                
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:  # FAISS returns -1 for empty slots
                        continue
                    
                    # Get metadata
                    if idx in metadata_dict:
                        metadata = metadata_dict[idx]
                        vector_id = id_mapping.get(idx, f"unknown_{idx}")
                        
                        # Skip deleted vectors
                        if metadata.custom_metadata.get('deleted', False):
                            continue
                        
                        # Apply filters
                        if filters and not self._matches_filters(metadata, filters):
                            continue
                        
                        # Create search result
                        result = SearchResult(
                            vector_id=vector_id,
                            similarity_score=float(score),
                            metadata=self._metadata_to_dict(metadata),
                            search_time=time.time() - start_time,
                            index_used=f"{vector_type.value}_{index.__class__.__name__}"
                        )
                        
                        results.append(result)
                        
                        if len(results) >= k:
                            break
                
                # Update statistics
                search_time = time.time() - start_time
                if vector_type == VectorType.CHUNK_EMBEDDING:
                    self.chunk_stats.search_count += 1
                    self.chunk_stats.avg_search_time = (
                        (self.chunk_stats.avg_search_time * (self.chunk_stats.search_count - 1) + search_time) /
                        self.chunk_stats.search_count
                    )
                else:
                    self.site_stats.search_count += 1
                    self.site_stats.avg_search_time = (
                        (self.site_stats.avg_search_time * (self.site_stats.search_count - 1) + search_time) /
                        self.site_stats.search_count
                    )
                
                return results
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(self.executor, _search_sync)
        
        # Cache results
        if self.enable_caching and self.cache:
            self.cache.put(query_vector, k, filters, results)
        
        return results
    
    def _matches_filters(self, metadata: VectorMetadata, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, value in filters.items():
            if hasattr(metadata, key):
                attr_value = getattr(metadata, key)
                
                if isinstance(value, list):
                    if attr_value not in value:
                        return False
                elif isinstance(value, dict):
                    # Range filters
                    if 'min' in value and attr_value < value['min']:
                        return False
                    if 'max' in value and attr_value > value['max']:
                        return False
                else:
                    if attr_value != value:
                        return False
            elif key in metadata.custom_metadata:
                if metadata.custom_metadata[key] != value:
                    return False
            else:
                return False
        
        return True
    
    def _metadata_to_dict(self, metadata: VectorMetadata) -> Dict[str, Any]:
        """Convert metadata object to dictionary"""
        return {
            'vector_id': metadata.vector_id,
            'vector_type': metadata.vector_type.value,
            'source_document': metadata.source_document,
            'chunk_index': metadata.chunk_index,
            'section_headers': metadata.section_headers,
            'page_numbers': metadata.page_numbers,
            'document_type': metadata.document_type,
            'quality_score': metadata.quality_score,
            'embedding_provider': metadata.embedding_provider,
            'created_at': metadata.created_at.isoformat(),
            'updated_at': metadata.updated_at.isoformat(),
            'tags': metadata.tags,
            'custom_metadata': metadata.custom_metadata
        }
    
    def get_index_stats(self, vector_type: VectorType) -> IndexStats:
        """Get statistics for a specific index"""
        if vector_type == VectorType.CHUNK_EMBEDDING:
            stats = self.chunk_stats
            if self.chunk_index:
                stats.total_vectors = self.chunk_index.ntotal
        else:
            stats = self.site_stats
            if self.site_index:
                stats.total_vectors = self.site_index.ntotal
        
        stats.last_updated = datetime.now()
        
        # Calculate memory usage (rough estimate)
        if vector_type == VectorType.CHUNK_EMBEDDING and self.chunk_index:
            stats.memory_usage_mb = (self.chunk_config.dimension * 4 * stats.total_vectors) / (1024 * 1024)
        elif vector_type == VectorType.SITE_EMBEDDING and self.site_index:
            stats.memory_usage_mb = (self.site_config.dimension * 4 * stats.total_vectors) / (1024 * 1024)
        
        # Add cache statistics
        if self.enable_caching and self.cache:
            cache_stats = self.cache.get_stats()
            stats.cache_hit_rate = cache_stats['hit_rate']
        
        return stats
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        chunk_stats = self.get_index_stats(VectorType.CHUNK_EMBEDDING)
        site_stats = self.get_index_stats(VectorType.SITE_EMBEDDING)
        
        total_vectors = chunk_stats.total_vectors + site_stats.total_vectors
        total_memory = chunk_stats.memory_usage_mb + site_stats.memory_usage_mb
        
        stats = {
            'total_vectors': total_vectors,
            'chunk_vectors': chunk_stats.total_vectors,
            'site_vectors': site_stats.total_vectors,
            'total_memory_mb': total_memory,
            'chunk_memory_mb': chunk_stats.memory_usage_mb,
            'site_memory_mb': site_stats.memory_usage_mb,
            'chunk_index_trained': chunk_stats.is_trained,
            'site_index_trained': site_stats.is_trained,
            'total_searches': chunk_stats.search_count + site_stats.search_count,
            'avg_search_time': (chunk_stats.avg_search_time + site_stats.avg_search_time) / 2,
            'cache_enabled': self.enable_caching,
            'cache_stats': self.cache.get_stats() if self.cache else None
        }
        
        return stats
    
    async def close(self):
        """Clean shutdown of the storage system"""
        # Save current state
        self._save_metadata(VectorType.CHUNK_EMBEDDING)
        self._save_metadata(VectorType.SITE_EMBEDDING)
        self._save_index(VectorType.CHUNK_EMBEDDING)
        self._save_index(VectorType.SITE_EMBEDDING)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("FAISS Vector Storage closed")


def create_vector_storage(config: Dict[str, Any]) -> FAISSVectorStorage:
    """Factory function to create vector storage instance"""
    return FAISSVectorStorage(config)
