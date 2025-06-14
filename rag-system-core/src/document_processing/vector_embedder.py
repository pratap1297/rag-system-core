"""
Vector Embedding Module for RAG System - Phase 3.2
High-performance vector embedding generation with quality assurance and optimization
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

from core.logging_system import get_logger
from core.exceptions import ProcessingError, ConfigurationError
from core.error_handler import with_error_handling
from core.monitoring import get_performance_monitor

# Import text chunker for integration
from .text_chunker import TextChunk


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    COHERE_EMBED_V3 = "cohere_embed_v3"
    OPENAI_ADA_002 = "openai_ada_002"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class InputType(Enum):
    """Cohere Embed v3 input types for optimization"""
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class EmbeddingQuality(Enum):
    """Embedding quality levels"""
    EXCELLENT = "excellent"  # > 0.9
    GOOD = "good"           # 0.8 - 0.9
    ACCEPTABLE = "acceptable"  # 0.7 - 0.8
    POOR = "poor"           # < 0.7


@dataclass
class EmbeddingMetrics:
    """Embedding quality and performance metrics"""
    vector_norm: float
    dimension_count: int
    generation_time: float
    provider_confidence: Optional[float] = None
    quality_score: float = 0.0
    quality_level: EmbeddingQuality = EmbeddingQuality.ACCEPTABLE
    
    # Quality indicators
    has_nan_values: bool = False
    has_inf_values: bool = False
    zero_variance: bool = False
    dimension_range: Tuple[float, float] = (0.0, 0.0)
    
    # Processing metadata
    batch_id: Optional[str] = None
    retry_count: int = 0
    cache_hit: bool = False


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text_id: str
    text_content: str
    embedding_vector: np.ndarray
    metrics: EmbeddingMetrics
    
    # Source information
    source_chunk: Optional[TextChunk] = None
    input_type: InputType = InputType.SEARCH_DOCUMENT
    provider: EmbeddingProvider = EmbeddingProvider.COHERE_EMBED_V3
    
    # Processing metadata
    timestamp: datetime = field(default_factory=datetime.now)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEmbeddingRequest:
    """Batch embedding request"""
    request_id: str
    texts: List[str]
    text_ids: List[str]
    input_type: InputType = InputType.SEARCH_DOCUMENT
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Source chunks for integration
    source_chunks: Optional[List[TextChunk]] = None
    
    def __post_init__(self):
        if len(self.texts) != len(self.text_ids):
            raise ValueError("texts and text_ids must have the same length")


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation"""
    request_id: str
    results: List[EmbeddingResult]
    batch_metrics: Dict[str, Any]
    
    # Processing information
    total_processing_time: float
    successful_embeddings: int
    failed_embeddings: int
    errors: List[str] = field(default_factory=list)


class EmbeddingCache:
    """High-performance embedding cache with TTL and LRU eviction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("embedding_cache")
        
        # Cache configuration
        self.max_size = config.get('max_size', 10000)
        self.ttl_hours = config.get('ttl_hours', 24)
        self.enable_persistence = config.get('enable_persistence', True)
        self.cache_file = Path(config.get('cache_file', 'data/embedding_cache.json'))
        
        # Cache storage
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Load existing cache
        if self.enable_persistence:
            self._load_cache()
        
        self.logger.info(f"Embedding cache initialized: max_size={self.max_size}, ttl={self.ttl_hours}h")
    
    def _generate_cache_key(self, text: str, input_type: InputType, provider: EmbeddingProvider) -> str:
        """Generate cache key for text and parameters"""
        content = f"{text}|{input_type.value}|{provider.value}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, input_type: InputType, provider: EmbeddingProvider) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        cache_key = self._generate_cache_key(text, input_type, provider)
        
        if cache_key not in self.cache:
            self.misses += 1
            return None
        
        # Check TTL
        entry = self.cache[cache_key]
        cached_time = datetime.fromisoformat(entry['timestamp'])
        if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
            del self.cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            self.misses += 1
            return None
        
        # Update access time
        self.access_times[cache_key] = datetime.now()
        self.hits += 1
        
        # Return embedding vector
        return np.array(entry['embedding'])
    
    def put(self, text: str, input_type: InputType, provider: EmbeddingProvider, 
            embedding: np.ndarray, metrics: EmbeddingMetrics):
        """Store embedding in cache"""
        cache_key = self._generate_cache_key(text, input_type, provider)
        
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        # Store embedding
        self.cache[cache_key] = {
            'embedding': embedding.tolist(),
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'vector_norm': metrics.vector_norm,
                'quality_score': metrics.quality_score,
                'generation_time': metrics.generation_time
            }
        }
        self.access_times[cache_key] = datetime.now()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.evictions += 1
    
    def _load_cache(self):
        """Load cache from persistent storage"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.cache = data.get('cache', {})
                    
                    # Rebuild access times (all entries get current time)
                    self.access_times = {key: datetime.now() for key in self.cache.keys()}
                    
                self.logger.info(f"Loaded {len(self.cache)} entries from cache")
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            self.cache = {}
            self.access_times = {}
    
    def save_cache(self):
        """Save cache to persistent storage"""
        if not self.enable_persistence:
            return
        
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'cache': self.cache,
                    'stats': self.get_stats()
                }, f, indent=2)
            self.logger.debug(f"Saved {len(self.cache)} entries to cache")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'ttl_hours': self.ttl_hours
        }
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class CohereEmbedProvider:
    """Cohere Embed v3 integration for high-quality embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("cohere_embed")
        
        # API configuration
        self.api_key = config.get('api_key', '')
        self.api_endpoint = config.get('api_endpoint', 'https://api.cohere.ai/v1/embed')
        self.model_name = config.get('model_name', 'embed-english-v3.0')
        self.max_batch_size = config.get('max_batch_size', 96)
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        
        # Rate limiting
        self.requests_per_minute = config.get('requests_per_minute', 100)
        self.request_times: List[datetime] = []
        
        if not self.api_key:
            raise ConfigurationError("Cohere API key is required")
        
        self.logger.info(f"Cohere Embed provider initialized: model={self.model_name}")
    
    async def generate_embeddings(self, texts: List[str], input_type: InputType = InputType.SEARCH_DOCUMENT) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts"""
        
        if len(texts) > self.max_batch_size:
            raise ValueError(f"Batch size {len(texts)} exceeds maximum {self.max_batch_size}")
        
        # Rate limiting check
        await self._check_rate_limit()
        
        # Prepare request
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'texts': texts,
            'model': self.model_name,
            'input_type': input_type.value,
            'embedding_types': ['float']
        }
        
        # Make API request with retries
        for attempt in range(self.max_retries):
            try:
                # Simulate API call (replace with actual HTTP request in production)
                await asyncio.sleep(0.1)  # Simulate network latency
                
                # Mock response for demonstration
                embeddings = []
                for text in texts:
                    # Generate mock 1024-dimensional embedding
                    embedding = np.random.normal(0, 1, 1024).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize
                    embeddings.append(embedding)
                
                self.logger.debug(f"Generated {len(embeddings)} embeddings")
                return embeddings
                
            except Exception as e:
                self.logger.warning(f"Embedding generation attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise ProcessingError(f"Failed to generate embeddings after {self.max_retries} attempts: {e}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = datetime.now()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(now)


class EmbeddingQualityAssessor:
    """Assess embedding quality and provide recommendations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("embedding_quality")
        
        # Quality thresholds
        self.min_norm = config.get('min_norm', 0.1)
        self.max_norm = config.get('max_norm', 2.0)
        self.min_variance = config.get('min_variance', 0.001)
        self.expected_dimensions = config.get('expected_dimensions', 1024)
    
    def assess_embedding_quality(self, embedding: np.ndarray, text: str, 
                               generation_time: float) -> EmbeddingMetrics:
        """Assess the quality of a generated embedding"""
        
        # Basic validation
        if embedding is None or len(embedding) == 0:
            raise ValueError("Embedding is None or empty")
        
        # Calculate metrics
        vector_norm = float(np.linalg.norm(embedding))
        dimension_count = len(embedding)
        
        # Quality checks
        has_nan_values = bool(np.isnan(embedding).any())
        has_inf_values = bool(np.isinf(embedding).any())
        zero_variance = bool(np.var(embedding) < self.min_variance)
        dimension_range = (float(np.min(embedding)), float(np.max(embedding)))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            embedding, vector_norm, has_nan_values, has_inf_values, zero_variance
        )
        
        # Determine quality level
        if quality_score >= 0.9:
            quality_level = EmbeddingQuality.EXCELLENT
        elif quality_score >= 0.8:
            quality_level = EmbeddingQuality.GOOD
        elif quality_score >= 0.7:
            quality_level = EmbeddingQuality.ACCEPTABLE
        else:
            quality_level = EmbeddingQuality.POOR
        
        return EmbeddingMetrics(
            vector_norm=vector_norm,
            dimension_count=dimension_count,
            generation_time=generation_time,
            quality_score=quality_score,
            quality_level=quality_level,
            has_nan_values=has_nan_values,
            has_inf_values=has_inf_values,
            zero_variance=zero_variance,
            dimension_range=dimension_range
        )
    
    def _calculate_quality_score(self, embedding: np.ndarray, vector_norm: float,
                               has_nan_values: bool, has_inf_values: bool, 
                               zero_variance: bool) -> float:
        """Calculate overall quality score"""
        
        score = 1.0
        
        # Penalize invalid values
        if has_nan_values:
            score -= 0.5
        if has_inf_values:
            score -= 0.5
        if zero_variance:
            score -= 0.3
        
        # Penalize abnormal norms
        if vector_norm < self.min_norm or vector_norm > self.max_norm:
            score -= 0.2
        
        # Penalize wrong dimensions
        if len(embedding) != self.expected_dimensions:
            score -= 0.3
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))


class VectorEmbedder:
    """Main vector embedding system with multi-provider support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("vector_embedder")
        self.monitor = get_performance_monitor()
        
        # Initialize components
        self.cache = EmbeddingCache(self.config.get('cache', {}))
        self.quality_assessor = EmbeddingQualityAssessor(self.config.get('quality', {}))
        
        # Initialize providers
        self.providers = {}
        self._initialize_providers()
        
        # Configuration
        self.default_provider = EmbeddingProvider(
            self.config.get('default_provider', 'cohere_embed_v3')
        )
        self.enable_caching = self.config.get('enable_caching', True)
        self.batch_size = self.config.get('batch_size', 32)
        self.max_concurrent_batches = self.config.get('max_concurrent_batches', 5)
        
        # Statistics
        self.total_embeddings_generated = 0
        self.total_processing_time = 0.0
        self.failed_embeddings = 0
        
        self.logger.info("Vector embedder initialized")
    
    def _initialize_providers(self):
        """Initialize embedding providers"""
        
        # Cohere Embed v3
        cohere_config = self.config.get('cohere', {})
        if cohere_config.get('enabled', True):
            try:
                self.providers[EmbeddingProvider.COHERE_EMBED_V3] = CohereEmbedProvider(cohere_config)
            except Exception as e:
                self.logger.warning(f"Failed to initialize Cohere provider: {e}")
        
        if not self.providers:
            raise ConfigurationError("No embedding providers available")
        
        self.logger.info(f"Initialized {len(self.providers)} embedding providers")
    
    @with_error_handling("vector_embedder", "embed_text")
    async def embed_text(self, text: str, input_type: InputType = InputType.SEARCH_DOCUMENT,
                        provider: Optional[EmbeddingProvider] = None) -> EmbeddingResult:
        """Generate embedding for a single text"""
        
        provider = provider or self.default_provider
        text_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        start_time = time.time()
        
        # Check cache first
        if self.enable_caching:
            cached_embedding = self.cache.get(text, input_type, provider)
            if cached_embedding is not None:
                generation_time = time.time() - start_time
                metrics = self.quality_assessor.assess_embedding_quality(
                    cached_embedding, text, generation_time
                )
                metrics.cache_hit = True
                
                return EmbeddingResult(
                    text_id=text_id,
                    text_content=text,
                    embedding_vector=cached_embedding,
                    metrics=metrics,
                    input_type=input_type,
                    provider=provider
                )
        
        # Generate embedding
        if provider not in self.providers:
            raise ProcessingError(f"Provider {provider.value} not available")
        
        try:
            embeddings = await self.providers[provider].generate_embeddings([text], input_type)
            embedding = embeddings[0]
            
            generation_time = time.time() - start_time
            
            # Assess quality
            metrics = self.quality_assessor.assess_embedding_quality(
                embedding, text, generation_time
            )
            
            # Cache result
            if self.enable_caching:
                self.cache.put(text, input_type, provider, embedding, metrics)
            
            # Update statistics
            self.total_embeddings_generated += 1
            self.total_processing_time += generation_time
            
            result = EmbeddingResult(
                text_id=text_id,
                text_content=text,
                embedding_vector=embedding,
                metrics=metrics,
                input_type=input_type,
                provider=provider
            )
            
            self.logger.debug(f"Generated embedding for text (quality: {metrics.quality_level.value})")
            return result
            
        except Exception as e:
            self.failed_embeddings += 1
            raise ProcessingError(f"Failed to generate embedding: {e}")
    
    @with_error_handling("vector_embedder", "embed_batch")
    async def embed_batch(self, request: BatchEmbeddingRequest) -> BatchEmbeddingResult:
        """Generate embeddings for a batch of texts"""
        
        start_time = time.time()
        results = []
        errors = []
        
        self.logger.info(f"Processing batch {request.request_id} with {len(request.texts)} texts")
        
        # Process in chunks if batch is too large
        chunk_size = min(self.batch_size, len(request.texts))
        
        for i in range(0, len(request.texts), chunk_size):
            chunk_texts = request.texts[i:i + chunk_size]
            chunk_ids = request.text_ids[i:i + chunk_size]
            chunk_chunks = request.source_chunks[i:i + chunk_size] if request.source_chunks else None
            
            try:
                chunk_results = await self._process_text_chunk(
                    chunk_texts, chunk_ids, chunk_chunks, request.input_type
                )
                results.extend(chunk_results)
                
            except Exception as e:
                error_msg = f"Failed to process chunk {i//chunk_size + 1}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        total_time = time.time() - start_time
        
        # Calculate batch metrics
        batch_metrics = self._calculate_batch_metrics(results, total_time)
        
        return BatchEmbeddingResult(
            request_id=request.request_id,
            results=results,
            batch_metrics=batch_metrics,
            total_processing_time=total_time,
            successful_embeddings=len(results),
            failed_embeddings=len(request.texts) - len(results),
            errors=errors
        )
    
    async def _process_text_chunk(self, texts: List[str], text_ids: List[str],
                                source_chunks: Optional[List[TextChunk]],
                                input_type: InputType) -> List[EmbeddingResult]:
        """Process a chunk of texts"""
        
        results = []
        
        # Check cache for all texts
        cached_results = []
        uncached_texts = []
        uncached_ids = []
        uncached_chunks = []
        
        for i, (text, text_id) in enumerate(zip(texts, text_ids)):
            if self.enable_caching:
                cached_embedding = self.cache.get(text, input_type, self.default_provider)
                if cached_embedding is not None:
                    metrics = self.quality_assessor.assess_embedding_quality(
                        cached_embedding, text, 0.0
                    )
                    metrics.cache_hit = True
                    
                    result = EmbeddingResult(
                        text_id=text_id,
                        text_content=text,
                        embedding_vector=cached_embedding,
                        metrics=metrics,
                        source_chunk=source_chunks[i] if source_chunks else None,
                        input_type=input_type,
                        provider=self.default_provider
                    )
                    cached_results.append(result)
                    continue
            
            uncached_texts.append(text)
            uncached_ids.append(text_id)
            if source_chunks:
                uncached_chunks.append(source_chunks[i])
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            start_time = time.time()
            
            provider = self.providers[self.default_provider]
            embeddings = await provider.generate_embeddings(uncached_texts, input_type)
            
            generation_time = time.time() - start_time
            
            for i, (text, text_id, embedding) in enumerate(zip(uncached_texts, uncached_ids, embeddings)):
                # Assess quality
                metrics = self.quality_assessor.assess_embedding_quality(
                    embedding, text, generation_time / len(uncached_texts)
                )
                
                # Cache result
                if self.enable_caching:
                    self.cache.put(text, input_type, self.default_provider, embedding, metrics)
                
                result = EmbeddingResult(
                    text_id=text_id,
                    text_content=text,
                    embedding_vector=embedding,
                    metrics=metrics,
                    source_chunk=uncached_chunks[i] if uncached_chunks else None,
                    input_type=input_type,
                    provider=self.default_provider
                )
                results.append(result)
        
        # Combine cached and new results
        all_results = cached_results + results
        
        # Sort by original order
        text_id_to_result = {r.text_id: r for r in all_results}
        ordered_results = [text_id_to_result[text_id] for text_id in text_ids if text_id in text_id_to_result]
        
        return ordered_results
    
    def _calculate_batch_metrics(self, results: List[EmbeddingResult], total_time: float) -> Dict[str, Any]:
        """Calculate metrics for a batch of results"""
        
        if not results:
            return {}
        
        # Quality distribution
        quality_counts = {}
        for result in results:
            quality = result.metrics.quality_level.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        # Performance metrics
        avg_generation_time = sum(r.metrics.generation_time for r in results) / len(results)
        cache_hits = sum(1 for r in results if r.metrics.cache_hit)
        cache_hit_rate = (cache_hits / len(results)) * 100
        
        # Vector statistics
        norms = [r.metrics.vector_norm for r in results]
        avg_norm = sum(norms) / len(norms)
        
        return {
            'total_embeddings': len(results),
            'quality_distribution': quality_counts,
            'avg_generation_time': avg_generation_time,
            'total_processing_time': total_time,
            'cache_hit_rate': cache_hit_rate,
            'avg_vector_norm': avg_norm,
            'throughput_per_second': len(results) / total_time if total_time > 0 else 0
        }
    
    async def embed_chunks(self, chunks: List[TextChunk], 
                          input_type: InputType = InputType.SEARCH_DOCUMENT) -> List[EmbeddingResult]:
        """Generate embeddings for text chunks from Phase 3.1"""
        
        # Create batch request
        texts = [chunk.text for chunk in chunks]
        text_ids = [chunk.chunk_id for chunk in chunks]
        
        request = BatchEmbeddingRequest(
            request_id=f"chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            texts=texts,
            text_ids=text_ids,
            input_type=input_type,
            source_chunks=chunks
        )
        
        # Process batch
        batch_result = await self.embed_batch(request)
        
        self.logger.info(
            f"Embedded {batch_result.successful_embeddings} chunks "
            f"(cache hit rate: {batch_result.batch_metrics.get('cache_hit_rate', 0):.1f}%)"
        )
        
        return batch_result.results
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        
        avg_processing_time = (
            self.total_processing_time / self.total_embeddings_generated
            if self.total_embeddings_generated > 0 else 0
        )
        
        return {
            'total_embeddings_generated': self.total_embeddings_generated,
            'failed_embeddings': self.failed_embeddings,
            'success_rate': (
                (self.total_embeddings_generated / 
                 (self.total_embeddings_generated + self.failed_embeddings)) * 100
                if (self.total_embeddings_generated + self.failed_embeddings) > 0 else 0
            ),
            'avg_processing_time': avg_processing_time,
            'total_processing_time': self.total_processing_time,
            'cache_stats': self.cache.get_stats(),
            'providers': list(self.providers.keys())
        }
    
    def save_cache(self):
        """Save embedding cache to persistent storage"""
        self.cache.save_cache()
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()


def create_vector_embedder(config: Dict[str, Any]) -> VectorEmbedder:
    """Factory function to create vector embedder"""
    return VectorEmbedder(config) 