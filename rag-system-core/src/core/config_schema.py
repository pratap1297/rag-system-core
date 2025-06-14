"""
Configuration Schema Definitions for RAG System
Defines all configuration dataclasses and structures
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# =============================================================================
# AZURE AI SERVICES CONFIGURATION
# =============================================================================

@dataclass
class AzureFoundryConfig:
    """Azure AI Foundry configuration"""
    enabled: bool = True
    workspace_name: str = ""
    resource_group: str = "default"
    subscription_id: str = ""

@dataclass
class AzureVisionConfig:
    """Azure Computer Vision configuration"""
    enabled: bool = True
    api_version: str = "2024-02-01"
    read_timeout: int = 30
    max_retries: int = 3

@dataclass
class AzureConfig:
    """Azure AI Services configuration"""
    # API Keys and Endpoints
    chat_api_key: str = ""
    chat_endpoint: str = ""
    embeddings_endpoint: str = ""
    embeddings_key: str = ""
    computer_vision_endpoint: str = ""
    computer_vision_key: str = ""
    
    # Model Names
    chat_model: str = ""
    embedding_model: str = ""
    
    # Sub-configurations
    foundry: AzureFoundryConfig = field(default_factory=AzureFoundryConfig)
    vision: AzureVisionConfig = field(default_factory=AzureVisionConfig)

# =============================================================================
# LLM PROVIDER CONFIGURATIONS
# =============================================================================

@dataclass
class GroqConfig:
    """Groq LLM configuration"""
    api_key: str = ""
    model_name: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30

@dataclass
class AzureLLMConfig:
    """Azure LLM configuration"""
    api_key: str = ""
    endpoint: str = ""
    model_name: str = ""
    api_version: str = "2024-02-15-preview"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30

@dataclass
class LLMConfig:
    """LLM configuration with multiple providers"""
    provider: str = "groq"  # groq, azure, openai, cohere
    
    # Provider-specific configurations
    groq: GroqConfig = field(default_factory=GroqConfig)
    azure_llm: AzureLLMConfig = field(default_factory=AzureLLMConfig)

# =============================================================================
# EMBEDDING CONFIGURATIONS
# =============================================================================

@dataclass
class SentenceTransformersConfig:
    """Sentence Transformers configuration"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    device: str = "cpu"
    batch_size: int = 32

@dataclass
class AzureEmbeddingConfig:
    """Azure embedding configuration"""
    api_key: str = ""
    endpoint: str = ""
    model_name: str = ""
    dimension: int = 1024
    api_version: str = "2024-02-15-preview"
    batch_size: int = 64

@dataclass
class EmbeddingConfig:
    """Embedding configuration with multiple providers"""
    provider: str = "sentence-transformers"  # sentence-transformers, azure
    
    # Provider-specific configurations
    sentence_transformers: SentenceTransformersConfig = field(default_factory=SentenceTransformersConfig)
    azure: AzureEmbeddingConfig = field(default_factory=AzureEmbeddingConfig)

# =============================================================================
# SERVICENOW CONFIGURATION
# =============================================================================

@dataclass
class ServiceNowSyncConfig:
    """ServiceNow sync configuration"""
    enabled: bool = True
    interval_minutes: int = 120
    max_records: int = 1000
    fields: str = ""
    query_filter: str = "state!=7"
    date_field: str = "sys_updated_on"

@dataclass
class ServiceNowConfig:
    """ServiceNow integration configuration"""
    enabled: bool = True
    instance: str = ""
    username: str = ""
    password: str = ""
    table: str = "incident"
    
    # Sub-configurations
    sync: ServiceNowSyncConfig = field(default_factory=ServiceNowSyncConfig)

# =============================================================================
# API CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: str = ""

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    log_format: str = "json"
    log_dir: str = "logs"
    environment: str = "development"
    console_output: bool = True
    file_output: bool = True
    structured_logging: bool = True
    log_rotation: Optional[Dict[str, Any]] = None
    error_log: Optional[Dict[str, Any]] = None

# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_retention_hours: int = 24
    health_check_interval_seconds: int = 60
    performance_monitoring: bool = True
    error_tracking: bool = True
    circuit_breaker: Optional[Dict[str, Any]] = None

# =============================================================================
# DOCUMENT PROCESSING CONFIGURATION CLASSES
# =============================================================================

@dataclass
class TextExtractionConfig:
    """Text extraction configuration"""
    pdf_strategy: str = "pypdf2"  # pypdf2, pdfplumber, pymupdf
    docx_strategy: str = "python-docx"
    fallback_enabled: bool = True
    max_file_size_mb: int = 50
    timeout_seconds: int = 30

@dataclass
class MetadataExtractionConfig:
    """Metadata extraction configuration"""
    extract_creation_date: bool = True
    extract_modification_date: bool = True
    extract_author: bool = True
    extract_title: bool = True
    extract_subject: bool = True
    extract_keywords: bool = True
    custom_fields: List[str] = field(default_factory=list)

@dataclass
class ChunkingConfig:
    """Basic chunking configuration"""
    default_strategy: str = "fixed_size"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    enable_semantic_chunking: bool = False

# =============================================================================
# MAIN DOCUMENT PROCESSING CONFIGURATION
# =============================================================================

# =============================================================================
# DOCUMENT PROCESSING CONFIGURATION CLASSES
# =============================================================================

@dataclass
class TextExtractionConfig:
    """Text extraction configuration"""
    pdf_strategy: str = "pypdf2"  # pypdf2, pdfplumber, pymupdf
    docx_strategy: str = "python-docx"
    fallback_enabled: bool = True
    max_file_size_mb: int = 50
    timeout_seconds: int = 30

@dataclass
class MetadataExtractionConfig:
    """Metadata extraction configuration"""
    extract_creation_date: bool = True
    extract_modification_date: bool = True
    extract_author: bool = True
    extract_title: bool = True
    extract_subject: bool = True
    extract_keywords: bool = True
    custom_fields: List[str] = field(default_factory=list)

@dataclass
class ChunkingConfig:
    """Basic chunking configuration"""
    default_strategy: str = "fixed_size"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    enable_semantic_chunking: bool = False

@dataclass
class DocumentProcessingConfig:
    """Document processing configuration"""
    text_extraction: TextExtractionConfig = field(default_factory=TextExtractionConfig)
    metadata_extraction: MetadataExtractionConfig = field(default_factory=MetadataExtractionConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    text_chunking: TextChunkingConfig = field(default_factory=TextChunkingConfig)
    vector_embedding: VectorEmbeddingConfig = field(default_factory=VectorEmbeddingConfig)
    vector_storage: VectorStorageConfig = field(default_factory=VectorStorageConfig)
    folder_scanner: FolderScannerConfig = field(default_factory=FolderScannerConfig)

# =============================================================================
# PHASE 3.1: TEXT CHUNKING CONFIGURATION
# =============================================================================

@dataclass
class BoundaryDetectionConfig:
    """Boundary detection configuration"""
    section_headers: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'confidence_weight': 0.9,
        'patterns': [
            r'^#{1,6}\s+.+$',
            r'^\d+\.\s+[A-Z][^.]*$',
            r'^[A-Z][A-Z\s]+:?\s*$',
            r'^Chapter\s+\d+',
            r'^Section\s+\d+',
            r'^Part\s+[IVX]+'
        ]
    })
    paragraphs: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'confidence_weight': 0.7,
        'patterns': [
            r'\n\s*\n',
            r'\n\s*[-•*]\s+',
            r'\n\s*\d+\.\s+'
        ]
    })
    sentences: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'confidence_weight': 0.5,
        'patterns': [
            r'[.!?]+\s+[A-Z]',
            r'[.!?]+\s*\n'
        ]
    })
    structures: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'confidence_weight': 0.8,
        'table_patterns': [
            r'\n\s*\|.*\|\s*\n',
            r'\n\s*\+[-+]+\+\s*\n'
        ]
    })

@dataclass
class DocumentTypeRuleConfig:
    """Document type-specific chunking rules"""
    prioritize_boundaries: List[str] = field(default_factory=list)
    ignore_boundaries: List[str] = field(default_factory=list)
    boost_structure_score: float = 1.0
    boost_completeness_score: float = 1.0
    boost_coherence_score: float = 1.0

@dataclass
class TextChunkingCacheConfig:
    """Text chunking cache configuration"""
    enabled: bool = True
    max_cache_size: int = 1000
    cache_ttl_hours: int = 24

@dataclass
class QualityWeightsConfig:
    """Quality assessment weights configuration"""
    completeness: float = 0.3
    coherence: float = 0.3
    structure: float = 0.2
    size: float = 0.2

@dataclass
class TextChunkingConfig:
    """Text chunking configuration"""
    # Strategy and size parameters
    strategy: str = "document_aware"
    target_size: int = 1000
    max_size: int = 1500
    min_size: int = 200
    overlap_size: int = 200
    
    # Token counting
    model_name: str = "gpt-3.5-turbo"
    
    # Quality assessment
    min_quality_score: float = 0.6
    optimal_size_range: List[int] = field(default_factory=lambda: [800, 1200])
    
    # Boundary detection
    boundary_detection: BoundaryDetectionConfig = field(default_factory=BoundaryDetectionConfig)
    
    # Document type rules
    document_type_rules: Dict[str, DocumentTypeRuleConfig] = field(default_factory=lambda: {
        'technical_manual': DocumentTypeRuleConfig(
            prioritize_boundaries=['section_header', 'paragraph'],
            boost_structure_score=1.2
        ),
        'procedural_document': DocumentTypeRuleConfig(
            prioritize_boundaries=['section_header', 'paragraph', 'list'],
            ignore_boundaries=['sentence'],
            boost_structure_score=1.3
        ),
        'safety_document': DocumentTypeRuleConfig(
            prioritize_boundaries=['section_header', 'paragraph'],
            boost_completeness_score=1.2
        ),
        'incident_report': DocumentTypeRuleConfig(
            prioritize_boundaries=['paragraph', 'sentence'],
            boost_coherence_score=1.1
        ),
        'maintenance_log': DocumentTypeRuleConfig(
            prioritize_boundaries=['paragraph', 'list'],
            boost_structure_score=1.1
        ),
        'general': DocumentTypeRuleConfig(
            prioritize_boundaries=['paragraph', 'sentence'],
            boost_coherence_score=1.0
        )
    })
    
    # Performance and caching
    caching: TextChunkingCacheConfig = field(default_factory=TextChunkingCacheConfig)
    quality_weights: QualityWeightsConfig = field(default_factory=QualityWeightsConfig)
    
    # Processing limits
    max_concurrent_chunks: int = 10
    processing_timeout_seconds: int = 60

# =============================================================================
# PHASE 3.2: VECTOR EMBEDDING CONFIGURATION
# =============================================================================

@dataclass
class VectorEmbeddingCacheConfig:
    """Vector embedding cache configuration"""
    max_size: int = 10000
    ttl_hours: int = 24
    enable_persistence: bool = True
    cache_file: str = "data/embedding_cache.json"
    auto_save_interval: int = 300
    enable_compression: bool = False

@dataclass
class VectorEmbeddingQualityConfig:
    """Vector embedding quality assessment configuration"""
    min_norm: float = 0.1
    max_norm: float = 2.0
    min_variance: float = 0.001
    expected_dimensions: int = 1024
    quality_threshold: float = 0.7
    norm_weight: float = 0.3
    variance_weight: float = 0.2
    dimension_weight: float = 0.3
    validity_weight: float = 0.2

@dataclass
class VectorEmbeddingPerformanceConfig:
    """Vector embedding performance configuration"""
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_processing_timeout: int = 30
    batch_retry_delay: float = 1.0
    enable_monitoring: bool = False

@dataclass
class VectorEmbeddingMonitoringConfig:
    """Vector embedding monitoring configuration"""
    enable_metrics: bool = True
    metrics_interval: int = 60
    alert_on_failures: bool = True
    failure_threshold: int = 5
    performance_threshold: int = 30

@dataclass
class CohereEmbedConfig:
    """Cohere Embed v3 provider configuration"""
    enabled: bool = True
    api_key: str = ""
    api_endpoint: str = "https://api.cohere.ai/v1/embed"
    model_name: str = "embed-english-v3.0"
    max_batch_size: int = 96
    timeout: int = 30
    max_retries: int = 3
    requests_per_minute: int = 100

@dataclass
class OpenAIEmbedConfig:
    """OpenAI embedding provider configuration"""
    enabled: bool = False
    api_key: str = ""
    model_name: str = "text-embedding-ada-002"
    max_batch_size: int = 2048
    timeout: int = 30
    max_retries: int = 3

@dataclass
class SentenceTransformersEmbedConfig:
    """Sentence Transformers provider configuration"""
    enabled: bool = False
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32

@dataclass
class VectorEmbeddingConfig:
    """Vector embedding configuration"""
    default_provider: str = "cohere_embed_v3"
    batch_size: int = 32
    max_concurrent_batches: int = 5
    enable_caching: bool = True
    
    # Provider configurations
    cohere: CohereEmbedConfig = field(default_factory=CohereEmbedConfig)
    openai: OpenAIEmbedConfig = field(default_factory=OpenAIEmbedConfig)
    sentence_transformers: SentenceTransformersEmbedConfig = field(default_factory=SentenceTransformersEmbedConfig)
    
    # Component configurations
    cache: VectorEmbeddingCacheConfig = field(default_factory=VectorEmbeddingCacheConfig)
    quality: VectorEmbeddingQualityConfig = field(default_factory=VectorEmbeddingQualityConfig)
    performance: VectorEmbeddingPerformanceConfig = field(default_factory=VectorEmbeddingPerformanceConfig)
    monitoring: VectorEmbeddingMonitoringConfig = field(default_factory=VectorEmbeddingMonitoringConfig)

# =============================================================================
# PHASE 4.1: VECTOR STORAGE CONFIGURATION
# =============================================================================

@dataclass
class VectorStorageBackupConfig:
    """Vector storage backup configuration"""
    enabled: bool = True
    interval_hours: int = 6
    retention_days: int = 30
    backup_path: str = "backups/vectors"

@dataclass
class VectorStorageMonitoringConfig:
    """Vector storage monitoring configuration"""
    enabled: bool = True
    metrics_collection: bool = True
    performance_alerts: bool = True
    memory_threshold_mb: int = 2048

@dataclass
class VectorStorageCacheConfig:
    """Vector storage cache configuration"""
    enabled: bool = True
    max_size: int = 1000
    ttl_hours: int = 24

@dataclass
class VectorStorageConfig:
    """Vector storage configuration for Phase 4.1"""
    storage_path: str = "data/vectors"
    
    # Chunk embedding index configuration
    chunk_index_type: str = "ivf_flat"
    chunk_dimension: int = 1024
    chunk_nlist: int = 100
    chunk_nprobe: int = 10
    
    # Site embedding index configuration
    site_index_type: str = "flat"
    site_dimension: int = 384
    site_nlist: int = 50
    site_nprobe: int = 5
    
    # Performance settings
    max_workers: int = 4
    search_timeout_seconds: int = 30
    max_search_results: int = 100
    
    # Cache configuration
    cache: VectorStorageCacheConfig = field(default_factory=VectorStorageCacheConfig)
    
    # Production-specific settings
    backup: Optional[VectorStorageBackupConfig] = None
    monitoring: Optional[VectorStorageMonitoringConfig] = None

# =============================================================================
# PHASE 5.1: FOLDER SCANNER CONFIGURATION
# =============================================================================

@dataclass
class FolderScannerPathRule:
    """Path-based metadata extraction rule"""
    pattern: str
    field: str  # site_id, category, department
    description: Optional[str] = None


@dataclass
class FolderScannerConfig:
    """Folder scanner configuration for Phase 5.1"""
    # Monitoring configuration
    monitored_directories: List[str] = field(default_factory=list)
    scan_interval: int = 60  # seconds
    max_depth: int = 10
    enable_content_hashing: bool = True
    
    # File filtering
    supported_extensions: List[str] = field(default_factory=lambda: [
        '.pdf', '.txt', '.docx', '.doc', '.md', '.json', '.csv', '.xlsx', '.pptx'
    ])
    max_file_size_mb: int = 100
    min_file_size_bytes: int = 1
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '.*', '__pycache__', '*.tmp', '*.log', '*.bak'
    ])
    
    # Processing configuration
    max_concurrent_files: int = 5
    retry_attempts: int = 3
    retry_delay: int = 60  # seconds
    processing_timeout: int = 300  # seconds
    
    # Metadata extraction
    path_metadata_rules: Dict[str, Dict[str, str]] = field(default_factory=dict)
    auto_categorization: bool = True
    
    # Performance settings
    enable_parallel_scanning: bool = True
    scan_batch_size: int = 100
    memory_limit_mb: int = 500
    
    # Integration settings
    auto_start: bool = True
    enable_monitoring: bool = True


# =============================================================================
# MAIN SYSTEM CONFIGURATION
# =============================================================================

@dataclass
class SystemConfig:
    """Main system configuration"""
    environment: str = "development"
    debug: bool = False
    data_dir: str = "data"
    log_dir: str = "logs"
    
    # Component configurations
    azure: AzureConfig = field(default_factory=AzureConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    servicenow: ServiceNowConfig = field(default_factory=ServiceNowConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Document processing
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig) 