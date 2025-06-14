"""
Enhanced Configuration Management for RAG System
Supports YAML configuration files, environment-specific configs, and Azure AI services
"""

import os
import yaml
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

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
class OpenAIConfig:
    """OpenAI configuration"""
    api_key: str = ""
    model_name: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30

@dataclass
class CohereConfig:
    """Cohere configuration"""
    api_key: str = ""
    model_name: str = "command-r-plus"
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
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    cohere: CohereConfig = field(default_factory=CohereConfig)

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
class CohereEmbeddingConfig:
    """Cohere embedding configuration"""
    api_key: str = ""
    model_name: str = "embed-english-v3.0"
    dimension: int = 1024
    batch_size: int = 96

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
    provider: str = "sentence-transformers"  # sentence-transformers, cohere, azure
    
    # Provider-specific configurations
    sentence_transformers: SentenceTransformersConfig = field(default_factory=SentenceTransformersConfig)
    cohere: CohereEmbeddingConfig = field(default_factory=CohereEmbeddingConfig)
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
class ServiceNowConnectionConfig:
    """ServiceNow connection configuration"""
    timeout: int = 30
    max_retries: int = 3
    verify_ssl: bool = True
    connection_pool_size: int = 5

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
    connection: ServiceNowConnectionConfig = field(default_factory=ServiceNowConnectionConfig)

# =============================================================================
# OTHER CONFIGURATIONS (Enhanced from original)
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = False
    requests_per_minute: int = 100
    burst_limit: int = 20

@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key: str = ""
    
    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

@dataclass
class FAISSConfig:
    """FAISS-specific configuration"""
    index_type: str = "IndexFlatIP"
    nlist: int = 100
    nprobe: int = 10
    ef_search: int = 50

@dataclass
class DatabaseConfig:
    """Database configuration"""
    faiss_index_path: str = "data/vectors/index.faiss"
    metadata_path: str = "data/metadata"
    backup_path: str = "data/backups"
    max_backup_count: int = 5
    
    # FAISS-specific settings
    faiss: FAISSConfig = field(default_factory=FAISSConfig)

@dataclass
class OCRConfig:
    """OCR configuration"""
    enabled: bool = True
    provider: str = "azure"  # azure, tesseract
    confidence_threshold: float = 0.7
    parallel_processing: bool = False

@dataclass
class IngestionConfig:
    """Data ingestion configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 100
    batch_size: int = 10
    supported_formats: List[str] = field(default_factory=lambda: [".pdf", ".docx", ".txt", ".md"])
    
    # OCR configuration
    ocr: OCRConfig = field(default_factory=OCRConfig)

@dataclass
class SearchConfig:
    """Search strategy configuration"""
    strategy: str = "hybrid"  # vector, keyword, hybrid
    vector_weight: float = 0.7
    keyword_weight: float = 0.3

@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_top_k: int = 3
    enable_reranking: bool = True
    
    # Search configuration
    search: SearchConfig = field(default_factory=SearchConfig)

@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    enabled: bool = True
    interval_seconds: int = 30
    timeout_seconds: int = 10

@dataclass
class PerformanceConfig:
    """Performance monitoring configuration"""
    track_response_times: bool = True
    track_memory_usage: bool = True
    track_api_calls: bool = True
    enable_profiling: bool = False

@dataclass
class AlertConfig:
    """Alert configuration"""
    enabled: bool = False
    webhook_url: str = ""
    error_threshold: int = 10
    response_time_threshold: int = 5000

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Sub-configurations
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

@dataclass
class FolderMonitoringConfig:
    """Folder monitoring configuration"""
    enabled: bool = True
    check_interval_seconds: int = 60
    auto_ingest: bool = True
    recursive: bool = True
    max_file_size_mb: int = 100
    monitored_folders: List[str] = field(default_factory=list)
    supported_extensions: List[str] = field(default_factory=lambda: [".txt", ".md", ".pdf", ".docx", ".json", ".csv"])

@dataclass
class CORSConfig:
    """CORS configuration"""
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class SecurityConfig:
    """Security configuration"""
    api_key_required: bool = True
    api_key: str = ""
    
    # CORS settings
    cors: CORSConfig = field(default_factory=CORSConfig)
    
    # Rate limiting
    rate_limiting: RateLimitConfig = field(default_factory=RateLimitConfig)

@dataclass
class GradioUIConfig:
    """Gradio UI configuration"""
    port: int = 7860
    share: bool = False
    auth: Optional[str] = None
    theme: str = "default"
    enable_queue: bool = False
    max_threads: int = 4

@dataclass
class ServiceNowUIConfig:
    """ServiceNow UI configuration"""
    port: int = 7861
    share: bool = False
    auth: Optional[str] = None

@dataclass
class UIConfig:
    """UI configuration"""
    gradio: GradioUIConfig = field(default_factory=GradioUIConfig)
    servicenow: ServiceNowUIConfig = field(default_factory=ServiceNowUIConfig)

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
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    folder_monitoring: FolderMonitoringConfig = field(default_factory=FolderMonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ui: UIConfig = field(default_factory=UIConfig)

# =============================================================================
# ENHANCED CONFIGURATION MANAGER
# =============================================================================

class EnhancedConfigManager:
    """Enhanced configuration manager with YAML support and environment-specific configs"""
    
    def __init__(self, environment: str = None, config_dir: str = None):
        self.environment = environment or os.getenv('RAG_ENVIRONMENT', 'development')
        self.config_dir = Path(config_dir or "config")
        self.config = self._load_config()
        self._apply_env_overrides()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from YAML files"""
        # Try environment-specific config first
        env_config_path = self.config_dir / "environments" / f"{self.environment}.yaml"
        
        if env_config_path.exists():
            logger.info(f"Loading environment config: {env_config_path}")
            return self._load_yaml_config(env_config_path)
        
        # Fallback to default config
        default_config_path = self.config_dir / "default.yaml"
        if default_config_path.exists():
            logger.info(f"Loading default config: {default_config_path}")
            return self._load_yaml_config(default_config_path)
        
        # Create default configuration
        logger.warning("No configuration file found, using defaults")
        return SystemConfig()
    
    def _load_yaml_config(self, config_path: Path) -> SystemConfig:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Expand environment variables
            config_data = self._expand_env_vars(config_data)
            
            return self._dict_to_config(config_data)
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return SystemConfig()
    
    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in configuration"""
        if isinstance(data, dict):
            return {key: self._expand_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            # Extract variable name and default value
            var_expr = data[2:-1]  # Remove ${ and }
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, "")
        else:
            return data
    
    def _dict_to_config(self, data: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig with nested dataclasses"""
        try:
            # Create main config
            main_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
            config = SystemConfig(**main_data)
            
            # Handle nested configurations
            if 'azure' in data:
                config.azure = self._create_azure_config(data['azure'])
            if 'llm' in data:
                config.llm = self._create_llm_config(data['llm'])
            if 'embedding' in data:
                config.embedding = self._create_embedding_config(data['embedding'])
            if 'servicenow' in data:
                config.servicenow = self._create_servicenow_config(data['servicenow'])
            if 'api' in data:
                config.api = self._create_api_config(data['api'])
            if 'database' in data:
                config.database = self._create_database_config(data['database'])
            if 'ingestion' in data:
                config.ingestion = self._create_ingestion_config(data['ingestion'])
            if 'retrieval' in data:
                config.retrieval = self._create_retrieval_config(data['retrieval'])
            if 'monitoring' in data:
                config.monitoring = self._create_monitoring_config(data['monitoring'])
            if 'folder_monitoring' in data:
                config.folder_monitoring = FolderMonitoringConfig(**data['folder_monitoring'])
            if 'security' in data:
                config.security = self._create_security_config(data['security'])
            if 'ui' in data:
                config.ui = self._create_ui_config(data['ui'])
            
            return config
            
        except Exception as e:
            logger.error(f"Error converting dict to config: {e}")
            return SystemConfig()
    
    def _create_azure_config(self, data: Dict[str, Any]) -> AzureConfig:
        """Create Azure configuration"""
        azure_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = AzureConfig(**azure_data)
        
        if 'foundry' in data:
            config.foundry = AzureFoundryConfig(**data['foundry'])
        if 'vision' in data:
            config.vision = AzureVisionConfig(**data['vision'])
        
        return config
    
    def _create_llm_config(self, data: Dict[str, Any]) -> LLMConfig:
        """Create LLM configuration"""
        llm_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = LLMConfig(**llm_data)
        
        if 'groq' in data:
            config.groq = GroqConfig(**data['groq'])
        if 'azure_llm' in data:
            config.azure_llm = AzureLLMConfig(**data['azure_llm'])
        if 'openai' in data:
            config.openai = OpenAIConfig(**data['openai'])
        if 'cohere' in data:
            config.cohere = CohereConfig(**data['cohere'])
        
        return config
    
    def _create_embedding_config(self, data: Dict[str, Any]) -> EmbeddingConfig:
        """Create embedding configuration"""
        emb_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = EmbeddingConfig(**emb_data)
        
        if 'sentence_transformers' in data:
            config.sentence_transformers = SentenceTransformersConfig(**data['sentence_transformers'])
        if 'cohere' in data:
            config.cohere = CohereEmbeddingConfig(**data['cohere'])
        if 'azure' in data:
            config.azure = AzureEmbeddingConfig(**data['azure'])
        
        return config
    
    def _create_servicenow_config(self, data: Dict[str, Any]) -> ServiceNowConfig:
        """Create ServiceNow configuration"""
        sn_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = ServiceNowConfig(**sn_data)
        
        if 'sync' in data:
            config.sync = ServiceNowSyncConfig(**data['sync'])
        if 'connection' in data:
            config.connection = ServiceNowConnectionConfig(**data['connection'])
        
        return config
    
    def _create_api_config(self, data: Dict[str, Any]) -> APIConfig:
        """Create API configuration"""
        api_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = APIConfig(**api_data)
        
        if 'rate_limit' in data:
            config.rate_limit = RateLimitConfig(**data['rate_limit'])
        
        return config
    
    def _create_database_config(self, data: Dict[str, Any]) -> DatabaseConfig:
        """Create database configuration"""
        db_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = DatabaseConfig(**db_data)
        
        if 'faiss' in data:
            config.faiss = FAISSConfig(**data['faiss'])
        
        return config
    
    def _create_ingestion_config(self, data: Dict[str, Any]) -> IngestionConfig:
        """Create ingestion configuration"""
        ing_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = IngestionConfig(**ing_data)
        
        if 'ocr' in data:
            config.ocr = OCRConfig(**data['ocr'])
        
        return config
    
    def _create_retrieval_config(self, data: Dict[str, Any]) -> RetrievalConfig:
        """Create retrieval configuration"""
        ret_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = RetrievalConfig(**ret_data)
        
        if 'search' in data:
            config.search = SearchConfig(**data['search'])
        
        return config
    
    def _create_monitoring_config(self, data: Dict[str, Any]) -> MonitoringConfig:
        """Create monitoring configuration"""
        mon_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = MonitoringConfig(**mon_data)
        
        if 'health_check' in data:
            config.health_check = HealthCheckConfig(**data['health_check'])
        if 'performance' in data:
            config.performance = PerformanceConfig(**data['performance'])
        if 'alerts' in data:
            config.alerts = AlertConfig(**data['alerts'])
        
        return config
    
    def _create_security_config(self, data: Dict[str, Any]) -> SecurityConfig:
        """Create security configuration"""
        sec_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = SecurityConfig(**sec_data)
        
        if 'cors' in data:
            config.cors = CORSConfig(**data['cors'])
        if 'rate_limiting' in data:
            config.rate_limiting = RateLimitConfig(**data['rate_limiting'])
        
        return config
    
    def _create_ui_config(self, data: Dict[str, Any]) -> UIConfig:
        """Create UI configuration"""
        config = UIConfig()
        
        if 'gradio' in data:
            config.gradio = GradioUIConfig(**data['gradio'])
        if 'servicenow' in data:
            config.servicenow = ServiceNowUIConfig(**data['servicenow'])
        
        return config
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides (for backward compatibility)"""
        # System level
        if os.getenv('RAG_ENVIRONMENT'):
            self.config.environment = os.getenv('RAG_ENVIRONMENT')
        if os.getenv('RAG_DEBUG'):
            self.config.debug = os.getenv('RAG_DEBUG').lower() == 'true'
        
        # API configuration
        if os.getenv('RAG_API_HOST'):
            self.config.api.host = os.getenv('RAG_API_HOST')
        if os.getenv('RAG_API_PORT'):
            self.config.api.port = int(os.getenv('RAG_API_PORT'))
        if os.getenv('RAG_API_KEY'):
            self.config.api.api_key = os.getenv('RAG_API_KEY')
        
        # LLM configuration
        if os.getenv('RAG_LLM_PROVIDER'):
            self.config.llm.provider = os.getenv('RAG_LLM_PROVIDER')
        
        # Groq API key
        if os.getenv('GROQ_API_KEY'):
            self.config.llm.groq.api_key = os.getenv('GROQ_API_KEY')
        
        # Azure configuration
        if os.getenv('AZURE_CHATAPI_KEY'):
            self.config.azure.chat_api_key = os.getenv('AZURE_CHATAPI_KEY')
            self.config.llm.azure_llm.api_key = os.getenv('AZURE_CHATAPI_KEY')
        
        if os.getenv('AZURE_CHAT_ENDPOINT'):
            self.config.azure.chat_endpoint = os.getenv('AZURE_CHAT_ENDPOINT')
            self.config.llm.azure_llm.endpoint = os.getenv('AZURE_CHAT_ENDPOINT')
        
        if os.getenv('CHAT_MODEL'):
            self.config.azure.chat_model = os.getenv('CHAT_MODEL')
            self.config.llm.azure_llm.model_name = os.getenv('CHAT_MODEL')
        
        # Azure embeddings
        if os.getenv('AZURE_EMBEDDINGS_KEY'):
            self.config.azure.embeddings_key = os.getenv('AZURE_EMBEDDINGS_KEY')
            self.config.embedding.azure.api_key = os.getenv('AZURE_EMBEDDINGS_KEY')
        
        if os.getenv('AZURE_EMBEDDINGS_ENDPOINT'):
            self.config.azure.embeddings_endpoint = os.getenv('AZURE_EMBEDDINGS_ENDPOINT')
            self.config.embedding.azure.endpoint = os.getenv('AZURE_EMBEDDINGS_ENDPOINT')
        
        if os.getenv('EMBEDDING_MODEL'):
            self.config.azure.embedding_model = os.getenv('EMBEDDING_MODEL')
            self.config.embedding.azure.model_name = os.getenv('EMBEDDING_MODEL')
        
        # Azure Computer Vision
        if os.getenv('AZURE_COMPUTER_VISION_ENDPOINT'):
            self.config.azure.computer_vision_endpoint = os.getenv('AZURE_COMPUTER_VISION_ENDPOINT')
        
        if os.getenv('AZURE_COMPUTER_VISION_KEY'):
            self.config.azure.computer_vision_key = os.getenv('AZURE_COMPUTER_VISION_KEY')
        
        # ServiceNow configuration
        if os.getenv('SERVICENOW_INSTANCE'):
            self.config.servicenow.instance = os.getenv('SERVICENOW_INSTANCE')
        if os.getenv('SERVICENOW_USERNAME'):
            self.config.servicenow.username = os.getenv('SERVICENOW_USERNAME')
        if os.getenv('SERVICENOW_PASSWORD'):
            self.config.servicenow.password = os.getenv('SERVICENOW_PASSWORD')
        
        # Logging
        if os.getenv('RAG_LOG_LEVEL'):
            self.config.monitoring.log_level = os.getenv('RAG_LOG_LEVEL')
    
    def get_config(self, component: Optional[str] = None) -> Any:
        """Get configuration for specific component or entire config"""
        if component is None:
            return self.config
        
        return getattr(self.config, component, None)
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to YAML file"""
        if output_path is None:
            output_path = self.config_dir / "environments" / f"{self.environment}.yaml"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save as YAML
        config_dict = asdict(self.config)
        
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def update_config(self, component: str, updates: Dict[str, Any]):
        """Update specific component configuration"""
        if hasattr(self.config, component):
            component_config = getattr(self.config, component)
            for key, value in updates.items():
                if hasattr(component_config, key):
                    setattr(component_config, key, value)
            logger.info(f"Updated {component} configuration")
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return any issues"""
        issues = {}
        
        # Validate Azure configuration
        if self.config.llm.provider == "azure" and not self.config.azure.chat_api_key:
            issues['azure_chat_api_key'] = "Azure Chat API key not configured"
        
        if self.config.embedding.provider == "azure" and not self.config.azure.embeddings_key:
            issues['azure_embeddings_key'] = "Azure Embeddings API key not configured"
        
        # Validate Groq configuration
        if self.config.llm.provider == "groq" and not self.config.llm.groq.api_key:
            issues['groq_api_key'] = "Groq API key not configured"
        
        # Validate ServiceNow configuration
        if self.config.servicenow.enabled:
            if not self.config.servicenow.instance:
                issues['servicenow_instance'] = "ServiceNow instance not configured"
            if not self.config.servicenow.username:
                issues['servicenow_username'] = "ServiceNow username not configured"
            if not self.config.servicenow.password:
                issues['servicenow_password'] = "ServiceNow password not configured"
        
        # Validate paths
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            issues['data_dir'] = f"Data directory does not exist: {data_dir}"
        
        return issues
    
    def get_provider_config(self, provider_type: str, provider_name: str) -> Optional[Any]:
        """Get configuration for a specific provider"""
        if provider_type == "llm":
            return getattr(self.config.llm, provider_name, None)
        elif provider_type == "embedding":
            return getattr(self.config.embedding, provider_name, None)
        else:
            return None
    
    def is_azure_enabled(self) -> bool:
        """Check if Azure AI services are enabled"""
        return (self.config.azure.foundry.enabled and 
                bool(self.config.azure.chat_api_key) and 
                bool(self.config.azure.embeddings_key))
    
    def get_active_llm_config(self) -> Any:
        """Get the configuration for the currently active LLM provider"""
        provider = self.config.llm.provider
        return self.get_provider_config("llm", provider)
    
    def get_active_embedding_config(self) -> Any:
        """Get the configuration for the currently active embedding provider"""
        provider = self.config.embedding.provider
        return self.get_provider_config("embedding", provider) 