"""
YAML Configuration Manager for RAG System
Supports environment-specific YAML configs and integrates with existing .env setup
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .config_schema import SystemConfig, AzureConfig, LLMConfig, EmbeddingConfig, ServiceNowConfig, APIConfig

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class YAMLConfigManager:
    """YAML-based configuration manager with environment variable support"""
    
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
        
        # Fallback to default configuration
        logger.warning("No YAML configuration file found, using defaults with .env overrides")
        return SystemConfig()
    
    def _load_yaml_config(self, config_path: Path) -> SystemConfig:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Expand environment variables in the YAML
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
        """Convert dictionary to SystemConfig"""
        try:
            # Extract main config data (non-dict values)
            main_data = {k: v for k, v in data.items() 
                        if not isinstance(v, dict) and k in ['environment', 'debug', 'data_dir', 'log_dir']}
            
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
            
            return config
            
        except Exception as e:
            logger.error(f"Error converting dict to config: {e}")
            return SystemConfig()
    
    def _create_azure_config(self, data: Dict[str, Any]) -> AzureConfig:
        """Create Azure configuration from dict"""
        from .config_schema import AzureFoundryConfig, AzureVisionConfig
        
        # Extract main Azure config
        azure_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = AzureConfig(**azure_data)
        
        # Handle nested configs
        if 'foundry' in data:
            config.foundry = AzureFoundryConfig(**data['foundry'])
        if 'vision' in data:
            config.vision = AzureVisionConfig(**data['vision'])
        
        return config
    
    def _create_llm_config(self, data: Dict[str, Any]) -> LLMConfig:
        """Create LLM configuration from dict"""
        from .config_schema import GroqConfig, AzureLLMConfig
        
        # Extract main LLM config
        llm_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = LLMConfig(**llm_data)
        
        # Handle provider configs
        if 'groq' in data:
            config.groq = GroqConfig(**data['groq'])
        if 'azure_llm' in data:
            config.azure_llm = AzureLLMConfig(**data['azure_llm'])
        
        return config
    
    def _create_embedding_config(self, data: Dict[str, Any]) -> EmbeddingConfig:
        """Create embedding configuration from dict"""
        from .config_schema import SentenceTransformersConfig, AzureEmbeddingConfig
        
        # Extract main embedding config
        emb_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = EmbeddingConfig(**emb_data)
        
        # Handle provider configs
        if 'sentence_transformers' in data:
            config.sentence_transformers = SentenceTransformersConfig(**data['sentence_transformers'])
        if 'azure' in data:
            config.azure = AzureEmbeddingConfig(**data['azure'])
        
        return config
    
    def _create_servicenow_config(self, data: Dict[str, Any]) -> ServiceNowConfig:
        """Create ServiceNow configuration from dict"""
        from .config_schema import ServiceNowSyncConfig
        
        # Extract main ServiceNow config
        sn_data = {k: v for k, v in data.items() if not isinstance(v, dict)}
        config = ServiceNowConfig(**sn_data)
        
        # Handle sync config
        if 'sync' in data:
            config.sync = ServiceNowSyncConfig(**data['sync'])
        
        return config
    
    def _create_api_config(self, data: Dict[str, Any]) -> APIConfig:
        """Create API configuration from dict"""
        return APIConfig(**data)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides for backward compatibility"""
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
        
        # Azure configuration from .env
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
        if os.getenv('SERVICENOW_SYNC_ENABLED'):
            self.config.servicenow.enabled = os.getenv('SERVICENOW_SYNC_ENABLED').lower() == 'true'
        if os.getenv('SERVICENOW_SYNC_INTERVAL'):
            self.config.servicenow.sync.interval_minutes = int(os.getenv('SERVICENOW_SYNC_INTERVAL'))
    
    def get_config(self, component: Optional[str] = None) -> Any:
        """Get configuration for specific component or entire config"""
        if component is None:
            return self.config
        
        return getattr(self.config, component, None)
    
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
    
    def is_azure_enabled(self) -> bool:
        """Check if Azure AI services are properly configured"""
        return (bool(self.config.azure.chat_api_key) and 
                bool(self.config.azure.embeddings_key) and
                bool(self.config.azure.computer_vision_key))
    
    def get_active_llm_config(self):
        """Get the configuration for the currently active LLM provider"""
        provider = self.config.llm.provider
        if provider == "groq":
            return self.config.llm.groq
        elif provider == "azure":
            return self.config.llm.azure_llm
        else:
            return None
    
    def get_active_embedding_config(self):
        """Get the configuration for the currently active embedding provider"""
        provider = self.config.embedding.provider
        if provider == "sentence-transformers":
            return self.config.embedding.sentence_transformers
        elif provider == "azure":
            return self.config.embedding.azure
        else:
            return None
    
    def save_config_summary(self, output_path: str = None):
        """Save a summary of current configuration (without sensitive data)"""
        if output_path is None:
            output_path = f"config_summary_{self.environment}.yaml"
        
        # Create sanitized config summary
        summary = {
            "environment": self.config.environment,
            "debug": self.config.debug,
            "data_dir": self.config.data_dir,
            "log_dir": self.config.log_dir,
            "azure": {
                "foundry_enabled": self.config.azure.foundry.enabled,
                "vision_enabled": self.config.azure.vision.enabled,
                "chat_model": self.config.azure.chat_model,
                "embedding_model": self.config.azure.embedding_model,
                "has_chat_key": bool(self.config.azure.chat_api_key),
                "has_embeddings_key": bool(self.config.azure.embeddings_key),
                "has_vision_key": bool(self.config.azure.computer_vision_key)
            },
            "llm": {
                "provider": self.config.llm.provider,
                "groq_configured": bool(self.config.llm.groq.api_key),
                "azure_configured": bool(self.config.llm.azure_llm.api_key)
            },
            "embedding": {
                "provider": self.config.embedding.provider,
                "azure_configured": bool(self.config.embedding.azure.api_key)
            },
            "servicenow": {
                "enabled": self.config.servicenow.enabled,
                "instance": self.config.servicenow.instance,
                "configured": bool(self.config.servicenow.username and self.config.servicenow.password)
            },
            "api": {
                "host": self.config.api.host,
                "port": self.config.api.port,
                "has_api_key": bool(self.config.api.api_key)
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration summary saved to {output_path}")
        return output_path 