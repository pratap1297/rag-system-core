"""
Enhanced Configuration Manager for RAG System
Supports YAML configuration files and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ConfigManager:
    """Enhanced configuration manager with YAML and .env support"""
    
    def __init__(self, environment: str = None, config_dir: str = None):
        self.environment = environment or os.getenv('RAG_ENVIRONMENT', 'development')
        self.config_dir = Path(config_dir or Path(__file__).parent)
        self.config = self._load_config()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML files and environment variables"""
        config = {}
        
        # Try to load environment-specific YAML config
        env_config_path = self.config_dir / "environments" / f"{self.environment}.yaml"
        
        if env_config_path.exists():
            logger.info(f"Loading YAML config: {env_config_path}")
            with open(env_config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                config.update(self._expand_env_vars(yaml_config))
        
        # Apply direct environment variable overrides
        self._apply_env_overrides(config)
        
        return config
    
    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in YAML config"""
        if isinstance(data, dict):
            return {key: self._expand_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            var_expr = data[2:-1]  # Remove ${ and }
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, "")
        else:
            return data
    
    def _apply_env_overrides(self, config: Dict[str, Any]):
        """Apply environment variable overrides"""
        # System settings
        config['environment'] = os.getenv('RAG_ENVIRONMENT', config.get('environment', 'development'))
        config['debug'] = os.getenv('RAG_DEBUG', str(config.get('debug', False))).lower() == 'true'
        
        # Azure configuration
        azure_config = config.setdefault('azure', {})
        azure_config['chat_api_key'] = os.getenv('AZURE_CHATAPI_KEY', azure_config.get('chat_api_key', ''))
        azure_config['chat_endpoint'] = os.getenv('AZURE_CHAT_ENDPOINT', azure_config.get('chat_endpoint', ''))
        azure_config['embeddings_key'] = os.getenv('AZURE_EMBEDDINGS_KEY', azure_config.get('embeddings_key', ''))
        azure_config['embeddings_endpoint'] = os.getenv('AZURE_EMBEDDINGS_ENDPOINT', azure_config.get('embeddings_endpoint', ''))
        azure_config['computer_vision_key'] = os.getenv('AZURE_COMPUTER_VISION_KEY', azure_config.get('computer_vision_key', ''))
        azure_config['computer_vision_endpoint'] = os.getenv('AZURE_COMPUTER_VISION_ENDPOINT', azure_config.get('computer_vision_endpoint', ''))
        azure_config['chat_model'] = os.getenv('CHAT_MODEL', azure_config.get('chat_model', ''))
        azure_config['embedding_model'] = os.getenv('EMBEDDING_MODEL', azure_config.get('embedding_model', ''))
        
        # LLM configuration
        llm_config = config.setdefault('llm', {})
        llm_config['provider'] = os.getenv('RAG_LLM_PROVIDER', llm_config.get('provider', 'groq'))
        
        # Groq configuration
        groq_config = llm_config.setdefault('groq', {})
        groq_config['api_key'] = os.getenv('GROQ_API_KEY', groq_config.get('api_key', ''))
        
        # API configuration
        api_config = config.setdefault('api', {})
        api_config['host'] = os.getenv('RAG_API_HOST', api_config.get('host', '0.0.0.0'))
        api_config['port'] = int(os.getenv('RAG_API_PORT', api_config.get('port', 8000)))
        api_config['api_key'] = os.getenv('RAG_API_KEY', api_config.get('api_key', ''))
        
        # ServiceNow configuration
        servicenow_config = config.setdefault('servicenow', {})
        servicenow_config['enabled'] = os.getenv('SERVICENOW_SYNC_ENABLED', str(servicenow_config.get('enabled', True))).lower() == 'true'
        servicenow_config['instance'] = os.getenv('SERVICENOW_INSTANCE', servicenow_config.get('instance', ''))
        servicenow_config['username'] = os.getenv('SERVICENOW_USERNAME', servicenow_config.get('username', ''))
        servicenow_config['password'] = os.getenv('SERVICENOW_PASSWORD', servicenow_config.get('password', ''))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary"""
        return self.config
    
    def get_azure_config(self) -> Dict[str, Any]:
        """Get Azure configuration"""
        return self.config.get('azure', {})
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get('llm', {})
    
    def get_servicenow_config(self) -> Dict[str, Any]:
        """Get ServiceNow configuration"""
        return self.config.get('servicenow', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.config.get('api', {})
    
    def is_azure_enabled(self) -> bool:
        """Check if Azure AI services are properly configured"""
        azure = self.get_azure_config()
        return (bool(azure.get('chat_api_key')) and 
                bool(azure.get('embeddings_key')) and
                bool(azure.get('computer_vision_key')))
    
    def validate_config(self) -> Dict[str, str]:
        """Validate configuration and return issues"""
        issues = {}
        
        # Validate Azure configuration
        azure = self.get_azure_config()
        if not azure.get('chat_api_key'):
            issues['azure_chat_api_key'] = "Azure Chat API key not configured"
        if not azure.get('embeddings_key'):
            issues['azure_embeddings_key'] = "Azure Embeddings API key not configured"
        
        # Validate LLM configuration
        llm = self.get_llm_config()
        provider = llm.get('provider', 'groq')
        if provider == 'groq' and not llm.get('groq', {}).get('api_key'):
            issues['groq_api_key'] = "Groq API key not configured"
        
        # Validate ServiceNow configuration
        servicenow = self.get_servicenow_config()
        if servicenow.get('enabled', False):
            if not servicenow.get('instance'):
                issues['servicenow_instance'] = "ServiceNow instance not configured"
            if not servicenow.get('username'):
                issues['servicenow_username'] = "ServiceNow username not configured"
            if not servicenow.get('password'):
                issues['servicenow_password'] = "ServiceNow password not configured"
        
        return issues
    
    def save_summary(self, output_path: str = None) -> str:
        """Save configuration summary (without sensitive data)"""
        if output_path is None:
            output_path = f"config_summary_{self.environment}.yaml"
        
        azure = self.get_azure_config()
        llm = self.get_llm_config()
        servicenow = self.get_servicenow_config()
        api = self.get_api_config()
        
        summary = {
            "environment": self.config.get('environment'),
            "debug": self.config.get('debug'),
            "azure": {
                "chat_model": azure.get('chat_model'),
                "embedding_model": azure.get('embedding_model'),
                "has_chat_key": bool(azure.get('chat_api_key')),
                "has_embeddings_key": bool(azure.get('embeddings_key')),
                "has_vision_key": bool(azure.get('computer_vision_key')),
                "foundry_enabled": azure.get('foundry', {}).get('enabled', True)
            },
            "llm": {
                "provider": llm.get('provider'),
                "groq_configured": bool(llm.get('groq', {}).get('api_key'))
            },
            "servicenow": {
                "enabled": servicenow.get('enabled'),
                "instance": servicenow.get('instance'),
                "configured": bool(servicenow.get('username') and servicenow.get('password'))
            },
            "api": {
                "host": api.get('host'),
                "port": api.get('port'),
                "has_api_key": bool(api.get('api_key'))
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration summary saved to {output_path}")
        return output_path 