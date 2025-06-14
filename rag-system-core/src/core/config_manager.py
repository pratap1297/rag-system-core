"""
Configuration Management for RAG System
Handles loading, validation, and management of system configuration
"""
import json
import os
import yaml
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import the updated configuration schema
# from .config_schema import SystemConfig

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Temporary minimal configuration for Phase 5.1
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

@dataclass
class DocumentProcessingConfig:
    """Document processing configuration"""
    folder_scanner: FolderScannerConfig = field(default_factory=FolderScannerConfig)

@dataclass
class SystemConfig:
    """Main system configuration"""
    environment: str = "development"
    debug: bool = False
    data_dir: str = "data"
    log_dir: str = "logs"
    
    # Document processing
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)

class ConfigManager:
    """Configuration manager with environment overrides"""
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "development"):
        self.environment = environment
        self.config_path = config_path or f"config/environments/{environment}.yaml"
        self.config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> SystemConfig:
        """Load configuration from YAML file or create default"""
        config_file = Path(self.config_path)
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                return self._dict_to_config(config_data)
            except Exception as e:
                print(f"Error loading config from {config_file}: {e}. Using defaults.")
        
        return SystemConfig()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig using the new schema"""
        try:
            # Extract document_processing section
            doc_processing_data = data.get('document_processing', {})
            folder_scanner_data = doc_processing_data.get('folder_scanner', {})
            
            # Create folder scanner config
            folder_scanner_config = FolderScannerConfig(**folder_scanner_data)
            
            # Create document processing config
            doc_processing_config = DocumentProcessingConfig(folder_scanner=folder_scanner_config)
            
            # Create system config
            system_config = SystemConfig(
                environment=data.get('environment', 'development'),
                debug=data.get('debug', False),
                data_dir=data.get('data_dir', 'data'),
                log_dir=data.get('log_dir', 'logs'),
                document_processing=doc_processing_config
            )
            
            return system_config
        except Exception as e:
            print(f"Error creating config from data: {e}. Using defaults.")
            return SystemConfig()
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # System level
        self.config.environment = os.getenv('RAG_ENVIRONMENT', self.config.environment)
        self.config.debug = os.getenv('RAG_DEBUG', str(self.config.debug)).lower() == 'true'
    
    def get_config(self, component: Optional[str] = None) -> Any:
        """Get configuration or specific component"""
        if component:
            return getattr(self.config, component, None)
        return self.config
    
    def save_config(self):
        """Save current configuration to file"""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False, indent=2)
    
    def update_config(self, component: str, updates: Dict[str, Any]):
        """Update specific component configuration"""
        if hasattr(self.config, component):
            component_config = getattr(self.config, component)
            for key, value in updates.items():
                if hasattr(component_config, key):
                    setattr(component_config, key, value)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate paths
        data_dir = Path(self.config.data_dir)
        if not data_dir.exists():
            validation_results['warnings'].append(f"Data directory does not exist: {data_dir}")
        
        log_dir = Path(self.config.log_dir)
        if not log_dir.exists():
            validation_results['warnings'].append(f"Log directory does not exist: {log_dir}")
        
        return validation_results 