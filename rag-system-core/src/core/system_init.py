"""
System Initialization
Initialize and configure all RAG system components
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any

from .dependency_container import DependencyContainer, register_core_services
from .error_handling import ErrorHandler, get_error_handler
from .config_manager import ConfigManager

def setup_logging(config_manager: ConfigManager):
    """Setup system logging"""
    config = config_manager.get_config()
    
    # Create logs directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, config.monitoring.log_level.upper())
    
    if config.monitoring.log_format == "json":
        # JSON logging format
        import json
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': self.formatTime(record),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'extra'):
                    log_entry.update(record.extra)
                
                return json.dumps(log_entry)
        
        formatter = JSONFormatter()
    else:
        # Standard logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / 'rag_system.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)

def create_data_directories(config_manager: ConfigManager):
    """Create necessary data directories"""
    print("     🔧 Getting config for directory creation...")
    config = config_manager.get_config()
    print("     📋 Config retrieved")
    
    directories = [
        config.data_dir,
        f"{config.data_dir}/metadata",
        f"{config.data_dir}/metadata/config",
        f"{config.data_dir}/vectors",
        f"{config.data_dir}/uploads",
        f"{config.data_dir}/logs",
        f"{config.data_dir}/backups",
        config.log_dir
    ]
    
    print(f"     📋 Creating {len(directories)} directories...")
    for i, directory in enumerate(directories):
        print(f"     📁 Creating directory {i+1}/{len(directories)}: {directory}")
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"     ✅ Created: {directory}")
        except Exception as e:
            print(f"     ❌ Failed to create {directory}: {e}")
            raise
    
    logging.info(f"Created data directories: {directories}")
    print("     ✅ All directories created successfully")

def validate_system_requirements(config_manager: ConfigManager):
    """Validate system requirements and configuration"""
    print("     🔧 Starting system requirements validation...")
    logging.info("Validating system requirements...")
    
    # Validate required configuration
    print("     📋 Validating required configuration...")
    config = config_manager.get_config()
    
    # Basic validation - check if config exists
    if not hasattr(config, 'environment'):
        raise ValueError("Invalid configuration: missing environment")
    
    print("     ✅ Configuration validation completed")
    
    # Check if LLM API key is provided (warn if missing)
    print("     🔧 Checking LLM API key...")
    if hasattr(config, 'llm') and hasattr(config.llm, 'api_key') and not config.llm.api_key:
        logging.warning("LLM API key not configured. Some features may not work.")
        print("     ⚠️ LLM API key not configured")
    else:
        print("     ✅ LLM API key check completed")
    
    # Validate data directories exist
    print("     🔧 Validating data directories...")
    if not Path(config.data_dir).exists():
        raise FileNotFoundError(f"Data directory does not exist: {config.data_dir}")
    print("     ✅ Data directories exist")
    
    # Check disk space (warn if low)
    print("     🔧 Checking disk space...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(config.data_dir)
        free_gb = free // (1024**3)
        print(f"     📊 Disk space: {free_gb}GB free")
        if free_gb < 1:
            logging.warning(f"Low disk space: {free_gb}GB free")
            print("     ⚠️ Low disk space warning")
        else:
            print("     ✅ Sufficient disk space")
    except Exception as e:
        print(f"     ⚠️ Could not check disk space: {e}")
        logging.warning(f"Could not check disk space: {e}")
    
    logging.info("System requirements validation completed")
    print("     ✅ System requirements validation completed")

def initialize_error_tracking(container: DependencyContainer):
    """Initialize error tracking system"""
    print("     🔧 Initializing error handler...")
    
    # Get or create error handler
    error_handler = get_error_handler()
    print("     ✅ Error handler initialized")
    
    # Register error handler in container
    container.register('error_handler', error_handler)
    print("     ✅ Error handler registered in container")
    
    logging.info("Error tracking initialized")
    return error_handler

def verify_dependencies():
    """Verify that all required dependencies are available"""
    required_packages = [
        'faiss',
        'sentence_transformers',
        'langchain',
        'fastapi',
        'uvicorn',
        'pydantic'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(f"Missing required packages: {', '.join(missing_packages)}")
    
    logging.info("All required dependencies are available")

def initialize_system() -> DependencyContainer:
    """Initialize the complete RAG system"""
    print("Initializing RAG System...")
    
    # Create dependency container
    container = DependencyContainer()
    
    try:
        # Step 1: Verify dependencies
        print("Step 1: Verifying dependencies...")
        verify_dependencies()
        print("   Dependencies verified")
        
        # Step 2: Register core services
        print("🔧 Step 2: Registering core services...")
        register_core_services(container)
        print("   ✅ Core services registered")
        
        # Step 3: Initialize configuration
        print("🔧 Step 3: Initializing configuration...")
        config_manager = container.get('config_manager')
        print("   ✅ Configuration initialized")
        
        # Step 4: Setup logging
        print("🔧 Step 4: Setting up logging...")
        setup_logging(config_manager)
        logging.info("RAG System initialization started")
        print("   ✅ Logging setup completed")
        
        # Step 5: Create data directories
        print("🔧 Step 5: Creating data directories...")
        create_data_directories(config_manager)
        print("   ✅ Data directories created")
        
        # Step 6: Validate system requirements
        print("🔧 Step 6: Validating system requirements...")
        validate_system_requirements(config_manager)
        print("   ✅ System requirements validated")
        
        # Step 7: Initialize error tracking
        print("🔧 Step 7: Initializing error tracking...")
        error_handler = initialize_error_tracking(container)
        print("   ✅ Error tracking initialized")
        
        # Step 8: Initialize core components
        logging.info("Initializing core components...")
        print("🔧 Step 8: Initializing core components...")
        
        # Initialize stores
        print("   📦 Initializing JSON store...")
        json_store = container.get('json_store')
        print("   ✅ JSON store initialized")
        
        print("   📦 Initializing metadata store...")
        metadata_store = container.get('metadata_store')
        print("   ✅ Metadata store initialized")
        
        # Initialize FAISS store
        print("   📦 Initializing FAISS store...")
        faiss_store = container.get('faiss_store')
        print("   ✅ FAISS store initialized")
        
        # Initialize embedder
        print("   📦 Initializing embedder...")
        embedder = container.get('embedder')
        print("   ✅ Embedder initialized")
        
        # Test LLM connection (if API key provided)
        print("   📦 Initializing LLM client...")
        config = config_manager.get_config()
        if config.llm.api_key:
            try:
                llm_client = container.get('llm_client')
                print("   ✅ LLM client initialized")
                logging.info(f"LLM client initialized: {config.llm.provider}")
            except Exception as e:
                print(f"   ⚠️ LLM client initialization failed: {e}")
                logging.warning(f"LLM client initialization failed: {e}")
        else:
            print("   ⚠️ LLM client skipped (no API key)")
        
        # Step 9: Save initial configuration
        print("🔧 Step 9: Saving configuration...")
        config_manager.save_config()
        print("   ✅ Configuration saved")
        
        logging.info("✅ RAG System initialization completed successfully")
        print("✅ RAG System initialization completed successfully")
        
        # Log system info
        print("🔧 Logging system info...")
        log_system_info(container)
        print("   ✅ System info logged")
        
        return container
        
    except Exception as e:
        logging.error(f"❌ RAG System initialization failed: {e}")
        raise

def log_system_info(container: DependencyContainer):
    """Log system information"""
    config_manager = container.get('config_manager')
    config = config_manager.get_config()
    
    system_info = {
        'environment': config.environment,
        'debug_mode': config.debug,
        'data_directory': config.data_dir,
        'embedding_model': config.embedding.model_name,
        'llm_provider': config.llm.provider,
        'llm_model': config.llm.model_name,
        'api_host': config.api.host,
        'api_port': config.api.port,
        'registered_services': container.list_services()
    }
    
    logging.info(f"System Configuration: {system_info}")

def health_check(container: DependencyContainer) -> Dict[str, Any]:
    """Perform system health check"""
    health_status = {
        'status': 'healthy',
        'timestamp': None,
        'components': {},
        'issues': []
    }
    
    from datetime import datetime
    health_status['timestamp'] = datetime.now().isoformat()
    
    try:
        # Check configuration
        config_manager = container.get('config_manager')
        config_issues = config_manager.validate_config()
        health_status['components']['config'] = {
            'status': 'healthy' if not config_issues else 'warning',
            'issues': config_issues
        }
        
        # Check data stores
        try:
            json_store = container.get('json_store')
            collections = json_store.list_collections()
            health_status['components']['json_store'] = {
                'status': 'healthy',
                'collections': len(collections)
            }
        except Exception as e:
            health_status['components']['json_store'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['issues'].append(f"JSON store error: {e}")
        
        # Check FAISS store
        try:
            faiss_store = container.get('faiss_store')
            index_info = faiss_store.get_index_info()
            health_status['components']['faiss_store'] = {
                'status': 'healthy',
                'index_size': index_info.get('ntotal', 0)
            }
        except Exception as e:
            health_status['components']['faiss_store'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['issues'].append(f"FAISS store error: {e}")
        
        # Check embedder
        try:
            embedder = container.get('embedder')
            # Test embedding
            test_embedding = embedder.embed_text("test")
            health_status['components']['embedder'] = {
                'status': 'healthy',
                'dimension': len(test_embedding)
            }
        except Exception as e:
            health_status['components']['embedder'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['issues'].append(f"Embedder error: {e}")
        
        # Check LLM client
        try:
            llm_client = container.get('llm_client')
            health_status['components']['llm_client'] = {
                'status': 'healthy',
                'provider': llm_client.provider
            }
        except Exception as e:
            health_status['components']['llm_client'] = {
                'status': 'warning',
                'error': str(e)
            }
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if 'error' in component_statuses:
            health_status['status'] = 'unhealthy'
        elif 'warning' in component_statuses:
            health_status['status'] = 'degraded'
        
    except Exception as e:
        health_status['status'] = 'error'
        health_status['issues'].append(f"Health check failed: {e}")
    
    return health_status 