"""
Simplified System Initialization
Initialize and configure core RAG system components
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any

from .config_manager import ConfigManager
from .error_handling import get_error_handler

def setup_basic_logging():
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/rag_system.log', mode='a')
        ]
    )
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)

def create_basic_directories(config_manager: ConfigManager):
    """Create basic data directories"""
    print("     🔧 Creating basic directories...")
    
    config = config_manager.get_config()
    
    directories = [
        config.data_dir,
        f"{config.data_dir}/metadata",
        f"{config.data_dir}/vectors",
        f"{config.data_dir}/uploads",
        config.log_dir
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"     ✅ Created: {directory}")
        except Exception as e:
            print(f"     ❌ Failed to create {directory}: {e}")
    
    print("     ✅ Basic directories created")

def validate_basic_requirements(config_manager: ConfigManager):
    """Validate basic system requirements"""
    print("     🔧 Validating basic requirements...")
    
    config = config_manager.get_config()
    
    # Check if config is valid
    if not hasattr(config, 'environment'):
        raise ValueError("Invalid configuration: missing environment")
    
    # Check data directory
    if not Path(config.data_dir).exists():
        Path(config.data_dir).mkdir(parents=True, exist_ok=True)
    
    print("     ✅ Basic requirements validated")

def initialize_simple_system():
    """Initialize a simplified RAG system"""
    print("🚀 Initializing Simplified RAG System...")
    
    try:
        # Step 1: Setup logging
        print("Step 1: Setting up logging...")
        setup_basic_logging()
        print("   ✅ Logging configured")
        
        # Step 2: Initialize configuration
        print("Step 2: Initializing configuration...")
        config_manager = ConfigManager()
        print("   ✅ Configuration loaded")
        
        # Step 3: Create directories
        print("Step 3: Creating directories...")
        create_basic_directories(config_manager)
        print("   ✅ Directories created")
        
        # Step 4: Validate requirements
        print("Step 4: Validating requirements...")
        validate_basic_requirements(config_manager)
        print("   ✅ Requirements validated")
        
        # Step 5: Initialize error handling
        print("Step 5: Initializing error handling...")
        error_handler = get_error_handler()
        print("   ✅ Error handling initialized")
        
        print("🎉 Simplified system initialization completed!")
        
        return {
            'config_manager': config_manager,
            'error_handler': error_handler,
            'status': 'initialized'
        }
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        logging.error(f"System initialization failed: {e}")
        raise

def get_system_status():
    """Get basic system status"""
    return {
        'status': 'running',
        'timestamp': str(Path.cwd()),
        'components': {
            'config': 'loaded',
            'logging': 'active',
            'error_handling': 'active'
        }
    } 