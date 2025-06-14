#!/usr/bin/env python3
"""
RAG System Entry Point
"""
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main entry point"""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logging.info("Starting RAG System...")
        
        # Initialize core system
        from src.core.system_init import initialize_system
        logging.info("Initializing system components...")
        container = initialize_system()
        
        # Setup monitoring
        try:
            from src.monitoring.setup import setup_monitoring
            logging.info("Setting up monitoring...")
            monitoring = setup_monitoring(container.get('config_manager'))
        except ImportError:
            logging.warning("Monitoring setup not available, using basic monitoring")
            monitoring = None
        
        # Initialize heartbeat monitor
        heartbeat_monitor = None
        try:
            from src.monitoring.heartbeat_monitor import initialize_heartbeat_monitor
            logging.info("Initializing heartbeat monitor...")
            heartbeat_monitor = initialize_heartbeat_monitor(container)
            logging.info("✅ Heartbeat monitor initialized successfully")
        except Exception as e:
            logging.warning(f"Heartbeat monitor initialization failed: {e}")
            logging.info("System will continue without heartbeat monitoring")
        
        # Initialize folder monitor
        folder_monitor = None
        try:
            from src.monitoring.folder_monitor import initialize_folder_monitor
            logging.info("Initializing folder monitor...")
            config_manager = container.get('config_manager')
            folder_monitor = initialize_folder_monitor(container, config_manager)
            logging.info("✅ Folder monitor initialized successfully")
        except Exception as e:
            logging.warning(f"Folder monitor initialization failed: {e}")
            logging.info("System will continue without folder monitoring")
        
        # Update the global monitors in the API module
        try:
            import src.api.main as api_main
            api_main.heartbeat_monitor = heartbeat_monitor
            logging.info("✅ Heartbeat monitor registered with API")
        except Exception as e:
            logging.warning(f"Failed to register heartbeat monitor with API: {e}")
        
        # Register folder monitor with API module
        try:
            import src.monitoring.folder_monitor as folder_monitor_module
            folder_monitor_module.folder_monitor = folder_monitor
            logging.info("✅ Folder monitor registered with API")
        except Exception as e:
            logging.warning(f"Failed to register folder monitor with API: {e}")
        
        # Create API app
        from src.api.main import create_api_app
        logging.info("Creating FastAPI application...")
        api_app = create_api_app(container, monitoring, heartbeat_monitor)
        logging.info("FastAPI application created successfully")
        
        # Start continuous health monitoring (if enabled in config)
        if heartbeat_monitor:
            config_manager = container.get('config_manager')
            config = config_manager.get_config()
            
            # Check if heartbeat is enabled in config (default: False)
            heartbeat_enabled = getattr(config, 'heartbeat', {}).get('enabled', False)
            
            if heartbeat_enabled:
                logging.info("Starting continuous health monitoring...")
                heartbeat_monitor.start_monitoring()
            else:
                logging.info("Heartbeat monitoring disabled in config")
        else:
            logging.info("Heartbeat monitor not available, skipping continuous monitoring")
        
        # Start folder monitoring (if enabled in config and folders are configured)
        if folder_monitor:
            config_manager = container.get('config_manager')
            config = config_manager.get_config()
            
            # Check if folder monitoring is enabled in config
            folder_config = getattr(config, 'folder_monitoring', None)
            if folder_config and getattr(folder_config, 'enabled', True):
                monitored_folders = getattr(folder_config, 'monitored_folders', [])
                if monitored_folders:
                    logging.info(f"Starting folder monitoring for {len(monitored_folders)} folders...")
                    result = folder_monitor.start_monitoring()
                    if result.get('success'):
                        logging.info("✅ Folder monitoring started successfully")
                    else:
                        logging.warning(f"Failed to start folder monitoring: {result.get('error')}")
                else:
                    logging.info("No folders configured for monitoring")
            else:
                logging.info("Folder monitoring disabled in config")
        else:
            logging.info("Folder monitor not available, skipping folder monitoring")
        
        # Start API server
        import uvicorn
        config_manager = container.get('config_manager')
        config = config_manager.get_config()
        
        logging.info(f"Starting server on {config.api.host}:{config.api.port}")
        
        uvicorn.run(
            api_app,
            host=config.api.host,
            port=config.api.port,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Failed to start RAG System: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 