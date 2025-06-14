#!/usr/bin/env python3
"""
RAG System API Server Startup
Dedicated script for starting only the API server without UI components
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )

def start_api_server(host: str = "0.0.0.0", port: int = 8000, 
                    workers: int = 1, reload: bool = False,
                    log_level: str = "info"):
    """Start the API server only"""
    try:
        logging.info("🚀 Starting RAG System API Server...")
        
        # Initialize core system
        from src.core.system_init import initialize_system
        logging.info("Initializing system components...")
        container = initialize_system()
        
        # Setup monitoring (optional for API-only mode)
        monitoring = None
        try:
            from src.monitoring.setup import setup_monitoring
            logging.info("Setting up monitoring...")
            monitoring = setup_monitoring(container.get('config_manager'))
        except ImportError:
            logging.warning("Monitoring setup not available")
        
        # Initialize heartbeat monitor (optional)
        heartbeat_monitor = None
        try:
            from src.monitoring.heartbeat_monitor import initialize_heartbeat_monitor
            logging.info("Initializing heartbeat monitor...")
            heartbeat_monitor = initialize_heartbeat_monitor(container)
            logging.info("✅ Heartbeat monitor initialized")
        except Exception as e:
            logging.warning(f"Heartbeat monitor initialization failed: {e}")
        
        # Initialize folder monitor (optional)
        folder_monitor = None
        try:
            from src.monitoring.folder_monitor import initialize_folder_monitor
            logging.info("Initializing folder monitor...")
            config_manager = container.get('config_manager')
            folder_monitor = initialize_folder_monitor(container, config_manager)
            logging.info("✅ Folder monitor initialized")
        except Exception as e:
            logging.warning(f"Folder monitor initialization failed: {e}")
        
        # Register monitors with API module
        try:
            import src.api.main as api_main
            api_main.heartbeat_monitor = heartbeat_monitor
            
            import src.monitoring.folder_monitor as folder_monitor_module
            folder_monitor_module.folder_monitor = folder_monitor
            logging.info("✅ Monitors registered with API")
        except Exception as e:
            logging.warning(f"Failed to register monitors: {e}")
        
        # Create API app
        from src.api.main import create_api_app
        logging.info("Creating FastAPI application...")
        api_app = create_api_app(container, monitoring, heartbeat_monitor)
        
        # Start monitoring services if available
        if heartbeat_monitor:
            config = container.get('config_manager').get_config()
            heartbeat_enabled = getattr(config, 'heartbeat', {}).get('enabled', False)
            if heartbeat_enabled:
                logging.info("Starting heartbeat monitoring...")
                heartbeat_monitor.start_monitoring()
        
        if folder_monitor:
            config = container.get('config_manager').get_config()
            folder_config = getattr(config, 'folder_monitoring', None)
            if folder_config and getattr(folder_config, 'enabled', True):
                monitored_folders = getattr(folder_config, 'monitored_folders', [])
                if monitored_folders:
                    logging.info(f"Starting folder monitoring for {len(monitored_folders)} folders...")
                    result = folder_monitor.start_monitoring()
                    if result.get('success'):
                        logging.info("✅ Folder monitoring started")
        
        # Start API server
        import uvicorn
        logging.info(f"🌐 Starting API server on {host}:{port}")
        logging.info(f"📖 API Documentation: http://{host}:{port}/docs")
        logging.info(f"🔍 Health Check: http://{host}:{port}/health")
        
        uvicorn.run(
            api_app,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level
        )
        
    except KeyboardInterrupt:
        logging.info("🛑 Received shutdown signal")
    except Exception as e:
        logging.error(f"❌ Failed to start API server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="RAG System API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"],
                       help="Log level")
    parser.add_argument("--log-file", help="Log file path (optional)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level.upper(), args.log_file)
    
    # Start API server
    start_api_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main() 