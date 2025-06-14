#!/usr/bin/env python3
"""
RAG System Full Startup
Complete system launcher with API and UI interfaces
"""

import logging
import sys
import argparse
import subprocess
import threading
import time
import signal
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class SystemLauncher:
    """Full system launcher with API and UI components"""
    
    def __init__(self):
        self.processes = []
        self.threads = []
        self.base_dir = Path(__file__).parent.parent
        self.shutdown_event = threading.Event()
        
    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def print_banner(self):
        """Print system banner"""
        print("=" * 70)
        print("🏗️  RAG SYSTEM CORE - FULL SYSTEM STARTUP")
        print("=" * 70)
        print("Enterprise RAG System with Multi-Provider AI/ML Support")
        print("Features: FastAPI, FAISS, Gradio UI, ServiceNow Integration")
        print("=" * 70)
    
    def check_requirements(self):
        """Check if requirements are installed"""
        try:
            import fastapi
            import gradio
            import sentence_transformers
            import faiss
            logging.info("✅ Core dependencies verified")
            return True
        except ImportError as e:
            logging.error(f"❌ Missing dependency: {e}")
            logging.error("Please run: pip install -r requirements.txt")
            return False
    
    def check_environment(self):
        """Check environment setup"""
        env_file = self.base_dir / ".env"
        if not env_file.exists():
            logging.warning("⚠️  .env file not found. Copying from template...")
            template_file = self.base_dir / ".env.template"
            if template_file.exists():
                import shutil
                shutil.copy(template_file, env_file)
                logging.info("✅ .env file created. Please edit it with your API keys.")
            else:
                logging.error("❌ .env.template not found")
                return False
        return True
    
    def start_api_server(self, host: str = "0.0.0.0", port: int = 8000, 
                        workers: int = 1, reload: bool = False):
        """Start API server in background thread"""
        def run_api():
            try:
                logging.info(f"🌐 Starting API server on {host}:{port}")
                
                # Initialize core system
                from src.core.system_init import initialize_system
                container = initialize_system()
                
                # Setup monitoring
                monitoring = None
                try:
                    from src.monitoring.setup import setup_monitoring
                    monitoring = setup_monitoring(container.get('config_manager'))
                except ImportError:
                    logging.warning("Monitoring setup not available")
                
                # Initialize monitors
                heartbeat_monitor = None
                folder_monitor = None
                
                try:
                    from src.monitoring.heartbeat_monitor import initialize_heartbeat_monitor
                    heartbeat_monitor = initialize_heartbeat_monitor(container)
                    logging.info("✅ Heartbeat monitor initialized")
                except Exception as e:
                    logging.warning(f"Heartbeat monitor failed: {e}")
                
                try:
                    from src.monitoring.folder_monitor import initialize_folder_monitor
                    config_manager = container.get('config_manager')
                    folder_monitor = initialize_folder_monitor(container, config_manager)
                    logging.info("✅ Folder monitor initialized")
                except Exception as e:
                    logging.warning(f"Folder monitor failed: {e}")
                
                # Register monitors
                try:
                    import src.api.main as api_main
                    api_main.heartbeat_monitor = heartbeat_monitor
                    
                    import src.monitoring.folder_monitor as folder_monitor_module
                    folder_monitor_module.folder_monitor = folder_monitor
                    logging.info("✅ Monitors registered")
                except Exception as e:
                    logging.warning(f"Failed to register monitors: {e}")
                
                # Create API app
                from src.api.main import create_api_app
                api_app = create_api_app(container, monitoring, heartbeat_monitor)
                
                # Start monitoring services
                if heartbeat_monitor:
                    config = container.get('config_manager').get_config()
                    if getattr(config, 'heartbeat', {}).get('enabled', False):
                        heartbeat_monitor.start_monitoring()
                
                if folder_monitor:
                    config = container.get('config_manager').get_config()
                    folder_config = getattr(config, 'folder_monitoring', None)
                    if folder_config and getattr(folder_config, 'enabled', True):
                        monitored_folders = getattr(folder_config, 'monitored_folders', [])
                        if monitored_folders:
                            result = folder_monitor.start_monitoring()
                            if result.get('success'):
                                logging.info("✅ Folder monitoring started")
                
                # Start API server
                import uvicorn
                uvicorn.run(
                    api_app,
                    host=host,
                    port=port,
                    workers=workers,
                    reload=reload,
                    log_level="info"
                )
                
            except Exception as e:
                logging.error(f"❌ API server failed: {e}")
                self.shutdown_event.set()
        
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        self.threads.append(api_thread)
        
        # Wait a moment for API to start
        time.sleep(3)
        return api_thread
    
    def start_ui_interface(self, ui_type: str = "gradio", port: int = 7860, 
                          share: bool = False):
        """Start UI interface in background thread"""
        def run_ui():
            try:
                if ui_type == "gradio":
                    logging.info(f"🎨 Starting Gradio UI on port {port}")
                    from src.api.gradio_ui import create_gradio_interface
                    from src.core.system_init import initialize_system
                    
                    container = initialize_system()
                    interface = create_gradio_interface(container)
                    interface.launch(
                        server_port=port,
                        share=share,
                        server_name="0.0.0.0",
                        show_error=True
                    )
                    
                elif ui_type == "servicenow":
                    logging.info(f"🎫 Starting ServiceNow UI on port {port}")
                    from src.api.servicenow_ui import create_servicenow_interface
                    from src.core.system_init import initialize_system
                    
                    container = initialize_system()
                    interface = create_servicenow_interface(container)
                    interface.launch(
                        server_port=port,
                        share=share,
                        server_name="0.0.0.0",
                        show_error=True
                    )
                    
            except Exception as e:
                logging.error(f"❌ UI {ui_type} failed: {e}")
                self.shutdown_event.set()
        
        ui_thread = threading.Thread(target=run_ui, daemon=True)
        ui_thread.start()
        self.threads.append(ui_thread)
        return ui_thread
    
    def start_full_system(self, api_port: int = 8000, ui_type: str = "gradio", 
                         ui_port: int = 7860, share: bool = False):
        """Start complete system with API and UI"""
        logging.info("🚀 Starting RAG System - Full Stack")
        
        # Start API server
        api_thread = self.start_api_server(port=api_port)
        
        # Start UI interface
        ui_thread = self.start_ui_interface(ui_type, ui_port, share)
        
        # Display system information
        self.display_system_info(api_port, ui_type, ui_port)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Keep system running
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("🛑 Shutdown signal received")
        finally:
            self.shutdown()
    
    def display_system_info(self, api_port: int, ui_type: str, ui_port: int):
        """Display system access information"""
        print("\n" + "="*70)
        print("🎯 RAG SYSTEM - READY FOR USE")
        print("="*70)
        print(f"🌐 API Server: http://localhost:{api_port}")
        print(f"📖 API Documentation: http://localhost:{api_port}/docs")
        print(f"🔍 Health Check: http://localhost:{api_port}/health")
        print(f"🎨 {ui_type.title()} UI: http://localhost:{ui_port}")
        print("\n💡 System Features:")
        print("   • Document ingestion and processing")
        print("   • Vector similarity search")
        print("   • AI-powered question answering")
        print("   • ServiceNow integration")
        print("   • Real-time monitoring")
        print("   • Folder monitoring")
        print("\nPress Ctrl+C to shutdown the system")
        print("="*70)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}")
        self.shutdown_event.set()
    
    def shutdown(self):
        """Graceful system shutdown"""
        logging.info("🛑 Shutting down RAG System...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Terminate processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
        
        logging.info("✅ System shutdown complete")
    
    def show_startup_menu(self):
        """Show interactive startup menu"""
        print("\n" + "="*70)
        print("🚀 RAG SYSTEM - STARTUP OPTIONS")
        print("="*70)
        print("1. Full System (API + Gradio UI)")
        print("2. Full System (API + ServiceNow UI)")
        print("3. API Server Only")
        print("4. UI Only (Gradio)")
        print("5. UI Only (ServiceNow)")
        print("6. Custom Configuration")
        print("7. Exit")
        print("-"*70)
        
        while True:
            choice = input("Select startup option (1-7): ").strip()
            
            if choice == "1":
                api_port = int(input("API Port (default 8000): ") or "8000")
                ui_port = int(input("UI Port (default 7860): ") or "7860")
                share = input("Share UI publicly? (y/n): ").lower() == 'y'
                self.start_full_system(api_port, "gradio", ui_port, share)
                break
                
            elif choice == "2":
                api_port = int(input("API Port (default 8000): ") or "8000")
                ui_port = int(input("UI Port (default 7861): ") or "7861")
                share = input("Share UI publicly? (y/n): ").lower() == 'y'
                self.start_full_system(api_port, "servicenow", ui_port, share)
                break
                
            elif choice == "3":
                api_port = int(input("API Port (default 8000): ") or "8000")
                self.start_api_server(port=api_port)
                print(f"🌐 API Server running on http://localhost:{api_port}")
                self._wait_for_shutdown()
                break
                
            elif choice == "4":
                ui_port = int(input("UI Port (default 7860): ") or "7860")
                share = input("Share publicly? (y/n): ").lower() == 'y'
                self.start_ui_interface("gradio", ui_port, share)
                self._wait_for_shutdown()
                break
                
            elif choice == "5":
                ui_port = int(input("UI Port (default 7861): ") or "7861")
                share = input("Share publicly? (y/n): ").lower() == 'y'
                self.start_ui_interface("servicenow", ui_port, share)
                self._wait_for_shutdown()
                break
                
            elif choice == "6":
                self._custom_configuration()
                break
                
            elif choice == "7":
                print("👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please select 1-7.")
    
    def _custom_configuration(self):
        """Handle custom configuration"""
        print("\n🔧 Custom Configuration")
        print("-" * 30)
        
        # API configuration
        start_api = input("Start API server? (y/n): ").lower() == 'y'
        api_port = 8000
        if start_api:
            api_port = int(input("API Port (default 8000): ") or "8000")
        
        # UI configuration
        start_ui = input("Start UI interface? (y/n): ").lower() == 'y'
        ui_type = "gradio"
        ui_port = 7860
        share = False
        
        if start_ui:
            ui_type = input("UI Type (gradio/servicenow): ").strip() or "gradio"
            ui_port = int(input(f"UI Port (default {7860 if ui_type == 'gradio' else 7861}): ") 
                         or str(7860 if ui_type == 'gradio' else 7861))
            share = input("Share publicly? (y/n): ").lower() == 'y'
        
        # Start configured system
        if start_api and start_ui:
            self.start_full_system(api_port, ui_type, ui_port, share)
        elif start_api:
            self.start_api_server(port=api_port)
            self._wait_for_shutdown()
        elif start_ui:
            self.start_ui_interface(ui_type, ui_port, share)
            self._wait_for_shutdown()
        else:
            print("❌ No components selected to start")
    
    def _wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            while not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("🛑 Shutdown signal received")
        finally:
            self.shutdown()

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="RAG System Full Startup")
    parser.add_argument("--mode", choices=["full", "api", "ui"], default="full",
                       help="Startup mode")
    parser.add_argument("--api-port", type=int, default=8000, help="API server port")
    parser.add_argument("--ui-type", choices=["gradio", "servicenow"], default="gradio",
                       help="UI interface type")
    parser.add_argument("--ui-port", type=int, default=7860, help="UI port")
    parser.add_argument("--share", action="store_true", help="Share UI publicly")
    parser.add_argument("--workers", type=int, default=1, help="API server workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Log level")
    parser.add_argument("--interactive", action="store_true", 
                       help="Show interactive menu")
    
    args = parser.parse_args()
    
    launcher = SystemLauncher()
    launcher.setup_logging(args.log_level.upper())
    launcher.print_banner()
    
    # Check requirements and environment
    if not launcher.check_requirements() or not launcher.check_environment():
        sys.exit(1)
    
    try:
        if args.interactive or len(sys.argv) == 1:
            # Show interactive menu
            launcher.show_startup_menu()
            
        elif args.mode == "full":
            launcher.start_full_system(
                api_port=args.api_port,
                ui_type=args.ui_type,
                ui_port=args.ui_port,
                share=args.share
            )
            
        elif args.mode == "api":
            launcher.start_api_server(
                port=args.api_port,
                workers=args.workers,
                reload=args.reload
            )
            launcher._wait_for_shutdown()
            
        elif args.mode == "ui":
            launcher.start_ui_interface(
                ui_type=args.ui_type,
                port=args.ui_port,
                share=args.share
            )
            launcher._wait_for_shutdown()
            
    except KeyboardInterrupt:
        logging.info("🛑 System startup cancelled by user")
    except Exception as e:
        logging.error(f"❌ System startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 