#!/usr/bin/env python3
"""
RAG System UI Startup
Consolidated script for launching various UI interfaces
"""

import logging
import sys
import argparse
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class UILauncher:
    """UI launcher with multiple interface options"""
    
    def __init__(self):
        self.processes = []
        self.threads = []
        self.base_dir = Path(__file__).parent.parent
        
    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def launch_gradio_ui(self, port: int = 7860, share: bool = False):
        """Launch the main Gradio UI"""
        logging.info(f"🎨 Starting Gradio UI on port {port}...")
        try:
            # Import and run Gradio UI
            sys.path.insert(0, str(self.base_dir / "src"))
            from src.api.gradio_ui import create_gradio_interface
            
            # Initialize system first
            from src.core.system_init import initialize_system
            container = initialize_system()
            
            # Create and launch interface
            interface = create_gradio_interface(container)
            interface.launch(
                server_port=port,
                share=share,
                server_name="0.0.0.0",
                show_error=True
            )
            
        except Exception as e:
            logging.error(f"❌ Failed to launch Gradio UI: {e}")
            raise
    
    def launch_servicenow_ui(self, port: int = 7861, share: bool = False):
        """Launch the ServiceNow UI"""
        logging.info(f"🎫 Starting ServiceNow UI on port {port}...")
        try:
            # Import and run ServiceNow UI
            sys.path.insert(0, str(self.base_dir / "src"))
            from src.api.servicenow_ui import create_servicenow_interface
            
            # Initialize system first
            from src.core.system_init import initialize_system
            container = initialize_system()
            
            # Create and launch interface
            interface = create_servicenow_interface(container)
            interface.launch(
                server_port=port,
                share=share,
                server_name="0.0.0.0",
                show_error=True
            )
            
        except Exception as e:
            logging.error(f"❌ Failed to launch ServiceNow UI: {e}")
            raise
    
    def launch_comprehensive_ui(self, port: int = 7862, share: bool = False):
        """Launch the comprehensive management UI"""
        logging.info(f"🔧 Starting Comprehensive UI on port {port}...")
        try:
            # Run the comprehensive UI script
            script_path = self.base_dir / "launch_comprehensive_ui.py"
            if script_path.exists():
                subprocess.run([sys.executable, str(script_path)])
            else:
                logging.error("❌ Comprehensive UI script not found")
                
        except Exception as e:
            logging.error(f"❌ Failed to launch Comprehensive UI: {e}")
            raise
    
    def launch_enhanced_ui(self, port: int = 7863, share: bool = False):
        """Launch the enhanced UI v2"""
        logging.info(f"✨ Starting Enhanced UI v2 on port {port}...")
        try:
            # Run the enhanced UI script
            script_path = self.base_dir / "launch_enhanced_ui_v2.py"
            if script_path.exists():
                subprocess.run([sys.executable, str(script_path)])
            else:
                logging.error("❌ Enhanced UI script not found")
                
        except Exception as e:
            logging.error(f"❌ Failed to launch Enhanced UI: {e}")
            raise
    
    def launch_multiple_uis(self, ui_types: List[str], ports: List[int], share: bool = False):
        """Launch multiple UIs simultaneously"""
        logging.info(f"🚀 Starting multiple UIs: {', '.join(ui_types)}")
        
        ui_functions = {
            'gradio': self.launch_gradio_ui,
            'servicenow': self.launch_servicenow_ui,
            'comprehensive': self.launch_comprehensive_ui,
            'enhanced': self.launch_enhanced_ui
        }
        
        for i, ui_type in enumerate(ui_types):
            if ui_type in ui_functions:
                port = ports[i] if i < len(ports) else 7860 + i
                
                # Launch in separate thread
                thread = threading.Thread(
                    target=ui_functions[ui_type],
                    args=(port, share),
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
                
                # Small delay between launches
                time.sleep(2)
        
        # Display access URLs
        print("\n" + "="*60)
        print("🎨 RAG SYSTEM UI INTERFACES")
        print("="*60)
        
        for i, ui_type in enumerate(ui_types):
            port = ports[i] if i < len(ports) else 7860 + i
            print(f"📊 {ui_type.title()} UI: http://localhost:{port}")
        
        print("\nPress Ctrl+C to stop all UIs")
        print("="*60)
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("🛑 Shutting down all UIs...")
    
    def show_ui_menu(self):
        """Show interactive UI selection menu"""
        print("\n" + "="*60)
        print("🎨 RAG SYSTEM UI LAUNCHER")
        print("="*60)
        print("1. Gradio UI (Main Interface)")
        print("2. ServiceNow UI (Enterprise Integration)")
        print("3. Comprehensive UI (Full Management)")
        print("4. Enhanced UI v2 (Modern Interface)")
        print("5. Multiple UIs (Select multiple)")
        print("6. Exit")
        print("-"*60)
        
        while True:
            choice = input("Select UI option (1-6): ").strip()
            
            if choice == "1":
                port = int(input("Port (default 7860): ") or "7860")
                share = input("Share publicly? (y/n): ").lower() == 'y'
                self.launch_gradio_ui(port, share)
                break
                
            elif choice == "2":
                port = int(input("Port (default 7861): ") or "7861")
                share = input("Share publicly? (y/n): ").lower() == 'y'
                self.launch_servicenow_ui(port, share)
                break
                
            elif choice == "3":
                self.launch_comprehensive_ui()
                break
                
            elif choice == "4":
                self.launch_enhanced_ui()
                break
                
            elif choice == "5":
                print("\nAvailable UIs: gradio, servicenow, comprehensive, enhanced")
                ui_list = input("Enter UI types (comma-separated): ").strip()
                ui_types = [ui.strip() for ui in ui_list.split(',')]
                
                ports_input = input("Enter ports (comma-separated, optional): ").strip()
                if ports_input:
                    ports = [int(p.strip()) for p in ports_input.split(',')]
                else:
                    ports = list(range(7860, 7860 + len(ui_types)))
                
                share = input("Share publicly? (y/n): ").lower() == 'y'
                self.launch_multiple_uis(ui_types, ports, share)
                break
                
            elif choice == "6":
                print("👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid choice. Please select 1-6.")

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="RAG System UI Launcher")
    parser.add_argument("--ui", choices=["gradio", "servicenow", "comprehensive", "enhanced"],
                       help="UI type to launch")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Share UI publicly")
    parser.add_argument("--multiple", nargs="+", 
                       choices=["gradio", "servicenow", "comprehensive", "enhanced"],
                       help="Launch multiple UIs")
    parser.add_argument("--ports", nargs="+", type=int, help="Ports for multiple UIs")
    parser.add_argument("--log-level", default="info", 
                       choices=["debug", "info", "warning", "error"],
                       help="Log level")
    parser.add_argument("--interactive", action="store_true", 
                       help="Show interactive menu")
    
    args = parser.parse_args()
    
    launcher = UILauncher()
    launcher.setup_logging(args.log_level.upper())
    
    try:
        if args.interactive or (not args.ui and not args.multiple):
            # Show interactive menu
            launcher.show_ui_menu()
            
        elif args.multiple:
            # Launch multiple UIs
            ports = args.ports or list(range(7860, 7860 + len(args.multiple)))
            launcher.launch_multiple_uis(args.multiple, ports, args.share)
            
        elif args.ui == "gradio":
            launcher.launch_gradio_ui(args.port, args.share)
            
        elif args.ui == "servicenow":
            launcher.launch_servicenow_ui(args.port, args.share)
            
        elif args.ui == "comprehensive":
            launcher.launch_comprehensive_ui(args.port, args.share)
            
        elif args.ui == "enhanced":
            launcher.launch_enhanced_ui(args.port, args.share)
            
    except KeyboardInterrupt:
        logging.info("🛑 UI launcher stopped by user")
    except Exception as e:
        logging.error(f"❌ UI launcher failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 