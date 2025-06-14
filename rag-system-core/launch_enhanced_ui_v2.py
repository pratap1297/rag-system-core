#!/usr/bin/env python3
"""
Enhanced RAG System UI Launcher v2
=================================
Launches the enhanced Gradio interface with heartbeat monitoring and comprehensive testing
"""

import sys
import os
import time
import requests
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_server_status(api_url: str = "http://localhost:8000", max_retries: int = 5) -> bool:
    """Check if the RAG system server is running"""
    print(f"🔍 Checking server status at {api_url}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server is running and healthy!")
                return True
            else:
                print(f"⚠️ Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries}: Server not responding - {e}")
            if attempt < max_retries - 1:
                print("⏳ Waiting 3 seconds before retry...")
                time.sleep(3)
    
    return False

def launch_enhanced_ui():
    """Launch the enhanced Gradio UI"""
    print("🚀 RAG System Enhanced Management UI Launcher v2")
    print("=" * 60)
    
    # Check if server is running
    if not check_server_status():
        print("\n❌ RAG System server is not running!")
        print("📋 To start the server, run:")
        print("   python main.py")
        print("\n🔄 Or if you're in the rag-system directory:")
        print("   cd rag-system && python main.py")
        return False
    
    # Import and launch the enhanced UI
    try:
        print("\n📦 Loading enhanced UI components...")
        from src.api.gradio_ui_enhanced_v2 import create_enhanced_ui
        
        print("🎛️ Creating enhanced interface...")
        interface = create_enhanced_ui()
        
        print("\n🌟 ENHANCED RAG SYSTEM MANAGEMENT UI v2")
        print("=" * 50)
        print("🌐 API Server: http://localhost:8000")
        print("🎛️ Management UI: http://localhost:7862")
        print("\n🔧 Features Available:")
        print("  💓 Real-time Heartbeat Monitoring")
        print("  🧪 Comprehensive RAG Testing (3 Core Tests)")
        print("  📊 System Performance Metrics")
        print("  🔍 Component Health Monitoring")
        print("  📋 System Diagnostics")
        print("  📈 Statistics Dashboard")
        print("\n🧪 Core RAG Tests:")
        print("  1. 📝 Learn from Documents - Verify system can ingest and retrieve unique information")
        print("  2. 🔄 Update Knowledge - Test document updates and knowledge refresh")
        print("  3. 🗑️ Delete Documents - Ensure deleted documents are removed from search")
        print("\n💓 Heartbeat Monitoring:")
        print("  • Real-time system status")
        print("  • Component health checks")
        print("  • Performance metrics")
        print("  • Manual health check triggers")
        print("\n🎯 Ready to launch! Press Ctrl+C to stop the UI")
        print("=" * 50)
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7862,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except ImportError as e:
        print(f"\n❌ Failed to import enhanced UI: {e}")
        print("📋 Make sure you're in the correct directory and all dependencies are installed")
        return False
    except Exception as e:
        print(f"\n❌ Failed to launch enhanced UI: {e}")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = launch_enhanced_ui()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Enhanced UI stopped by user")
        print("Thank you for using the RAG System Enhanced Management Interface!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1) 