#!/usr/bin/env python3
"""
Launch script for RAG System UI with ServiceNow integration
"""
import subprocess
import sys
import time
import threading
import os
from pathlib import Path

def launch_main_ui():
    """Launch the main RAG system UI"""
    print("🚀 Starting main RAG system UI on port 7860...")
    try:
        subprocess.run([
            sys.executable, "-m", "src.api.gradio_ui"
        ], cwd=Path(__file__).parent)
    except Exception as e:
        print(f"❌ Error launching main UI: {e}")

def launch_servicenow_ui():
    """Launch the ServiceNow UI"""
    print("🎫 Starting ServiceNow UI on port 7861...")
    time.sleep(2)  # Wait a bit for main UI to start
    try:
        subprocess.run([
            sys.executable, "-m", "src.api.servicenow_ui"
        ], cwd=Path(__file__).parent)
    except Exception as e:
        print(f"❌ Error launching ServiceNow UI: {e}")

def main():
    """Launch both UIs"""
    print("🔧 RAG System with ServiceNow Integration")
    print("=" * 50)
    
    # Start main UI in a separate thread
    main_ui_thread = threading.Thread(target=launch_main_ui, daemon=True)
    main_ui_thread.start()
    
    # Start ServiceNow UI in a separate thread
    servicenow_ui_thread = threading.Thread(target=launch_servicenow_ui, daemon=True)
    servicenow_ui_thread.start()
    
    print("\n✅ Both UIs are starting...")
    print("📊 Main RAG System UI: http://localhost:7860")
    print("🎫 ServiceNow UI: http://localhost:7861")
    print("\nPress Ctrl+C to stop both UIs")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down UIs...")
        sys.exit(0)

if __name__ == "__main__":
    main() 