#!/usr/bin/env python3
"""
Restart API Script
Safely restart the RAG API server
"""
import subprocess
import time
import requests
import sys
import os
import signal
from pathlib import Path

API_URL = "http://localhost:8000"
API_PORT = 8000

def find_process_on_port(port):
    """Find process running on specified port"""
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(
                ['netstat', '-ano'], 
                capture_output=True, 
                text=True
            )
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        return int(pid)
        else:  # Unix/Linux
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'], 
                capture_output=True, 
                text=True
            )
            if result.stdout.strip():
                return int(result.stdout.strip())
    except Exception as e:
        print(f"Error finding process: {e}")
    return None

def kill_process(pid):
    """Kill process by PID"""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
        else:  # Unix/Linux
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Process already dead
        print(f"✅ Killed process {pid}")
        return True
    except Exception as e:
        print(f"❌ Failed to kill process {pid}: {e}")
        return False

def check_api_health():
    """Check if API is responding"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server"""
    try:
        print("🚀 Starting API server...")
        
        # Change to rag-system directory
        rag_dir = Path(__file__).parent
        os.chdir(rag_dir)
        
        # Start the server in background
        if os.name == 'nt':  # Windows
            subprocess.Popen(
                ['python', 'main.py'],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:  # Unix/Linux
            subprocess.Popen(
                ['python', 'main.py'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        for i in range(30):  # Wait up to 30 seconds
            time.sleep(1)
            if check_api_health():
                print("✅ API server started successfully!")
                return True
            print(f"   Waiting... ({i+1}/30)")
        
        print("❌ API server failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return False

def main():
    """Main restart function"""
    print("🔄 RAG API Server Restart Script")
    print("=" * 40)
    
    # Check if API is currently running
    print("🔍 Checking current API status...")
    if check_api_health():
        print("✅ API is currently running")
        
        # Ask user if they want to restart
        response = input("Do you want to restart it? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Restart cancelled")
            return 0
    else:
        print("❌ API is not responding")
    
    # Find and kill existing process
    print(f"🔍 Looking for process on port {API_PORT}...")
    pid = find_process_on_port(API_PORT)
    
    if pid:
        print(f"📍 Found process {pid} on port {API_PORT}")
        if kill_process(pid):
            time.sleep(3)  # Wait for cleanup
        else:
            print("⚠️  Failed to kill process, continuing anyway...")
    else:
        print("ℹ️  No process found on port")
    
    # Start new server
    if start_api_server():
        print("\n🎉 API server restarted successfully!")
        print(f"🌐 API available at: {API_URL}")
        print("📖 API docs at: http://localhost:8000/docs")
        return 0
    else:
        print("\n❌ Failed to restart API server")
        print("💡 Try running manually: python main.py")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 