#!/usr/bin/env python3
"""
RAG System Startup Script
Simple script to start the RAG system with basic configuration
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_environment():
    """Check if environment is properly configured"""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        template_file = Path(".env.template")
        if template_file.exists():
            import shutil
            shutil.copy(template_file, env_file)
            print("✅ .env file created from template")
            print("📝 Please edit .env file and add your API keys")
        else:
            print("❌ .env.template not found")
            return False
    
    # Check data directories
    data_dir = Path("data")
    if not data_dir.exists():
        print("📁 Creating data directories...")
        os.system("python scripts/setup.py")
    
    print("✅ Environment check completed")
    return True

def start_system():
    """Start the RAG system"""
    try:
        from src.core.system_init import initialize_system
        from src.api.main import create_api_app
        
        print("🚀 Starting RAG System...")
        
        # Initialize system
        container = initialize_system()
        print("✅ System initialized")
        
        # Create API app
        app = create_api_app(container)
        print("✅ API application created")
        
        # Start server
        import uvicorn
        config_manager = container.get('config_manager')
        config = config_manager.get_config()
        
        print(f"🌐 Starting server on http://{config.api.host}:{config.api.port}")
        print(f"📚 API Documentation: http://{config.api.host}:{config.api.port}/docs")
        print("🛑 Press Ctrl+C to stop the server")
        
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start system: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("🎯 RAG System Startup")
    print("=" * 50)
    
    if not check_environment():
        print("❌ Environment check failed")
        return
    
    print("\n🚀 Starting RAG System...")
    start_system()

if __name__ == "__main__":
    main() 