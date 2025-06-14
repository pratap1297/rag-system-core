#!/usr/bin/env python3
"""
Diagnostic Script for RAG System
Helps identify what's causing the main script to hang
"""
import sys
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def timeout_handler(func, timeout=30):
    """Run a function with timeout"""
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"⏰ Function timed out after {timeout} seconds")
        return None, "TIMEOUT"
    elif exception[0]:
        return None, exception[0]
    else:
        return result[0], None

def test_step(step_name, func, timeout=30):
    """Test a step with timeout"""
    print(f"🔍 Testing: {step_name}")
    start_time = time.time()
    
    result, error = timeout_handler(func, timeout)
    elapsed = time.time() - start_time
    
    if error == "TIMEOUT":
        print(f"❌ {step_name} - TIMED OUT after {timeout}s")
        return False
    elif error:
        print(f"❌ {step_name} - ERROR: {error}")
        return False
    else:
        print(f"✅ {step_name} - OK ({elapsed:.2f}s)")
        return True

def main():
    """Run diagnostics"""
    print("🔍 RAG System Diagnostic Tool")
    print("=" * 50)
    
    # Test 1: Basic imports
    def test_basic_imports():
        import fastapi
        import uvicorn
        import logging
        return True
    
    test_step("Basic imports", test_basic_imports)
    
    # Test 2: Core imports
    def test_core_imports():
        from src.core.config_manager import ConfigManager
        from src.core.dependency_container import DependencyContainer
        return True
    
    test_step("Core imports", test_core_imports)
    
    # Test 3: Config manager creation
    def test_config_manager():
        from src.core.config_manager import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        return config
    
    test_step("Config manager creation", test_config_manager)
    
    # Test 4: Dependency container
    def test_dependency_container():
        from src.core.dependency_container import DependencyContainer, register_core_services
        container = DependencyContainer()
        register_core_services(container)
        # Test getting a service
        config_manager = container.get('config_manager')
        return container
    
    test_step("Dependency container", test_dependency_container)
    
    # Test 5: System initialization (this is likely where it hangs)
    def test_system_init():
        from src.core.system_init import initialize_system
        container = initialize_system()
        return container
    
    if not test_step("System initialization", test_system_init, timeout=60):
        print("\n🚨 SYSTEM INITIALIZATION IS HANGING!")
        print("This is likely where the main script gets stuck.")
        
        # Test individual parts of system init
        print("\n🔍 Testing individual system init components...")
        
        def test_verify_deps():
            from src.core.system_init import verify_dependencies
            verify_dependencies()
            return True
        
        test_step("Dependency verification", test_verify_deps)
        
        def test_create_dirs():
            from src.core.config_manager import ConfigManager
            from src.core.system_init import create_data_directories
            config_manager = ConfigManager()
            create_data_directories(config_manager)
            return True
        
        test_step("Directory creation", test_create_dirs)
        
        def test_validate_reqs():
            from src.core.config_manager import ConfigManager
            from src.core.system_init import validate_system_requirements
            config_manager = ConfigManager()
            validate_system_requirements(config_manager)
            return True
        
        test_step("System requirements validation", test_validate_reqs, timeout=30)
        
        def test_embedder():
            from src.ingestion.embedder import Embedder
            embedder = Embedder()
            return embedder
        
        test_step("Embedder initialization", test_embedder, timeout=60)
        
        def test_faiss():
            from src.storage.faiss_store import FAISSStore
            faiss_store = FAISSStore()
            return faiss_store
        
        test_step("FAISS store initialization", test_faiss)
        
        return
    
    # Test 6: API creation
    def test_api_creation():
        from src.core.system_init import initialize_system
        from src.api.main import create_api_app
        container = initialize_system()
        api_app = create_api_app(container, None)
        return api_app
    
    test_step("API creation", test_api_creation, timeout=90)
    
    print("\n✅ All tests completed!")
    print("If you see this message, the system should work normally.")

if __name__ == "__main__":
    main() 