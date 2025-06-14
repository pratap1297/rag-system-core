"""
Dependency Injection Container
Manages system components and their dependencies
"""
from typing import Dict, Any, Callable, Optional, TypeVar, Type
import threading
from functools import wraps

T = TypeVar('T')

class DependencyContainer:
    """Simple dependency injection container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._lock = threading.RLock()  # Use reentrant lock to allow recursive calls
        self._creating = set()  # Track services being created to prevent circular deps
    
    def register(self, name: str, factory: Callable, singleton: bool = True):
        """Register a service factory"""
        with self._lock:
            self._factories[name] = factory
            if not singleton and name in self._singletons:
                del self._singletons[name]
    
    def register_instance(self, name: str, instance: Any):
        """Register a service instance directly"""
        with self._lock:
            self._services[name] = instance
            self._singletons[name] = instance
    
    def get(self, name: str) -> Any:
        """Get a service instance"""
        # Check if already instantiated (thread-safe read)
        if name in self._singletons:
            return self._singletons[name]
        
        if name in self._services:
            return self._services[name]
        
        # Use lock for creation
        with self._lock:
            # Double-check after acquiring lock
            if name in self._singletons:
                return self._singletons[name]
            
            if name in self._services:
                return self._services[name]
            
            # Check for circular dependency
            if hasattr(self, '_creating') and name in self._creating:
                raise RuntimeError(f"Circular dependency detected for service '{name}'")
            
            # Create from factory
            if name in self._factories:
                if not hasattr(self, '_creating'):
                    self._creating = set()
                
                self._creating.add(name)
                try:
                    factory = self._factories[name]
                    instance = factory(self)
                    self._singletons[name] = instance
                    return instance
                finally:
                    self._creating.discard(name)
            
            raise KeyError(f"Service '{name}' not registered")
    
    def has(self, name: str) -> bool:
        """Check if service is registered"""
        return name in self._services or name in self._factories or name in self._singletons
    
    def list_services(self) -> list:
        """List all registered services"""
        all_services = set()
        all_services.update(self._services.keys())
        all_services.update(self._factories.keys())
        all_services.update(self._singletons.keys())
        return list(all_services)

def inject(*dependencies):
    """Decorator for dependency injection"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Assume first argument is the container
            if args and hasattr(args[0], 'get'):
                container = args[0]
                injected_deps = {}
                
                for dep_name in dependencies:
                    if dep_name not in kwargs:
                        injected_deps[dep_name] = container.get(dep_name)
                
                kwargs.update(injected_deps)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Service registration helpers
def create_config_manager(container: DependencyContainer):
    """Factory for ConfigManager"""
    from .config_manager import ConfigManager
    return ConfigManager()

def create_json_store(container: DependencyContainer):
    """Factory for JSONStore"""
    print(f"     🔧 Creating JSON store...")
    from .json_store import JSONStore
    print(f"     📋 JSONStore imported")
    # Use default path to avoid circular dependency with config_manager
    json_store = JSONStore("data")
    print(f"     ✅ JSON store created successfully")
    return json_store

def create_metadata_store(container: DependencyContainer):
    """Factory for PersistentJSONMetadataStore"""
    try:
        from ..storage.persistent_metadata_store import PersistentJSONMetadataStore
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from storage.persistent_metadata_store import PersistentJSONMetadataStore
    # Use default path to avoid circular dependency with config_manager
    return PersistentJSONMetadataStore("data/metadata")

def create_log_store(container: DependencyContainer):
    """Factory for MemoryLogStore"""
    print(f"     🔧 Creating memory log store...")
    from .memory_store import MemoryLogStore
    print(f"     📋 Creating log store with default path...")
    # Use default path to avoid circular dependency with config_manager
    log_store = MemoryLogStore("data/logs")
    print(f"     ✅ Memory log store created successfully")
    return log_store

def create_faiss_store(container: DependencyContainer):
    """Factory for FAISSStore"""
    print(f"     🔧 Creating FAISS store...")
    try:
        from ..storage.faiss_store import FAISSStore
        from .constants import get_embedding_dimension
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from storage.faiss_store import FAISSStore
        sys.path.insert(0, str(Path(__file__).parent))
        from constants import get_embedding_dimension
    
    # Get config to determine embedding dimension
    config_manager = container.get('config_manager')
    embedding_config = config_manager.get_config('embedding')
    
    # Get dynamic dimension based on provider and model
    dimension = get_embedding_dimension(
        embedding_config.provider, 
        embedding_config.model_name
    )
    
    print(f"     📋 FAISS config: path=data/vectors/faiss_index.bin, dimension={dimension}")
    faiss_store = FAISSStore(
        index_path="data/vectors/faiss_index.bin",
        dimension=dimension
    )
    print(f"     ✅ FAISS store created successfully with dimension {dimension}")
    return faiss_store

def create_embedder(container: DependencyContainer):
    """Factory for Embedder"""
    print(f"     🔧 Creating embedder...")
    try:
        from ..ingestion.embedder import Embedder
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ingestion.embedder import Embedder
    import os
    
    # Get config to use correct embedding provider and model
    config_manager = container.get('config_manager')
    embedding_config = config_manager.get_config('embedding')
    
    print(f"     📋 Embedder config: provider={embedding_config.provider}, model={embedding_config.model_name}")
    embedder = Embedder(
        provider=embedding_config.provider,
        model_name=embedding_config.model_name,
        device=embedding_config.device,
        batch_size=embedding_config.batch_size,
        api_key=embedding_config.api_key
    )
    print(f"     ✅ Embedder created successfully")
    return embedder

def create_chunker(container: DependencyContainer):
    """Factory for Chunker"""
    try:
        from ..ingestion.chunker import Chunker
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ingestion.chunker import Chunker
    # Use default values to avoid circular dependency with config_manager
    # Enable semantic chunking by default for better performance
    return Chunker(
        chunk_size=1000,
        chunk_overlap=200,
        use_semantic=True  # Enable semantic chunking
    )

def create_llm_client(container: DependencyContainer):
    """Factory for LLMClient"""
    try:
        from ..retrieval.llm_client import LLMClient
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from retrieval.llm_client import LLMClient
    import os
    # Use default values to avoid circular dependency with config_manager
    return LLMClient(
        provider="groq",
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=1000
    )

def create_reranker(container: DependencyContainer):
    """Factory for Reranker"""
    print(f"     🔧 Creating reranker...")
    try:
        from ..retrieval.reranker import create_reranker as create_reranker_func
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from retrieval.reranker import create_reranker as create_reranker_func
    
    config_manager = container.get('config_manager')
    reranker = create_reranker_func(config_manager)
    print(f"     ✅ Reranker created successfully: {reranker.get_model_info()['model_name']}")
    return reranker

def create_query_enhancer(container: DependencyContainer):
    """Factory for QueryEnhancer"""
    print(f"     🔧 Creating query enhancer...")
    try:
        from ..retrieval.query_enhancer import create_query_enhancer as create_query_enhancer_func
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from retrieval.query_enhancer import create_query_enhancer as create_query_enhancer_func
    
    config_manager = container.get('config_manager')
    query_enhancer = create_query_enhancer_func(config_manager)
    print(f"     ✅ Query enhancer created successfully: {query_enhancer.get_enhancer_info()['enhancer_type']}")
    return query_enhancer

def create_query_engine(container: DependencyContainer):
    """Factory for QueryEngine"""
    try:
        from ..retrieval.query_engine import QueryEngine
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from retrieval.query_engine import QueryEngine
    return QueryEngine(
        faiss_store=container.get('faiss_store'),
        embedder=container.get('embedder'),
        llm_client=container.get('llm_client'),
        metadata_store=container.get('metadata_store'),
        config_manager=container.get('config_manager'),
        reranker=container.get('reranker'),
        query_enhancer=container.get('query_enhancer')
    )

def create_ingestion_engine(container: DependencyContainer):
    """Factory for IngestionEngine"""
    try:
        from ..ingestion.ingestion_engine import IngestionEngine
    except ImportError:
        # Fallback for when running as script
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ingestion.ingestion_engine import IngestionEngine
    return IngestionEngine(
        chunker=container.get('chunker'),
        embedder=container.get('embedder'),
        faiss_store=container.get('faiss_store'),
        metadata_store=container.get('metadata_store'),
        config_manager=container.get('config_manager')
    )

def create_servicenow_integration(container: DependencyContainer):
    """Factory for ServiceNow Integration"""
    print(f"     🔧 Creating ServiceNow integration...")
    try:
        from ..integrations.servicenow import ServiceNowIntegration
        config_manager = container.get('config_manager')
        ingestion_engine = container.get('ingestion_engine')
        
        integration = ServiceNowIntegration(
            config_manager=config_manager,
            ingestion_engine=ingestion_engine
        )
        print(f"     ✅ ServiceNow integration created successfully")
        return integration
    except Exception as e:
        print(f"     ⚠️ ServiceNow integration creation failed: {e}")
        # Return None if ServiceNow integration fails - it's optional
        return None

def register_core_services(container: DependencyContainer):
    """Register all core services"""
    container.register('config_manager', create_config_manager)
    container.register('json_store', create_json_store)
    container.register('metadata_store', create_metadata_store)
    container.register('log_store', create_log_store)
    container.register('faiss_store', create_faiss_store)
    container.register('embedder', create_embedder)
    container.register('chunker', create_chunker)
    container.register('llm_client', create_llm_client)
    container.register('reranker', create_reranker)
    container.register('query_enhancer', create_query_enhancer)
    container.register('query_engine', create_query_engine)
    container.register('ingestion_engine', create_ingestion_engine)
    container.register('servicenow_integration', create_servicenow_integration)

# Global container instance for API access
_global_container = None

def get_dependency_container() -> DependencyContainer:
    """Get the global dependency container"""
    global _global_container
    if _global_container is None:
        raise RuntimeError("Dependency container not initialized")
    return _global_container

def set_dependency_container(container: DependencyContainer):
    """Set the global dependency container"""
    global _global_container
    _global_container = container 