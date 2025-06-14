"""
Comprehensive Heartbeat Monitoring System for RAG System
Monitors all system components and provides detailed health status
"""
import asyncio
import logging
import time
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import threading
from contextlib import contextmanager
import os

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    """Individual component health status"""
    name: str
    status: HealthStatus
    response_time_ms: float
    last_check: str
    details: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        return result

@dataclass
class SystemHealth:
    """Overall system health"""
    overall_status: HealthStatus
    timestamp: str
    uptime_seconds: int
    components: List[ComponentHealth]
    performance_metrics: Dict[str, Any]
    alerts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['overall_status'] = self.overall_status.value
        result['components'] = [comp.to_dict() for comp in self.components]
        return result

class HeartbeatMonitor:
    """Comprehensive system health monitoring for RAG system"""
    
    def __init__(self, container=None):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
        self.check_interval = 30  # seconds
        self.is_running = False
        self.last_health_check = None
        self.last_check_time = None
        self.health_history = []
        self.max_history = 100
        self.interval = 30  # For API compatibility
        
        # Performance thresholds
        self.thresholds = {
            'api_response_time_ms': 5000,  # 5 seconds
            'memory_usage_percent': 85,
            'cpu_usage_percent': 80,
            'disk_usage_percent': 90,
            'vector_search_time_ms': 1000,
            'embedding_time_ms': 3000,
            'llm_response_time_ms': 10000,
            'query_response_time_ms': 5000
        }
        
        self.logger.info("Heartbeat monitor initialized")
    
    async def comprehensive_health_check(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        start_time = time.time()
        components = []
        alerts = []
        
        self.logger.info("🔍 Starting comprehensive health check...")
        
        # Check all components
        components.append(await self._check_api_server())
        components.append(await self._check_storage_layer())
        components.append(await self._check_vector_store())
        components.append(await self._check_embeddings())
        components.append(await self._check_llm_service())
        components.append(await self._check_dependency_container())
        components.append(await self._check_ingestion_engine())
        components.append(await self._check_query_engine())
        components.append(await self._check_system_resources())
        
        # Calculate overall status
        critical_components = [c for c in components if c.status == HealthStatus.CRITICAL]
        warning_components = [c for c in components if c.status == HealthStatus.WARNING]
        
        if critical_components:
            overall_status = HealthStatus.CRITICAL
            alerts.append(f"Critical issues detected in: {', '.join([c.name for c in critical_components])}")
        elif warning_components:
            overall_status = HealthStatus.WARNING
            alerts.append(f"Warnings detected in: {', '.join([c.name for c in warning_components])}")
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Get performance metrics
        performance_metrics = await self._get_performance_metrics()
        
        # Check performance thresholds
        for metric, threshold in self.thresholds.items():
            if metric in performance_metrics and performance_metrics[metric] > threshold:
                alerts.append(f"Performance threshold exceeded: {metric} = {performance_metrics[metric]} > {threshold}")
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.WARNING
        
        # Create system health object
        system_health = SystemHealth(
            overall_status=overall_status,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=int(time.time() - self.start_time),
            components=components,
            performance_metrics=performance_metrics,
            alerts=alerts
        )
        
        # Store health check result
        self.last_health_check = system_health
        self.last_check_time = time.time()
        self._store_health_history(system_health)
        
        check_duration = (time.time() - start_time) * 1000
        self.logger.info(f"✅ Health check completed in {check_duration:.0f}ms - Status: {overall_status.value}")
        
        return system_health
    
    async def _check_api_server(self) -> ComponentHealth:
        """Check FastAPI server health"""
        start_time = time.time()
        
        try:
            # Test basic endpoint
            response = requests.get("http://localhost:8000/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                details = {
                    "status_code": response.status_code,
                    "response_data": response.json(),
                    "server_host": "localhost",
                    "server_port": 8000
                }
                error_msg = None
            else:
                status = HealthStatus.WARNING
                details = {"status_code": response.status_code}
                error_msg = f"Unexpected status code: {response.status_code}"
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            status = HealthStatus.CRITICAL
            details = {"error_type": type(e).__name__}
            error_msg = str(e)
        
        return ComponentHealth(
            name="API Server",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_storage_layer(self) -> ComponentHealth:
        """Check storage layer (memory stores and file system)"""
        start_time = time.time()
        
        try:
            # Check data directories exist and are writable
            data_path = Path("data")
            required_dirs = ["metadata", "vectors", "logs", "config"]
            missing_dirs = [d for d in required_dirs if not (data_path / d).exists()]
            
            # Test write permissions
            test_file = data_path / "health_check_test.txt"
            try:
                test_file.write_text("health check test")
                test_file.unlink()
                write_permission = True
            except Exception:
                write_permission = False
            
            if not missing_dirs and write_permission:
                status = HealthStatus.HEALTHY
                error_msg = None
            else:
                status = HealthStatus.WARNING
                error_msg = f"Missing directories: {missing_dirs}" if missing_dirs else "Write permission denied"
            
            # Get storage statistics
            if data_path.exists():
                disk_usage = psutil.disk_usage(str(data_path))
                details = {
                    "data_directory": str(data_path),
                    "directories_status": {d: (data_path / d).exists() for d in required_dirs},
                    "write_permission": write_permission,
                    "disk_total_gb": round(disk_usage.total / (1024**3), 2),
                    "disk_used_gb": round(disk_usage.used / (1024**3), 2),
                    "disk_free_gb": round(disk_usage.free / (1024**3), 2),
                    "disk_usage_percent": round((disk_usage.used / disk_usage.total) * 100, 1)
                }
            else:
                details = {"data_directory": "Not found"}
                status = HealthStatus.CRITICAL
                error_msg = "Data directory does not exist"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            details = {"error_type": type(e).__name__}
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="Storage Layer",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_vector_store(self) -> ComponentHealth:
        """Check FAISS vector store"""
        start_time = time.time()
        
        try:
            if self.container:
                faiss_store = self.container.get('faiss_store')
                
                # Get vector store statistics
                stats = faiss_store.get_stats()
                
                # Test vector search if vectors exist
                if stats['vector_count'] > 0:
                    # Get dynamic dimension from config
                    config_manager = self.container.get('config_manager')
                    embedding_config = config_manager.get_config('embedding')
                    
                    # Import dimension helper
                    try:
                        from ..core.constants import get_embedding_dimension
                    except ImportError:
                        import sys
                        from pathlib import Path
                        sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
                        from constants import get_embedding_dimension
                    
                    dimension = get_embedding_dimension(
                        embedding_config.provider, 
                        embedding_config.model_name
                    )
                    
                    # Create dummy query vector with correct dimension
                    dummy_vector = [0.1] * dimension
                    search_start = time.time()
                    try:
                        results = faiss_store.search(dummy_vector, k=min(5, stats['vector_count']))
                        search_time = (time.time() - search_start) * 1000
                        search_successful = len(results) > 0
                    except Exception:
                        search_time = 0
                        search_successful = False
                else:
                    search_time = 0
                    search_successful = True  # No vectors to search
                
                if search_successful:
                    status = HealthStatus.HEALTHY
                    error_msg = None
                else:
                    status = HealthStatus.WARNING
                    error_msg = "Vector search failed"
                
                details = {
                    **stats,
                    "search_test_time_ms": search_time,
                    "search_successful": search_successful,
                    "index_file_exists": Path("data/vectors/faiss_index.bin").exists()
                }
            else:
                status = HealthStatus.WARNING
                details = {"container": "Not available"}
                error_msg = "Container not provided"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            details = {"error_type": type(e).__name__}
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="Vector Store (FAISS)",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_embeddings(self) -> ComponentHealth:
        """Check Cohere embedding service"""
        start_time = time.time()
        
        try:
            if self.container:
                embedder = self.container.get('embedder')
                
                # Test embedding generation
                test_text = "Health check test for embedding service"
                embeddings = embedder.embed_texts([test_text])
                
                # Get expected dimension from config
                config_manager = self.container.get('config_manager')
                embedding_config = config_manager.get_config('embedding')
                
                # Import dimension helper
                try:
                    from ..core.constants import get_embedding_dimension
                except ImportError:
                    import sys
                    from pathlib import Path
                    sys.path.insert(0, str(Path(__file__).parent.parent / 'core'))
                    from constants import get_embedding_dimension
                
                expected_dim = get_embedding_dimension(
                    embedding_config.provider, 
                    embedding_config.model_name
                )
                actual_dim = len(embeddings[0]) if embeddings else 0
                
                if actual_dim == expected_dim and all(isinstance(x, (int, float)) for x in embeddings[0]):
                    status = HealthStatus.HEALTHY
                    error_msg = None
                else:
                    status = HealthStatus.CRITICAL
                    error_msg = f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
                
                details = {
                    "provider": embedding_config.provider,
                    "model": embedding_config.model_name,
                    "expected_dimension": expected_dim,
                    "actual_dimension": actual_dim,
                    "test_embedding_sample": embeddings[0][:5] if embeddings and len(embeddings[0]) >= 5 else [],
                    "embedding_successful": len(embeddings) > 0 and len(embeddings[0]) > 0
                }
            else:
                status = HealthStatus.WARNING
                details = {"container": "Not available"}
                error_msg = "Container not provided"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            # Get config for error details
            try:
                config_manager = self.container.get('config_manager')
                embedding_config = config_manager.get_config('embedding')
                provider = embedding_config.provider
                model = embedding_config.model_name
            except:
                provider = "unknown"
                model = "unknown"
            
            details = {
                "provider": provider,
                "model": model,
                "error_type": type(e).__name__
            }
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        # Get service name from config
        try:
            config_manager = self.container.get('config_manager')
            embedding_config = config_manager.get_config('embedding')
            service_name = f"Embedding Service ({embedding_config.provider.title()})"
        except:
            service_name = "Embedding Service"
        
        return ComponentHealth(
            name=service_name,
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_llm_service(self) -> ComponentHealth:
        """Check Groq LLM service"""
        start_time = time.time()
        
        try:
            if self.container:
                llm_client = self.container.get('llm_client')
                
                # Test LLM generation
                test_prompt = "Respond with exactly: 'Health check successful'"
                response = llm_client.generate(test_prompt, max_tokens=10, temperature=0.0)
                
                # Verify response
                response_valid = len(response) > 0 and "health check" in response.lower()
                
                if response_valid:
                    status = HealthStatus.HEALTHY
                    error_msg = None
                else:
                    status = HealthStatus.WARNING
                    error_msg = f"Unexpected LLM response: {response[:100]}"
                
                details = {
                    "provider": "groq",
                    "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "test_response": response[:100],
                    "response_length": len(response),
                    "response_valid": response_valid
                }
            else:
                status = HealthStatus.WARNING
                details = {"container": "Not available"}
                error_msg = "Container not provided"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            details = {
                "provider": "groq",
                "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
                "error_type": type(e).__name__
            }
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="LLM Service (Groq)",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_dependency_container(self) -> ComponentHealth:
        """Check dependency injection container"""
        start_time = time.time()
        
        try:
            if self.container:
                # Test getting all registered services
                services = self.container.list_services()
                expected_services = [
                    'config_manager', 'json_store', 'metadata_store', 'log_store',
                    'faiss_store', 'embedder', 'chunker', 'llm_client',
                    'query_engine', 'ingestion_engine'
                ]
                
                missing_services = [s for s in expected_services if s not in services]
                
                # Test creating a service
                try:
                    config_manager = self.container.get('config_manager')
                    service_creation_ok = config_manager is not None
                except Exception:
                    service_creation_ok = False
                
                if not missing_services and service_creation_ok:
                    status = HealthStatus.HEALTHY
                    error_msg = None
                else:
                    status = HealthStatus.WARNING
                    error_msg = f"Missing services: {missing_services}" if missing_services else "Service creation failed"
                
                details = {
                    "registered_services": services,
                    "expected_services": expected_services,
                    "missing_services": missing_services,
                    "service_creation_test": service_creation_ok,
                    "total_services": len(services)
                }
            else:
                status = HealthStatus.CRITICAL
                details = {"container": "Not available"}
                error_msg = "Dependency container not provided"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            details = {"error_type": type(e).__name__}
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="Dependency Container",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_ingestion_engine(self) -> ComponentHealth:
        """Check ingestion engine"""
        start_time = time.time()
        
        try:
            if self.container:
                ingestion_engine = self.container.get('ingestion_engine')
                
                # Test text ingestion
                test_text = "This is a health check test for the ingestion engine."
                test_metadata = {"source": "health_check", "timestamp": datetime.now().isoformat()}
                
                # Create temporary test file
                test_file = Path("data/health_check_test.txt")
                test_file.parent.mkdir(parents=True, exist_ok=True)
                test_file.write_text(test_text)
                
                try:
                    result = ingestion_engine.ingest_file(str(test_file), test_metadata)
                    ingestion_successful = result.get('status') == 'success'
                    chunks_created = result.get('chunks_created', 0)
                    
                    # Clean up test file
                    test_file.unlink(missing_ok=True)
                    
                    if ingestion_successful:
                        status = HealthStatus.HEALTHY
                        error_msg = None
                    else:
                        status = HealthStatus.WARNING
                        error_msg = "Ingestion test failed"
                    
                    details = {
                        "ingestion_test": ingestion_successful,
                        "chunks_created": chunks_created,
                        "test_result": result
                    }
                except Exception as e:
                    test_file.unlink(missing_ok=True)
                    raise e
            else:
                status = HealthStatus.WARNING
                details = {"container": "Not available"}
                error_msg = "Container not provided"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            details = {"error_type": type(e).__name__}
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="Ingestion Engine",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_query_engine(self) -> ComponentHealth:
        """Check query engine"""
        start_time = time.time()
        
        try:
            if self.container:
                query_engine = self.container.get('query_engine')
                
                # Test query processing
                test_query = "What is the system status?"
                
                try:
                    response = query_engine.process_query(test_query)
                    query_successful = len(response.get('answer', '')) > 0
                    sources_found = len(response.get('sources', []))
                    
                    if query_successful:
                        status = HealthStatus.HEALTHY
                        error_msg = None
                    else:
                        status = HealthStatus.WARNING
                        error_msg = "Query test returned empty response"
                    
                    details = {
                        "query_test": query_successful,
                        "sources_found": sources_found,
                        "response_length": len(response.get('answer', '')),
                        "test_query": test_query
                    }
                except Exception as query_error:
                    # Query might fail if no vectors exist, which is not critical
                    status = HealthStatus.WARNING
                    error_msg = f"Query test failed: {str(query_error)}"
                    details = {
                        "query_test": False,
                        "error": str(query_error)
                    }
            else:
                status = HealthStatus.WARNING
                details = {"container": "Not available"}
                error_msg = "Container not provided"
                
        except Exception as e:
            status = HealthStatus.CRITICAL
            details = {"error_type": type(e).__name__}
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="Query Engine",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resources (CPU, Memory, Disk)"""
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Determine status based on thresholds
            warnings = []
            if cpu_percent > self.thresholds['cpu_usage_percent']:
                warnings.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > self.thresholds['memory_usage_percent']:
                warnings.append(f"High memory usage: {memory.percent}%")
            if disk.percent > self.thresholds['disk_usage_percent']:
                warnings.append(f"High disk usage: {disk.percent}%")
            
            if warnings:
                status = HealthStatus.WARNING
                error_msg = "; ".join(warnings)
            else:
                status = HealthStatus.HEALTHY
                error_msg = None
            
            details = {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": round(disk.percent, 1),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "cpu_count": psutil.cpu_count(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            details = {"error_type": type(e).__name__}
            error_msg = str(e)
        
        response_time = (time.time() - start_time) * 1000
        
        return ComponentHealth(
            name="System Resources",
            status=status,
            response_time_ms=response_time,
            last_check=datetime.now().isoformat(),
            details=details,
            error_message=error_msg
        )
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # RAG system metrics
            vector_count = 0
            if self.container:
                try:
                    faiss_store = self.container.get('faiss_store')
                    stats = faiss_store.get_stats()
                    vector_count = stats.get('vector_count', 0)
                except Exception:
                    pass
            
            # Performance metrics (removed API key exposure for security)
            metrics = {
                "system_cpu_percent": round(cpu_percent, 1),
                "system_memory_percent": round(memory.percent, 1),
                "system_disk_percent": round(disk.percent, 1),
                "total_vectors": vector_count,
                "uptime_seconds": int(time.time() - self.start_time),
                "uptime_hours": round((time.time() - self.start_time) / 3600, 1)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}
    
    def _store_health_history(self, health: SystemHealth):
        """Store health check in history"""
        self.health_history.append(health.to_dict())
        
        # Keep only recent history
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
        
        # Save to file
        try:
            logs_dir = Path("data/logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            health_file = logs_dir / "health_history.json"
            with open(health_file, 'w') as f:
                json.dump({
                    'history': self.health_history,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save health history: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        if not self.last_health_check:
            return {"status": "unknown", "message": "No health check performed yet"}
        
        return {
            "overall_status": self.last_health_check.overall_status.value,
            "timestamp": self.last_health_check.timestamp,
            "uptime_hours": round((time.time() - self.start_time) / 3600, 1),
            "component_count": len(self.last_health_check.components),
            "healthy_components": len([c for c in self.last_health_check.components if c.status == HealthStatus.HEALTHY]),
            "warning_components": len([c for c in self.last_health_check.components if c.status == HealthStatus.WARNING]),
            "critical_components": len([c for c in self.last_health_check.components if c.status == HealthStatus.CRITICAL]),
            "alerts": self.last_health_check.alerts
        }
    
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("🚀 Starting continuous health monitoring...")
        
        def monitor_loop():
            while self.is_running:
                try:
                    # Run health check
                    health = asyncio.run(self.comprehensive_health_check())
                    
                    # Log status
                    if health.overall_status == HealthStatus.CRITICAL:
                        self.logger.error(f"🚨 CRITICAL system health issues detected")
                    elif health.overall_status == HealthStatus.WARNING:
                        self.logger.warning(f"⚠️ System health warnings detected")
                    else:
                        self.logger.info(f"✅ System health check passed")
                    
                    # Wait for next check
                    time.sleep(self.check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.check_interval)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous health monitoring"""
        self.is_running = False
        self.logger.info("🛑 Stopping health monitoring...")

# Global instance (will be initialized with container in main.py)
heartbeat_monitor = None

def initialize_heartbeat_monitor(container):
    """Initialize the global heartbeat monitor with dependency container"""
    global heartbeat_monitor
    heartbeat_monitor = HeartbeatMonitor(container)
    return heartbeat_monitor 