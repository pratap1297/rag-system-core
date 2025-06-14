"""
Basic Monitoring System for RAG System
"""

import time
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .logging_system import get_logging_manager

@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.logging_manager = get_logging_manager()
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float):
        """Record a metric value"""
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=time.time()
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric_point)
        
        # Keep only last 1000 points per metric
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def record_api_request(self, method: str, path: str, status_code: int, response_time: float):
        """Record API request metrics"""
        
        self.record_metric("api_requests_total", 1)
        self.record_metric("api_response_time_seconds", response_time)
        
        self.logging_manager.log_api_request(
            method=method,
            path=path,
            status_code=status_code,
            response_time=response_time
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "uptime_seconds": time.time() - self.start_time
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get basic health status"""
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if cpu_percent > 90 or memory.percent > 90:
                status = "unhealthy"
            elif cpu_percent > 70 or memory.percent > 80:
                status = "degraded"
            
            return {
                "overall_status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "last_updated": time.time()
            }
        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "last_updated": time.time()
            }

# Global monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor 