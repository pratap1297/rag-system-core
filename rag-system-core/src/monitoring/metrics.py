"""
Metrics and Performance Monitoring for RAG System
Provides comprehensive system metrics, performance tracking, and health monitoring
"""

import time
import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: Optional[float] = None

@dataclass
class ApplicationMetrics:
    """Application-level metrics"""
    requests_total: int = 0
    requests_per_second: float = 0.0
    response_time_avg: float = 0.0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    queue_size: int = 0

class MetricsCollector:
    """Collects and manages system and application metrics"""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque())
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
        self._cleanup_thread = None
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for metric cleanup"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_old_metrics()
                    time.sleep(300)  # Cleanup every 5 minutes
                except Exception as e:
                    logger.error(f"Metrics cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_old_metrics(self):
        """Remove old metric points beyond retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for metric_name, points in self.metrics.items():
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
    
    def record_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Record a counter metric"""
        with self.lock:
            self.counters[name] += value
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=self.counters[name],
                tags=tags or {}
            ))
    
    def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a gauge metric"""
        with self.lock:
            self.gauges[name] = value
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags=tags or {}
            ))
    
    def record_timer(self, name: str, duration: float, tags: Dict[str, str] = None):
        """Record a timer metric"""
        with self.lock:
            self.timers[name].append(duration)
            # Keep only last 1000 measurements for percentile calculations
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            
            self.metrics[name].append(MetricPoint(
                timestamp=datetime.now(),
                value=duration,
                tags=tags or {}
            ))
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            # Load average (Unix-like systems only)
            load_average = None
            try:
                load_average = psutil.getloadavg()[0]
            except (AttributeError, OSError):
                pass  # Not available on Windows
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                load_average=load_average
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0)
    
    def get_application_metrics(self) -> ApplicationMetrics:
        """Get current application metrics"""
        with self.lock:
            # Calculate request rate
            requests_total = self.counters.get('requests_total', 0)
            
            # Calculate average response time
            response_times = self.timers.get('response_time', [])
            response_time_avg = sum(response_times) / len(response_times) if response_times else 0.0
            
            # Calculate 95th percentile response time
            if response_times:
                sorted_times = sorted(response_times)
                p95_index = int(len(sorted_times) * 0.95)
                response_time_p95 = sorted_times[p95_index] if p95_index < len(sorted_times) else 0.0
            else:
                response_time_p95 = 0.0
            
            # Calculate error rate
            errors_total = self.counters.get('errors_total', 0)
            error_rate = (errors_total / requests_total * 100) if requests_total > 0 else 0.0
            
            return ApplicationMetrics(
                requests_total=requests_total,
                response_time_avg=response_time_avg,
                response_time_p95=response_time_p95,
                error_rate=error_rate,
                active_connections=self.gauges.get('active_connections', 0),
                cache_hit_rate=self.gauges.get('cache_hit_rate', 0.0),
                queue_size=int(self.gauges.get('queue_size', 0))
            )
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[MetricPoint]:
        """Get metric history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            points = self.metrics.get(name, deque())
            return [point for point in points if point.timestamp >= cutoff_time]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        system_metrics = self.get_system_metrics()
        app_metrics = self.get_application_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'cpu_percent': system_metrics.cpu_percent,
                'memory_percent': system_metrics.memory_percent,
                'memory_used_mb': system_metrics.memory_used_mb,
                'memory_available_mb': system_metrics.memory_available_mb,
                'disk_usage_percent': system_metrics.disk_usage_percent,
                'disk_free_gb': system_metrics.disk_free_gb,
                'load_average': system_metrics.load_average
            },
            'application': {
                'requests_total': app_metrics.requests_total,
                'response_time_avg': app_metrics.response_time_avg,
                'response_time_p95': app_metrics.response_time_p95,
                'error_rate': app_metrics.error_rate,
                'active_connections': app_metrics.active_connections,
                'cache_hit_rate': app_metrics.cache_hit_rate,
                'queue_size': app_metrics.queue_size
            },
            'counters': dict(self.counters),
            'gauges': dict(self.gauges)
        }

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_collector: MetricsCollector, metric_name: str, tags: Dict[str, str] = None):
        self.metrics_collector = metrics_collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics_collector.record_timer(self.metric_name, duration, self.tags)

class HealthChecker:
    """Health check system for monitoring component status"""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def register_check(self, name: str, check_func: Callable[[], bool], description: str = ""):
        """Register a health check function"""
        with self.lock:
            self.checks[name] = check_func
            self.check_results[name] = {
                'description': description,
                'status': 'unknown',
                'last_check': None,
                'error': None
            }
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.checks:
            return {'status': 'error', 'error': f'Check {name} not found'}
        
        try:
            start_time = time.time()
            result = self.checks[name]()
            duration = time.time() - start_time
            
            status = 'healthy' if result else 'unhealthy'
            
            with self.lock:
                self.check_results[name].update({
                    'status': status,
                    'last_check': datetime.now().isoformat(),
                    'duration': duration,
                    'error': None
                })
            
            return self.check_results[name].copy()
            
        except Exception as e:
            with self.lock:
                self.check_results[name].update({
                    'status': 'error',
                    'last_check': datetime.now().isoformat(),
                    'error': str(e)
                })
            
            return self.check_results[name].copy()
    
    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks"""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        results = self.run_all_checks()
        
        healthy_count = sum(1 for r in results.values() if r['status'] == 'healthy')
        total_count = len(results)
        
        overall_status = 'healthy'
        if healthy_count == 0:
            overall_status = 'critical'
        elif healthy_count < total_count:
            overall_status = 'degraded'
        
        return {
            'status': overall_status,
            'healthy_checks': healthy_count,
            'total_checks': total_count,
            'timestamp': datetime.now().isoformat(),
            'checks': results
        }

# Global instances
_metrics_collector = None
_health_checker = None

def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

def initialize_monitoring():
    """Initialize monitoring system with default health checks"""
    health_checker = get_health_checker()
    
    # Register basic health checks
    def check_memory():
        """Check if memory usage is below 90%"""
        memory = psutil.virtual_memory()
        return memory.percent < 90
    
    def check_disk_space():
        """Check if disk usage is below 95%"""
        disk = psutil.disk_usage('/')
        usage_percent = (disk.used / disk.total) * 100
        return usage_percent < 95
    
    def check_cpu():
        """Check if CPU usage is below 95%"""
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent < 95
    
    health_checker.register_check('memory', check_memory, 'Memory usage check')
    health_checker.register_check('disk_space', check_disk_space, 'Disk space check')
    health_checker.register_check('cpu', check_cpu, 'CPU usage check')
    
    logger.info("Monitoring system initialized")

# Convenience functions
def record_counter(name: str, value: int = 1, tags: Dict[str, str] = None):
    """Record a counter metric"""
    get_metrics_collector().record_counter(name, value, tags)

def record_gauge(name: str, value: float, tags: Dict[str, str] = None):
    """Record a gauge metric"""
    get_metrics_collector().record_gauge(name, value, tags)

def record_timer(name: str, duration: float, tags: Dict[str, str] = None):
    """Record a timer metric"""
    get_metrics_collector().record_timer(name, duration, tags)

def timer(metric_name: str, tags: Dict[str, str] = None):
    """Create a performance timer context manager"""
    return PerformanceTimer(get_metrics_collector(), metric_name, tags) 