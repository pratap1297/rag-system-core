#!/usr/bin/env python3
"""
Enhanced Health Check Utility for RAG System
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_manager import ConfigManager
from core.logging_system import setup_logging, get_logger
from core.monitoring import get_performance_monitor

class EnhancedHealthChecker:
    """Enhanced health checker"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        
        if self.config and hasattr(self.config, 'logging'):
            setup_logging(self.config.logging.__dict__)
        
        self.logger = get_logger("health_check")
        self.monitor = get_performance_monitor()
    
    def run_check(self):
        """Run comprehensive health check"""
        
        start_time = time.time()
        results = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "checks": {},
            "duration_seconds": 0
        }
        
        try:
            # System checks
            results["checks"]["system"] = self._check_system()
            results["checks"]["config"] = self._check_config()
            results["checks"]["logging"] = self._check_logging()
            results["checks"]["monitoring"] = self._check_monitoring()
            
            # Calculate overall status
            results["overall_status"] = self._calculate_status(results["checks"])
            results["duration_seconds"] = time.time() - start_time
            
            self.logger.info(f"Health check completed - Status: {results['overall_status']}")
            
        except Exception as e:
            results["overall_status"] = "error"
            results["error"] = str(e)
            results["duration_seconds"] = time.time() - start_time
        
        return results
    
    def _check_system(self):
        """Check system resources"""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if cpu > 90 or memory.percent > 90:
                status = "critical"
            elif cpu > 70 or memory.percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "cpu_percent": cpu,
                "memory_percent": memory.percent
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_config(self):
        """Check configuration"""
        try:
            if not self.config:
                return {"status": "critical", "message": "No configuration loaded"}
            
            return {
                "status": "healthy",
                "environment": self.config.environment
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_logging(self):
        """Check logging system"""
        try:
            log_dir = Path("logs")
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            
            return {"status": "healthy", "log_dir": str(log_dir)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _check_monitoring(self):
        """Check monitoring system"""
        try:
            self.monitor.record_metric("health_test", 1.0)
            return {"status": "healthy", "metrics_count": len(self.monitor.metrics)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _calculate_status(self, checks):
        """Calculate overall status"""
        statuses = [check.get("status", "unknown") for check in checks.values()]
        
        if "critical" in statuses or "error" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        else:
            return "healthy"

def main():
    """Main function"""
    try:
        checker = EnhancedHealthChecker()
        results = checker.run_check()
        
        print(json.dumps(results, indent=2, default=str))
        
        if results["overall_status"] in ["critical", "error"]:
            sys.exit(1)
        elif results["overall_status"] == "warning":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Health check failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 