#!/usr/bin/env python3
"""
Enhanced Health Check Utility for RAG System
Comprehensive system health monitoring with logging and error handling
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_manager import ConfigManager
from core.logging_system import setup_logging, get_logger
from core.monitoring import get_performance_monitor
from core.exceptions import RAGSystemError, ErrorContext
from core.error_handler import get_error_handler

class HealthChecker:
    """Enhanced health checker with comprehensive monitoring"""
    
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        if self.config and hasattr(self.config, 'logging'):
            setup_logging(self.config.logging.__dict__)
        
        self.logger = get_logger("health_check")
        self.monitor = get_performance_monitor()
        self.error_handler = get_error_handler()
        
        self.logger.info("Health checker initialized")
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        
        start_time = time.time()
        results = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "checks": {},
            "summary": {},
            "duration_seconds": 0
        }
        
        try:
            self.logger.info("Starting comprehensive health check")
            
            # Core system checks
            results["checks"]["system_resources"] = self._check_system_resources()
            results["checks"]["configuration"] = self._check_configuration()
            results["checks"]["logging_system"] = self._check_logging_system()
            results["checks"]["monitoring_system"] = self._check_monitoring_system()
            
            # Service checks
            results["checks"]["azure_services"] = self._check_azure_services()
            results["checks"]["servicenow"] = self._check_servicenow()
            results["checks"]["vector_database"] = self._check_vector_database()
            
            # Infrastructure checks
            results["checks"]["disk_space"] = self._check_disk_space()
            results["checks"]["network_connectivity"] = self._check_network_connectivity()
            results["checks"]["dependencies"] = self._check_dependencies()
            
            # Calculate overall status
            results["overall_status"] = self._calculate_overall_status(results["checks"])
            results["summary"] = self._generate_summary(results["checks"])
            
            duration = time.time() - start_time
            results["duration_seconds"] = duration
            
            self.logger.info(f"Health check completed in {duration:.2f}s - Status: {results['overall_status']}")
            
            # Record metrics
            self.monitor.record_metric("health_check_duration_seconds", duration)
            self.monitor.record_metric("health_check_status", 1 if results["overall_status"] == "healthy" else 0)
            
        except Exception as e:
            error_context = ErrorContext(component="health_check", operation="comprehensive_check")
            rag_error = self.error_handler.handle_error(e, error_context)
            
            results["overall_status"] = "error"
            results["error"] = str(rag_error)
            results["duration_seconds"] = time.time() - start_time
        
        return results
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                status = "critical"
                issues.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                status = "warning"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                status = "critical"
                issues.append(f"Critical memory usage: {memory.percent:.1f}%")
            elif memory.percent > 80:
                status = "warning" if status == "healthy" else status
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check system resources: {e}"]
            }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration status"""
        
        try:
            if not self.config:
                return {
                    "status": "critical",
                    "issues": ["Configuration not loaded"]
                }
            
            issues = []
            status = "healthy"
            
            # Check Azure configuration
            if not self.config.azure.chat_api_key:
                issues.append("Azure chat API key not configured")
                status = "warning"
            
            if not self.config.azure.embeddings_key:
                issues.append("Azure embeddings key not configured")
                status = "warning"
            
            # Check ServiceNow configuration
            if self.config.servicenow.enabled and not self.config.servicenow.instance:
                issues.append("ServiceNow enabled but instance not configured")
                status = "warning"
            
            return {
                "status": status,
                "environment": self.config.environment,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check configuration: {e}"]
            }
    
    def _check_logging_system(self) -> Dict[str, Any]:
        """Check logging system status"""
        
        try:
            log_dir = Path(self.config.logging.log_dir if self.config and hasattr(self.config, 'logging') else "logs")
            
            issues = []
            status = "healthy"
            
            # Check log directory
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)
                issues.append("Created missing log directory")
            
            # Test log writing
            test_file = log_dir / ".health_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except Exception:
                issues.append("Cannot write to log directory")
                status = "critical"
            
            # Check log file sizes
            log_files = list(log_dir.glob("*.log"))
            large_files = [f for f in log_files if f.stat().st_size > 100 * 1024 * 1024]  # 100MB
            
            if large_files:
                issues.append(f"Large log files detected: {len(large_files)} files > 100MB")
                status = "warning"
            
            return {
                "status": status,
                "log_directory": str(log_dir),
                "log_files_count": len(log_files),
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check logging system: {e}"]
            }
    
    def _check_monitoring_system(self) -> Dict[str, Any]:
        """Check monitoring system status"""
        
        try:
            # Test monitoring system
            test_metric_name = "health_check_test"
            self.monitor.record_metric(test_metric_name, 1.0)
            
            # Check if metric was recorded
            if test_metric_name in self.monitor.metrics:
                status = "healthy"
                issues = []
            else:
                status = "warning"
                issues = ["Monitoring system not recording metrics properly"]
            
            # Get system metrics
            system_metrics = self.monitor.get_system_metrics()
            health_status = self.monitor.get_health_status()
            
            return {
                "status": status,
                "metrics_count": len(self.monitor.metrics),
                "system_metrics": system_metrics,
                "health_status": health_status,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check monitoring system: {e}"]
            }
    
    def _check_azure_services(self) -> Dict[str, Any]:
        """Check Azure services connectivity"""
        
        try:
            if not self.config or not self.config.azure.chat_api_key:
                return {
                    "status": "skipped",
                    "reason": "Azure not configured",
                    "issues": []
                }
            
            # This would normally test actual Azure connectivity
            # For now, just check configuration
            issues = []
            status = "healthy"
            
            if not self.config.azure.chat_endpoint:
                issues.append("Azure chat endpoint not configured")
                status = "warning"
            
            if not self.config.azure.embeddings_endpoint:
                issues.append("Azure embeddings endpoint not configured")
                status = "warning"
            
            return {
                "status": status,
                "chat_model": self.config.azure.chat_model,
                "embedding_model": self.config.azure.embedding_model,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check Azure services: {e}"]
            }
    
    def _check_servicenow(self) -> Dict[str, Any]:
        """Check ServiceNow connectivity"""
        
        try:
            if not self.config or not self.config.servicenow.enabled:
                return {
                    "status": "skipped",
                    "reason": "ServiceNow not enabled",
                    "issues": []
                }
            
            issues = []
            status = "healthy"
            
            if not self.config.servicenow.instance:
                issues.append("ServiceNow instance not configured")
                status = "critical"
            
            if not self.config.servicenow.username:
                issues.append("ServiceNow username not configured")
                status = "critical"
            
            return {
                "status": status,
                "instance": self.config.servicenow.instance,
                "table": self.config.servicenow.table,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check ServiceNow: {e}"]
            }
    
    def _check_vector_database(self) -> Dict[str, Any]:
        """Check vector database status"""
        
        try:
            # Check if vector database files exist
            data_dir = Path(self.config.data_dir if self.config else "data")
            vector_files = list(data_dir.glob("*.faiss")) if data_dir.exists() else []
            
            status = "healthy" if vector_files else "warning"
            issues = [] if vector_files else ["No vector database files found"]
            
            return {
                "status": status,
                "data_directory": str(data_dir),
                "vector_files_count": len(vector_files),
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check vector database: {e}"]
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        
        try:
            import psutil
            
            disk = psutil.disk_usage('/')
            
            status = "healthy"
            issues = []
            
            if disk.percent > 95:
                status = "critical"
                issues.append(f"Critical disk usage: {disk.percent:.1f}%")
            elif disk.percent > 85:
                status = "warning"
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            return {
                "status": status,
                "disk_percent": disk.percent,
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3),
                "total_gb": disk.total / (1024**3),
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check disk space: {e}"]
            }
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        
        try:
            import socket
            
            # Test basic internet connectivity
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=5)
                internet_status = "healthy"
                internet_issues = []
            except Exception:
                internet_status = "warning"
                internet_issues = ["No internet connectivity"]
            
            return {
                "status": internet_status,
                "internet_connectivity": internet_status == "healthy",
                "issues": internet_issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check network connectivity: {e}"]
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies"""
        
        try:
            required_packages = [
                "fastapi", "uvicorn", "pydantic", "sentence-transformers",
                "faiss-cpu", "groq", "requests", "psutil", "PyYAML"
            ]
            
            missing_packages = []
            
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    missing_packages.append(package)
            
            status = "healthy" if not missing_packages else "critical"
            issues = [f"Missing packages: {', '.join(missing_packages)}"] if missing_packages else []
            
            return {
                "status": status,
                "required_packages": len(required_packages),
                "missing_packages": missing_packages,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check dependencies: {e}"]
            }
    
    def _calculate_overall_status(self, checks: Dict[str, Any]) -> str:
        """Calculate overall system status"""
        
        critical_count = sum(1 for check in checks.values() if check.get("status") == "critical")
        error_count = sum(1 for check in checks.values() if check.get("status") == "error")
        warning_count = sum(1 for check in checks.values() if check.get("status") == "warning")
        
        if critical_count > 0 or error_count > 0:
            return "critical"
        elif warning_count > 0:
            return "warning"
        else:
            return "healthy"
    
    def _generate_summary(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of health check results"""
        
        total_checks = len(checks)
        healthy_count = sum(1 for check in checks.values() if check.get("status") == "healthy")
        warning_count = sum(1 for check in checks.values() if check.get("status") == "warning")
        critical_count = sum(1 for check in checks.values() if check.get("status") == "critical")
        error_count = sum(1 for check in checks.values() if check.get("status") == "error")
        skipped_count = sum(1 for check in checks.values() if check.get("status") == "skipped")
        
        all_issues = []
        for check_name, check_result in checks.items():
            if "issues" in check_result and check_result["issues"]:
                for issue in check_result["issues"]:
                    all_issues.append(f"{check_name}: {issue}")
        
        return {
            "total_checks": total_checks,
            "healthy": healthy_count,
            "warning": warning_count,
            "critical": critical_count,
            "error": error_count,
            "skipped": skipped_count,
            "all_issues": all_issues
        }

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="RAG System Health Check")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize health checker
        health_checker = HealthChecker(args.config)
        
        # Run comprehensive check
        results = health_checker.run_comprehensive_check()
        
        # Output results
        if args.output == "json":
            print(json.dumps(results, indent=2, default=str))
        else:
            print_text_results(results, args.verbose)
        
        # Exit with appropriate code
        if results["overall_status"] in ["critical", "error"]:
            sys.exit(1)
        elif results["overall_status"] == "warning":
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Health check failed: {e}", file=sys.stderr)
        sys.exit(1)

def print_text_results(results: Dict[str, Any], verbose: bool = False):
    """Print results in text format"""
    
    # Overall status
    status_emoji = {
        "healthy": "✅",
        "warning": "⚠️",
        "critical": "❌",
        "error": "💥"
    }
    
    print(f"\n{status_emoji.get(results['overall_status'], '❓')} Overall Status: {results['overall_status'].upper()}")
    print(f"⏱️  Duration: {results['duration_seconds']:.2f} seconds")
    
    # Summary
    if "summary" in results:
        summary = results["summary"]
        print(f"\n📊 Summary:")
        print(f"   Total Checks: {summary['total_checks']}")
        print(f"   ✅ Healthy: {summary['healthy']}")
        print(f"   ⚠️  Warning: {summary['warning']}")
        print(f"   ❌ Critical: {summary['critical']}")
        print(f"   💥 Error: {summary['error']}")
        print(f"   ⏭️  Skipped: {summary['skipped']}")
    
    # Issues
    if "summary" in results and results["summary"]["all_issues"]:
        print(f"\n🚨 Issues Found:")
        for issue in results["summary"]["all_issues"]:
            print(f"   • {issue}")
    
    # Detailed results
    if verbose and "checks" in results:
        print(f"\n📋 Detailed Results:")
        for check_name, check_result in results["checks"].items():
            status = check_result.get("status", "unknown")
            emoji = status_emoji.get(status, "❓")
            print(f"   {emoji} {check_name}: {status}")
            
            if check_result.get("issues"):
                for issue in check_result["issues"]:
                    print(f"      • {issue}")

if __name__ == "__main__":
    main() 