#!/usr/bin/env python3
"""
Enhanced Health Check Utility for RAG System
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.config_manager import ConfigManager
from core.logging_system import setup_logging, get_logger
from core.monitoring import get_performance_monitor

class EnhancedHealthChecker:
    """Enhanced health checker with comprehensive monitoring"""
    
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        if self.config and hasattr(self.config, 'logging'):
            setup_logging(self.config.logging.__dict__)
        
        self.logger = get_logger("health_check")
        self.monitor = get_performance_monitor()
        
        self.logger.info("Enhanced health checker initialized")
    
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
            results["checks"]["disk_space"] = self._check_disk_space()
            results["checks"]["dependencies"] = self._check_dependencies()
            
            # Calculate overall status
            results["overall_status"] = self._calculate_overall_status(results["checks"])
            results["summary"] = self._generate_summary(results["checks"])
            
            duration = time.time() - start_time
            results["duration_seconds"] = duration
            
            self.logger.info(f"Health check completed in {duration:.2f}s - Status: {results['overall_status']}")
            
            # Record metrics
            self.monitor.record_metric("health_check_duration_seconds", duration)
            
        except Exception as e:
            results["overall_status"] = "error"
            results["error"] = str(e)
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
            log_dir = Path("logs")
            
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
            
            return {
                "status": status,
                "log_directory": str(log_dir),
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
            self.monitor.record_metric("health_check_test", 1.0)
            
            system_metrics = self.monitor.get_system_metrics()
            health_status = self.monitor.get_health_status()
            
            return {
                "status": "healthy",
                "metrics_count": len(self.monitor.metrics),
                "system_metrics": system_metrics,
                "health_status": health_status,
                "issues": []
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check monitoring system: {e}"]
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
                "issues": issues
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "issues": [f"Failed to check disk space: {e}"]
            }
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies"""
        
        try:
            required_packages = [
                "fastapi", "uvicorn", "pydantic", "sentence_transformers",
                "faiss", "groq", "requests", "psutil", "yaml"
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
            "all_issues": all_issues
        }

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Enhanced RAG System Health Check")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    try:
        # Initialize health checker
        health_checker = EnhancedHealthChecker(args.config)
        
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