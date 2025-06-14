#!/usr/bin/env python3
"""
RAG System Diagnostics Utility
System diagnostics and troubleshooting tools
"""

import argparse
import sys
import logging
import json
import os
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class SystemDiagnostics:
    """System diagnostics and troubleshooting"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        
    def check_environment(self) -> Dict[str, Any]:
        """Check environment configuration"""
        results = {
            "python_version": sys.version,
            "platform": sys.platform,
            "working_directory": str(Path.cwd()),
            "base_directory": str(self.base_dir),
            "environment_variables": {},
            "files": {}
        }
        
        # Check important environment variables
        env_vars = [
            "GROQ_API_KEY", "COHERE_API_KEY", "OPENAI_API_KEY",
            "PYTHONPATH", "PATH"
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                # Mask API keys for security
                if "API_KEY" in var:
                    results["environment_variables"][var] = f"{'*' * (len(value) - 4)}{value[-4:]}" if len(value) > 4 else "***"
                else:
                    results["environment_variables"][var] = value
            else:
                results["environment_variables"][var] = None
        
        # Check important files
        important_files = [
            ".env", ".env.template", "requirements.txt", "config/default.yaml",
            "src/core/config_manager.py", "src/api/main.py"
        ]
        
        for file_path in important_files:
            full_path = self.base_dir / file_path
            results["files"][file_path] = {
                "exists": full_path.exists(),
                "size": full_path.stat().st_size if full_path.exists() else 0,
                "readable": os.access(full_path, os.R_OK) if full_path.exists() else False
            }
        
        return results
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check Python dependencies"""
        results = {
            "installed_packages": {},
            "missing_packages": [],
            "version_conflicts": []
        }
        
        # Required packages from requirements.txt
        required_packages = [
            "fastapi", "uvicorn", "pydantic", "sentence-transformers",
            "cohere", "faiss-cpu", "groq", "openai", "PyPDF2",
            "python-docx", "numpy", "pandas", "gradio", "requests"
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                # Try to get version info
                try:
                    import pkg_resources
                    version = pkg_resources.get_distribution(package).version
                    results["installed_packages"][package] = version
                except:
                    results["installed_packages"][package] = "unknown version"
            except ImportError:
                results["missing_packages"].append(package)
        
        return results
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        results = {
            "cpu": {
                "count": psutil.cpu_count(),
                "usage_percent": psutil.cpu_percent(interval=1),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "used": psutil.disk_usage('/').used,
                "percent": psutil.disk_usage('/').percent
            }
        }
        
        return results
    
    def check_ports(self) -> Dict[str, Any]:
        """Check if required ports are available"""
        results = {
            "port_status": {},
            "listening_ports": []
        }
        
        # Check common RAG system ports
        ports_to_check = [8000, 7860, 7861, 7862, 7863]
        
        for port in ports_to_check:
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                sock.close()
                
                results["port_status"][port] = {
                    "available": result != 0,
                    "in_use": result == 0
                }
            except Exception as e:
                results["port_status"][port] = {
                    "available": False,
                    "error": str(e)
                }
        
        # Get all listening ports
        try:
            connections = psutil.net_connections(kind='inet')
            for conn in connections:
                if conn.status == 'LISTEN':
                    results["listening_ports"].append({
                        "port": conn.laddr.port,
                        "address": conn.laddr.ip,
                        "pid": conn.pid
                    })
        except Exception as e:
            results["listening_ports"] = f"Error: {e}"
        
        return results
    
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run complete system diagnostics"""
        print(f"{Fore.CYAN}🔍 RAG SYSTEM - FULL DIAGNOSTICS{Style.RESET_ALL}")
        print("=" * 60)
        
        diagnostics = {
            "timestamp": str(datetime.now()),
            "environment": self.check_environment(),
            "dependencies": self.check_dependencies(),
            "system_resources": self.check_system_resources(),
            "ports": self.check_ports()
        }
        
        return diagnostics
    
    def display_diagnostics(self, diagnostics: Dict[str, Any]):
        """Display diagnostics in a readable format"""
        
        # Environment Check
        print(f"\n{Fore.CYAN}🌍 ENVIRONMENT CHECK{Style.RESET_ALL}")
        print("-" * 30)
        env = diagnostics["environment"]
        print(f"Python Version: {env['python_version'].split()[0]}")
        print(f"Platform: {env['platform']}")
        print(f"Working Directory: {env['working_directory']}")
        
        # Environment Variables
        print(f"\n{Fore.CYAN}🔧 ENVIRONMENT VARIABLES{Style.RESET_ALL}")
        for var, value in env["environment_variables"].items():
            status = f"{Fore.GREEN}✅" if value else f"{Fore.RED}❌"
            print(f"{status} {var}: {value or 'Not set'}{Style.RESET_ALL}")
        
        # File Check
        print(f"\n{Fore.CYAN}📁 FILE CHECK{Style.RESET_ALL}")
        for file_path, info in env["files"].items():
            status = f"{Fore.GREEN}✅" if info["exists"] else f"{Fore.RED}❌"
            size_info = f" ({info['size']} bytes)" if info["exists"] else ""
            print(f"{status} {file_path}{size_info}{Style.RESET_ALL}")
        
        # Dependencies Check
        print(f"\n{Fore.CYAN}📦 DEPENDENCIES CHECK{Style.RESET_ALL}")
        deps = diagnostics["dependencies"]
        
        print(f"{Fore.GREEN}✅ Installed Packages:{Style.RESET_ALL}")
        for package, version in deps["installed_packages"].items():
            print(f"   • {package}: {version}")
        
        if deps["missing_packages"]:
            print(f"\n{Fore.RED}❌ Missing Packages:{Style.RESET_ALL}")
            for package in deps["missing_packages"]:
                print(f"   • {package}")
        
        # System Resources
        print(f"\n{Fore.CYAN}💻 SYSTEM RESOURCES{Style.RESET_ALL}")
        resources = diagnostics["system_resources"]
        
        print(f"CPU: {resources['cpu']['count']} cores, {resources['cpu']['usage_percent']:.1f}% usage")
        
        memory = resources["memory"]
        memory_gb = memory["total"] / (1024**3)
        memory_used_gb = memory["used"] / (1024**3)
        print(f"Memory: {memory_used_gb:.1f}GB / {memory_gb:.1f}GB ({memory['percent']:.1f}% used)")
        
        disk = resources["disk"]
        disk_gb = disk["total"] / (1024**3)
        disk_used_gb = disk["used"] / (1024**3)
        print(f"Disk: {disk_used_gb:.1f}GB / {disk_gb:.1f}GB ({disk['percent']:.1f}% used)")
        
        # Port Status
        print(f"\n{Fore.CYAN}🌐 PORT STATUS{Style.RESET_ALL}")
        ports = diagnostics["ports"]
        
        for port, status in ports["port_status"].items():
            if status.get("available"):
                print(f"{Fore.GREEN}✅ Port {port}: Available{Style.RESET_ALL}")
            elif status.get("in_use"):
                print(f"{Fore.YELLOW}⚠️  Port {port}: In use{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}❌ Port {port}: Error - {status.get('error', 'Unknown')}{Style.RESET_ALL}")

def main():
    """Main entry point for diagnostics utility"""
    parser = argparse.ArgumentParser(description="RAG System Diagnostics Utility")
    parser.add_argument("--output", choices=["console", "json"], default="console",
                       help="Output format")
    parser.add_argument("--save", help="Save diagnostics to file")
    parser.add_argument("--log-level", default="warning",
                       choices=["debug", "info", "warning", "error"],
                       help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        diagnostics_tool = SystemDiagnostics()
        results = diagnostics_tool.run_full_diagnostics()
        
        if args.output == "json":
            print(json.dumps(results, indent=2))
        else:
            diagnostics_tool.display_diagnostics(results)
        
        if args.save:
            with open(args.save, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n{Fore.GREEN}✅ Diagnostics saved to {args.save}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Diagnostics failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 