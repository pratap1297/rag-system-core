#!/usr/bin/env python3
"""
Command-line health check utility for RAG system
Usage: python health_check_cli.py [options]
"""

import argparse
import json
import sys
from datetime import datetime
from typing import Dict, Any
import requests
from tabulate import tabulate
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored output
colorama.init()

class HealthCheckCLI:
    """Command-line interface for health checking"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status"""
        colors = {
            'healthy': Fore.GREEN,
            'warning': Fore.YELLOW,
            'critical': Fore.RED,
            'unknown': Fore.MAGENTA
        }
        return colors.get(status.lower(), Fore.WHITE)
    
    def _format_uptime(self, seconds: int) -> str:
        """Format uptime in human readable format"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    
    def check_basic_health(self) -> Dict[str, Any]:
        """Check basic system health"""
        try:
            response = requests.get(f"{self.base_url}/health/summary", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def check_comprehensive_health(self) -> Dict[str, Any]:
        """Check comprehensive system health"""
        try:
            response = requests.get(
                f"{self.base_url}/heartbeat", 
                headers=self.headers, 
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def check_component_health(self) -> Dict[str, Any]:
        """Check individual component health"""
        try:
            response = requests.get(
                f"{self.base_url}/health/components", 
                headers=self.headers, 
                timeout=20
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            response = requests.get(
                f"{self.base_url}/health/performance", 
                headers=self.headers, 
                timeout=15
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def display_basic_health(self):
        """Display basic health check results"""
        print(f"\n{Fore.CYAN}🔍 RAG SYSTEM - BASIC HEALTH CHECK{Style.RESET_ALL}")
        print("=" * 50)
        
        health = self.check_basic_health()
        
        if "error" in health:
            print(f"{Fore.RED}❌ Health check failed: {health['error']}{Style.RESET_ALL}")
            return False
        
        status = health.get('overall_status', 'unknown')
        status_color = self._get_status_color(status)
        
        print(f"Overall Status: {status_color}{status.upper()}{Style.RESET_ALL}")
        print(f"Uptime: {self._format_uptime(health.get('uptime_seconds', 0))}")
        print(f"Last Check: {health.get('timestamp', 'unknown')}")
        
        # Component summary
        if 'component_count' in health:
            print(f"\nComponent Summary:")
            print(f"  {Fore.GREEN}✅ Healthy: {health.get('healthy_components', 0)}{Style.RESET_ALL}")
            print(f"  {Fore.YELLOW}⚠️  Warning: {health.get('warning_components', 0)}{Style.RESET_ALL}")
            print(f"  {Fore.RED}🚨 Critical: {health.get('critical_components', 0)}{Style.RESET_ALL}")
        
        # Alerts
        alerts = health.get('alerts', [])
        if alerts:
            print(f"\n{Fore.YELLOW}⚠️ ALERTS:{Style.RESET_ALL}")
            for alert in alerts:
                print(f"  • {alert}")
        
        return status == 'healthy'
    
    def display_comprehensive_health(self):
        """Display comprehensive health check results"""
        print(f"\n{Fore.CYAN}🔍 RAG SYSTEM - COMPREHENSIVE HEALTH CHECK{Style.RESET_ALL}")
        print("=" * 60)
        
        health = self.check_comprehensive_health()
        
        if "error" in health:
            print(f"{Fore.RED}❌ Health check failed: {health['error']}{Style.RESET_ALL}")
            return False
        
        # Overall status
        status = health.get('overall_status', 'unknown')
        status_color = self._get_status_color(status)
        print(f"Overall Status: {status_color}{status.upper()}{Style.RESET_ALL}")
        print(f"Timestamp: {health.get('timestamp', 'unknown')}")
        print(f"Uptime: {self._format_uptime(health.get('uptime_seconds', 0))}")
        
        # Component details
        components = health.get('components', [])
        if components:
            print(f"\n{Fore.CYAN}📋 COMPONENT HEALTH:{Style.RESET_ALL}")
            
            table_data = []
            for comp in components:
                comp_status = comp.get('status', 'unknown')
                comp_color = self._get_status_color(comp_status)
                
                table_data.append([
                    comp.get('name', 'Unknown'),
                    f"{comp_color}{comp_status.upper()}{Style.RESET_ALL}",
                    f"{comp.get('response_time_ms', 0):.0f}ms",
                    comp.get('error_message', 'None')[:50] if comp.get('error_message') else 'None'
                ])
            
            print(tabulate(
                table_data,
                headers=['Component', 'Status', 'Response Time', 'Error'],
                tablefmt='grid'
            ))
        
        # Performance metrics
        perf_metrics = health.get('performance_metrics', {})
        if perf_metrics:
            print(f"\n{Fore.CYAN}📊 PERFORMANCE METRICS:{Style.RESET_ALL}")
            
            perf_data = [
                ['CPU Usage', f"{perf_metrics.get('system_cpu_percent', 0)}%"],
                ['Memory Usage', f"{perf_metrics.get('system_memory_percent', 0)}%"],
                ['Disk Usage', f"{perf_metrics.get('system_disk_percent', 0)}%"],
                ['Total Vectors', perf_metrics.get('total_vectors', 0)],
                ['Uptime Hours', perf_metrics.get('uptime_hours', 0)],
                ['Groq API Key', '✅' if perf_metrics.get('api_key_groq_configured') else '❌'],
                ['Cohere API Key', '✅' if perf_metrics.get('api_key_cohere_configured') else '❌']
            ]
            
            print(tabulate(
                perf_data,
                headers=['Metric', 'Value'],
                tablefmt='simple'
            ))
        
        # Alerts
        alerts = health.get('alerts', [])
        if alerts:
            print(f"\n{Fore.YELLOW}⚠️ SYSTEM ALERTS:{Style.RESET_ALL}")
            for i, alert in enumerate(alerts, 1):
                print(f"  {i}. {alert}")
        
        return status == 'healthy'
    
    def display_component_details(self):
        """Display detailed component information"""
        print(f"\n{Fore.CYAN}🔧 RAG SYSTEM - COMPONENT DETAILS{Style.RESET_ALL}")
        print("=" * 60)
        
        comp_health = self.check_component_health()
        
        if "error" in comp_health:
            print(f"{Fore.RED}❌ Component check failed: {comp_health['error']}{Style.RESET_ALL}")
            return False
        
        components = comp_health.get('components', [])
        
        for comp in components:
            name = comp.get('name', 'Unknown')
            status = comp.get('status', 'unknown')
            status_color = self._get_status_color(status)
            
            print(f"\n{Fore.BLUE}🔹 {name}{Style.RESET_ALL}")
            print(f"   Status: {status_color}{status.upper()}{Style.RESET_ALL}")
            print(f"   Response Time: {comp.get('response_time_ms', 0):.0f}ms")
            print(f"   Last Check: {comp.get('last_check', 'unknown')}")
            
            if comp.get('error_message'):
                print(f"   {Fore.RED}Error: {comp['error_message']}{Style.RESET_ALL}")
            
            # Display key details
            details = comp.get('details', {})
            if details:
                print(f"   Details:")
                for key, value in list(details.items())[:5]:  # Show first 5 details
                    if isinstance(value, (dict, list)):
                        print(f"     {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"     {key}: {value}")
        
        return True
    
    def display_performance_details(self):
        """Display performance metrics details"""
        print(f"\n{Fore.CYAN}📊 RAG SYSTEM - PERFORMANCE METRICS{Style.RESET_ALL}")
        print("=" * 50)
        
        metrics = self.get_performance_metrics()
        
        if "error" in metrics:
            print(f"{Fore.RED}❌ Performance check failed: {metrics['error']}{Style.RESET_ALL}")
            return False
        
        # System metrics
        print(f"\n{Fore.BLUE}🖥️ System Resources:{Style.RESET_ALL}")
        system_data = [
            ['CPU Usage', f"{metrics.get('system_cpu_percent', 0)}%"],
            ['Memory Usage', f"{metrics.get('system_memory_percent', 0)}%"],
            ['Disk Usage', f"{metrics.get('system_disk_percent', 0)}%"],
            ['Uptime', f"{metrics.get('uptime_hours', 0):.1f} hours"]
        ]
        print(tabulate(system_data, headers=['Resource', 'Usage'], tablefmt='simple'))
        
        # RAG system metrics
        print(f"\n{Fore.BLUE}🤖 RAG System Metrics:{Style.RESET_ALL}")
        rag_data = [
            ['Total Vectors', metrics.get('total_vectors', 0)],
            ['Groq API Key', '✅ Configured' if metrics.get('api_key_groq_configured') else '❌ Missing'],
            ['Cohere API Key', '✅ Configured' if metrics.get('api_key_cohere_configured') else '❌ Missing']
        ]
        print(tabulate(rag_data, headers=['Metric', 'Value'], tablefmt='simple'))
        
        # Vector store metrics
        vector_metrics = metrics.get('vector_store_metrics', {})
        if vector_metrics:
            print(f"\n{Fore.BLUE}🗄️ Vector Store Metrics:{Style.RESET_ALL}")
            vector_data = []
            for key, value in vector_metrics.items():
                if isinstance(value, (int, float, str)):
                    vector_data.append([key.replace('_', ' ').title(), value])
            
            if vector_data:
                print(tabulate(vector_data, headers=['Metric', 'Value'], tablefmt='simple'))
        
        return True
    
    def continuous_monitor(self, interval: int = 30):
        """Run continuous health monitoring"""
        print(f"\n{Fore.CYAN}📡 CONTINUOUS HEALTH MONITORING{Style.RESET_ALL}")
        print(f"Checking every {interval} seconds. Press Ctrl+C to stop.")
        print("=" * 50)
        
        try:
            while True:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Checking system health...")
                
                health = self.check_basic_health()
                
                if "error" in health:
                    print(f"{Fore.RED}❌ {health['error']}{Style.RESET_ALL}")
                else:
                    status = health.get('overall_status', 'unknown')
                    status_color = self._get_status_color(status)
                    print(f"Status: {status_color}{status.upper()}{Style.RESET_ALL}")
                    
                    if status != 'healthy':
                        alerts = health.get('alerts', [])
                        for alert in alerts:
                            print(f"  ⚠️ {alert}")
                
                import time
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}🛑 Monitoring stopped by user{Style.RESET_ALL}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="RAG System Health Check CLI")
    parser.add_argument('--url', default='http://localhost:8000', help='RAG system base URL')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--mode', choices=['basic', 'comprehensive', 'components', 'performance', 'monitor'], 
                       default='basic', help='Health check mode')
    parser.add_argument('--interval', type=int, default=30, help='Monitor interval in seconds')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = HealthCheckCLI(args.url, args.api_key)
    
    # Run health check based on mode
    try:
        if args.mode == 'basic':
            if args.json:
                result = cli.check_basic_health()
                print(json.dumps(result, indent=2))
            else:
                success = cli.display_basic_health()
                sys.exit(0 if success else 1)
                
        elif args.mode == 'comprehensive':
            if args.json:
                result = cli.check_comprehensive_health()
                print(json.dumps(result, indent=2))
            else:
                success = cli.display_comprehensive_health()
                sys.exit(0 if success else 1)
                
        elif args.mode == 'components':
            if args.json:
                result = cli.check_component_health()
                print(json.dumps(result, indent=2))
            else:
                success = cli.display_component_details()
                sys.exit(0 if success else 1)
                
        elif args.mode == 'performance':
            if args.json:
                result = cli.get_performance_metrics()
                print(json.dumps(result, indent=2))
            else:
                success = cli.display_performance_details()
                sys.exit(0 if success else 1)
                
        elif args.mode == 'monitor':
            cli.continuous_monitor(args.interval)
            
    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 