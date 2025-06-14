"""
Web Interface for RAG System
Provides REST API endpoints for system monitoring, configuration, and management
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create mock classes for when FastAPI is not available
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
        def get(self, *args, **kwargs): return lambda f: f
        def post(self, *args, **kwargs): return lambda f: f
        def put(self, *args, **kwargs): return lambda f: f
        def delete(self, *args, **kwargs): return lambda f: f
        def add_middleware(self, *args, **kwargs): pass
        def mount(self, *args, **kwargs): pass
    
    class BaseModel:
        pass
    
    class HTTPException(Exception):
        def __init__(self, status_code, detail): pass
    
    JSONResponse = dict
    HTMLResponse = str

logger = logging.getLogger(__name__)

# Request/Response Models
class SystemStatusResponse(BaseModel):
    """System status response model"""
    status: str
    timestamp: str
    uptime: float
    version: str
    components: Dict[str, str]

class MetricsResponse(BaseModel):
    """Metrics response model"""
    timestamp: str
    system: Dict[str, Any]
    application: Dict[str, Any]
    counters: Dict[str, int]
    gauges: Dict[str, float]

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    healthy_checks: int
    total_checks: int
    timestamp: str
    checks: Dict[str, Dict[str, Any]]

class ConfigUpdateRequest(BaseModel):
    """Configuration update request model"""
    component: str
    updates: Dict[str, Any]

class ScannerCommandRequest(BaseModel):
    """Scanner command request model"""
    command: str
    parameters: Optional[Dict[str, Any]] = None

class WebInterface:
    """Web interface for RAG system management"""
    
    def __init__(self, config_manager=None, dependency_container=None):
        self.config_manager = config_manager
        self.dependency_container = dependency_container
        self.app = FastAPI(
            title="RAG System API",
            description="Enterprise RAG System Management Interface",
            version="1.0.0"
        ) if FASTAPI_AVAILABLE else None
        
        if self.app and FASTAPI_AVAILABLE:
            self._setup_middleware()
            self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        if not FASTAPI_AVAILABLE:
            return
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        if not FASTAPI_AVAILABLE:
            return
        
        # System status endpoints
        @self.app.get("/api/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get system status"""
            try:
                from ..monitoring import get_health_checker
                
                health_checker = get_health_checker()
                health_status = health_checker.get_overall_health()
                
                return SystemStatusResponse(
                    status=health_status['status'],
                    timestamp=datetime.now().isoformat(),
                    uptime=0.0,  # TODO: Implement uptime tracking
                    version="1.0.0",
                    components={
                        "api": "running",
                        "monitoring": "running",
                        "storage": "running"
                    }
                )
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/health", response_model=HealthCheckResponse)
        async def get_health_status():
            """Get detailed health check status"""
            try:
                from ..monitoring import get_health_checker
                
                health_checker = get_health_checker()
                health_status = health_checker.get_overall_health()
                
                return HealthCheckResponse(**health_status)
            except Exception as e:
                logger.error(f"Error getting health status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Get system metrics"""
            try:
                from ..monitoring import get_metrics_collector
                
                metrics_collector = get_metrics_collector()
                metrics = metrics_collector.get_all_metrics()
                
                return MetricsResponse(**metrics)
            except Exception as e:
                logger.error(f"Error getting metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Configuration endpoints
        @self.app.get("/api/config")
        async def get_configuration():
            """Get system configuration"""
            try:
                if not self.config_manager:
                    raise HTTPException(status_code=503, detail="Configuration manager not available")
                
                config = self.config_manager.get_config()
                # Convert to dict for JSON serialization
                from dataclasses import asdict
                return asdict(config)
            except Exception as e:
                logger.error(f"Error getting configuration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/config")
        async def update_configuration(request: ConfigUpdateRequest):
            """Update system configuration"""
            try:
                if not self.config_manager:
                    raise HTTPException(status_code=503, detail="Configuration manager not available")
                
                self.config_manager.update_config(request.component, request.updates)
                return {"status": "success", "message": "Configuration updated"}
            except Exception as e:
                logger.error(f"Error updating configuration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Scanner management endpoints
        @self.app.get("/api/scanner/status")
        async def get_scanner_status():
            """Get folder scanner status"""
            try:
                # TODO: Implement scanner status retrieval
                return {
                    "status": "running",
                    "monitored_directories": [],
                    "files_tracked": 0,
                    "queue_size": 0
                }
            except Exception as e:
                logger.error(f"Error getting scanner status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/scanner/command")
        async def execute_scanner_command(request: ScannerCommandRequest):
            """Execute scanner command"""
            try:
                # TODO: Implement scanner command execution
                return {
                    "status": "success",
                    "command": request.command,
                    "result": "Command executed successfully"
                }
            except Exception as e:
                logger.error(f"Error executing scanner command: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Document processing endpoints
        @self.app.get("/api/documents")
        async def list_documents():
            """List processed documents"""
            try:
                # TODO: Implement document listing
                return {
                    "documents": [],
                    "total": 0,
                    "page": 1,
                    "per_page": 50
                }
            except Exception as e:
                logger.error(f"Error listing documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/search")
        async def search_documents(query: str, limit: int = 10):
            """Search documents"""
            try:
                # TODO: Implement document search
                return {
                    "query": query,
                    "results": [],
                    "total": 0,
                    "limit": limit
                }
            except Exception as e:
                logger.error(f"Error searching documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Static file serving for web UI
        @self.app.get("/", response_class=HTMLResponse)
        async def serve_index():
            """Serve main web interface"""
            return self._get_dashboard_html()
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def serve_dashboard():
            """Serve dashboard page"""
            return self._get_dashboard_html()
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>RAG System Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .card { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
                .status-item { text-align: center; }
                .status-value { font-size: 2em; font-weight: bold; color: #27ae60; }
                .status-label { color: #7f8c8d; margin-top: 5px; }
                .btn { background: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
                .btn:hover { background: #2980b9; }
                .error { color: #e74c3c; }
                .success { color: #27ae60; }
                .warning { color: #f39c12; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 RAG System Dashboard</h1>
                    <p>Enterprise Document Processing and Retrieval System</p>
                </div>
                
                <div class="card">
                    <h2>System Status</h2>
                    <div class="status-grid">
                        <div class="status-item">
                            <div class="status-value" id="system-status">Loading...</div>
                            <div class="status-label">System Health</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value" id="documents-count">-</div>
                            <div class="status-label">Documents Processed</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value" id="queue-size">-</div>
                            <div class="status-label">Processing Queue</div>
                        </div>
                        <div class="status-item">
                            <div class="status-value" id="memory-usage">-</div>
                            <div class="status-label">Memory Usage</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Quick Actions</h2>
                    <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                    <button class="btn" onclick="viewLogs()">View Logs</button>
                    <button class="btn" onclick="runHealthCheck()">Health Check</button>
                    <button class="btn" onclick="viewMetrics()">View Metrics</button>
                </div>
                
                <div class="card">
                    <h2>Recent Activity</h2>
                    <div id="activity-log">
                        <p>Loading recent activity...</p>
                    </div>
                </div>
            </div>
            
            <script>
                async function fetchStatus() {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        
                        document.getElementById('system-status').textContent = data.status;
                        document.getElementById('system-status').className = 'status-value ' + 
                            (data.status === 'healthy' ? 'success' : 'error');
                    } catch (error) {
                        document.getElementById('system-status').textContent = 'Error';
                        document.getElementById('system-status').className = 'status-value error';
                    }
                }
                
                async function fetchMetrics() {
                    try {
                        const response = await fetch('/api/metrics');
                        const data = await response.json();
                        
                        document.getElementById('memory-usage').textContent = 
                            Math.round(data.system.memory_percent) + '%';
                        document.getElementById('queue-size').textContent = data.application.queue_size;
                    } catch (error) {
                        console.error('Error fetching metrics:', error);
                    }
                }
                
                function refreshStatus() {
                    fetchStatus();
                    fetchMetrics();
                }
                
                function viewLogs() {
                    alert('Log viewer not implemented yet');
                }
                
                function runHealthCheck() {
                    fetch('/api/health')
                        .then(response => response.json())
                        .then(data => {
                            alert(`Health Check: ${data.status}\\n${data.healthy_checks}/${data.total_checks} checks passing`);
                        })
                        .catch(error => alert('Health check failed: ' + error));
                }
                
                function viewMetrics() {
                    window.open('/api/metrics', '_blank');
                }
                
                // Initialize dashboard
                refreshStatus();
                setInterval(refreshStatus, 30000); // Refresh every 30 seconds
            </script>
        </body>
        </html>
        """
    
    def get_app(self):
        """Get FastAPI application instance"""
        return self.app

def create_web_interface(config_manager=None, dependency_container=None) -> WebInterface:
    """Create web interface instance"""
    return WebInterface(config_manager, dependency_container)

# Fallback functions when FastAPI is not available
def create_simple_status_server(port: int = 8080):
    """Create a simple HTTP status server using built-in modules"""
    import http.server
    import socketserver
    import json
    from urllib.parse import urlparse, parse_qs
    
    class StatusHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                status = {
                    'status': 'running',
                    'timestamp': datetime.now().isoformat(),
                    'message': 'RAG System is operational'
                }
                
                self.wfile.write(json.dumps(status).encode())
            
            elif parsed_path.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                html = """
                <html>
                <head><title>RAG System</title></head>
                <body>
                    <h1>RAG System Status</h1>
                    <p>System is running</p>
                    <p><a href="/status">JSON Status</a></p>
                </body>
                </html>
                """
                
                self.wfile.write(html.encode())
            
            else:
                self.send_response(404)
                self.end_headers()
        
        def log_message(self, format, *args):
            # Suppress default logging
            pass
    
    return StatusHandler, port 