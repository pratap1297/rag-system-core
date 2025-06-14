"""
Enhanced Logging Manager for RAG System
Provides centralized, configurable logging with structured output and monitoring integration
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import traceback
from dataclasses import dataclass, asdict

try:
    from pythonjsonlogger import jsonlogger
    HAS_JSON_LOGGER = True
except ImportError:
    HAS_JSON_LOGGER = False

@dataclass
class LogContext:
    """Structured logging context"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add custom context if available
        if hasattr(record, 'context') and self.include_context:
            if isinstance(record.context, LogContext):
                log_entry["context"] = asdict(record.context)
            elif isinstance(record.context, dict):
                log_entry["context"] = record.context
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 
                          'process', 'getMessage', 'exc_info', 'exc_text', 'stack_info',
                          'context']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)

class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for development"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Create colored format
        colored_levelname = f"{level_color}{record.levelname:8}{reset_color}"
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        # Create base message
        message = f"{timestamp} | {colored_levelname} | {record.name:20} | {record.getMessage()}"
        
        # Add context if available
        if hasattr(record, 'context'):
            if isinstance(record.context, LogContext):
                context_str = f"[{record.context.component}:{record.context.operation}]"
                message = f"{timestamp} | {colored_levelname} | {record.name:20} | {context_str} {record.getMessage()}"
        
        # Add exception information
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message

class LoggingManager:
    """Centralized logging manager with configuration support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Get configuration
        log_level = self.config.get('log_level', 'INFO')
        log_format = self.config.get('log_format', 'json')
        log_dir = Path(self.config.get('log_dir', 'logs'))
        environment = self.config.get('environment', 'development')
        
        # Create log directory
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Setup console handler
        self._setup_console_handler(environment, log_format)
        
        # Setup file handlers
        self._setup_file_handlers(log_dir, log_format)
        
        # Setup error handler
        self._setup_error_handler(log_dir)
        
        # Setup application-specific loggers
        self._setup_application_loggers()
    
    def _setup_console_handler(self, environment: str, log_format: str):
        """Setup console handler with appropriate formatting"""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if environment == 'development':
            # Use colored formatter for development
            formatter = ColoredConsoleFormatter()
        else:
            # Use structured formatter for production
            if log_format == 'json' and HAS_JSON_LOGGER:
                formatter = jsonlogger.JsonFormatter(
                    '%(asctime)s %(name)s %(levelname)s %(message)s'
                )
            else:
                formatter = StructuredFormatter()
        
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add to root logger
        logging.getLogger().addHandler(console_handler)
        self.handlers['console'] = console_handler
    
    def _setup_file_handlers(self, log_dir: Path, log_format: str):
        """Setup rotating file handlers"""
        # Main application log
        app_log_file = log_dir / "rag_system.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        if log_format == 'json':
            app_handler.setFormatter(StructuredFormatter())
        else:
            app_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
            ))
        
        app_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(app_handler)
        self.handlers['app_file'] = app_handler
        
        # API access log
        api_log_file = log_dir / "api_access.log"
        api_handler = logging.handlers.RotatingFileHandler(
            api_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
            encoding='utf-8'
        )
        api_handler.setFormatter(StructuredFormatter())
        api_handler.setLevel(logging.INFO)
        self.handlers['api_file'] = api_handler
        
        # Performance log
        perf_log_file = log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding='utf-8'
        )
        perf_handler.setFormatter(StructuredFormatter())
        perf_handler.setLevel(logging.INFO)
        self.handlers['performance'] = perf_handler
    
    def _setup_error_handler(self, log_dir: Path):
        """Setup dedicated error handler"""
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        
        logging.getLogger().addHandler(error_handler)
        self.handlers['error'] = error_handler
    
    def _setup_application_loggers(self):
        """Setup application-specific loggers"""
        # API logger
        api_logger = logging.getLogger('rag_system.api')
        api_logger.addHandler(self.handlers['api_file'])
        self.loggers['api'] = api_logger
        
        # Performance logger
        perf_logger = logging.getLogger('rag_system.performance')
        perf_logger.addHandler(self.handlers['performance'])
        self.loggers['performance'] = perf_logger
        
        # Azure services logger
        azure_logger = logging.getLogger('rag_system.azure')
        self.loggers['azure'] = azure_logger
        
        # ServiceNow logger
        servicenow_logger = logging.getLogger('rag_system.servicenow')
        self.loggers['servicenow'] = servicenow_logger
        
        # Vector database logger
        vector_logger = logging.getLogger('rag_system.vector')
        self.loggers['vector'] = vector_logger
        
        # LLM logger
        llm_logger = logging.getLogger('rag_system.llm')
        self.loggers['llm'] = llm_logger
        
        # Embedding logger
        embedding_logger = logging.getLogger('rag_system.embedding')
        self.loggers['embedding'] = embedding_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name"""
        if name in self.loggers:
            return self.loggers[name]
        
        # Create new logger
        logger = logging.getLogger(f'rag_system.{name}')
        self.loggers[name] = logger
        return logger
    
    def log_with_context(self, logger_name: str, level: str, message: str, 
                        context: Optional[Union[LogContext, Dict[str, Any]]] = None,
                        **kwargs):
        """Log message with structured context"""
        logger = self.get_logger(logger_name)
        
        # Create log record
        log_method = getattr(logger, level.lower())
        
        # Add context to extra
        extra = kwargs.copy()
        if context:
            extra['context'] = context
        
        log_method(message, extra=extra)
    
    def log_api_request(self, method: str, path: str, status_code: int, 
                       response_time: float, user_id: Optional[str] = None,
                       request_id: Optional[str] = None, **kwargs):
        """Log API request with structured data"""
        context = LogContext(
            component="api",
            operation="request",
            user_id=user_id,
            request_id=request_id,
            metadata={
                "method": method,
                "path": path,
                "status_code": status_code,
                "response_time_ms": response_time * 1000,
                **kwargs
            }
        )
        
        self.log_with_context(
            'api', 'info',
            f"{method} {path} - {status_code} ({response_time:.3f}s)",
            context
        )
    
    def log_performance(self, operation: str, duration: float, 
                       component: str, **metadata):
        """Log performance metrics"""
        context = LogContext(
            component=component,
            operation=operation,
            metadata={
                "duration_ms": duration * 1000,
                **metadata
            }
        )
        
        self.log_with_context(
            'performance', 'info',
            f"{component}.{operation} completed in {duration:.3f}s",
            context
        )
    
    def log_error(self, component: str, operation: str, error: Exception,
                  context_data: Optional[Dict[str, Any]] = None,
                  user_id: Optional[str] = None):
        """Log error with full context"""
        context = LogContext(
            component=component,
            operation=operation,
            user_id=user_id,
            metadata=context_data or {}
        )
        
        logger = self.get_logger(component)
        logger.error(
            f"Error in {component}.{operation}: {str(error)}",
            exc_info=True,
            extra={'context': context}
        )
    
    def log_azure_operation(self, service: str, operation: str, 
                           success: bool, duration: float,
                           **metadata):
        """Log Azure service operations"""
        context = LogContext(
            component="azure",
            operation=f"{service}.{operation}",
            metadata={
                "service": service,
                "success": success,
                "duration_ms": duration * 1000,
                **metadata
            }
        )
        
        level = 'info' if success else 'error'
        status = 'succeeded' if success else 'failed'
        
        self.log_with_context(
            'azure', level,
            f"Azure {service}.{operation} {status} in {duration:.3f}s",
            context
        )
    
    def log_servicenow_sync(self, operation: str, records_processed: int,
                           success: bool, duration: float, **metadata):
        """Log ServiceNow sync operations"""
        context = LogContext(
            component="servicenow",
            operation=operation,
            metadata={
                "records_processed": records_processed,
                "success": success,
                "duration_ms": duration * 1000,
                **metadata
            }
        )
        
        level = 'info' if success else 'error'
        status = 'succeeded' if success else 'failed'
        
        self.log_with_context(
            'servicenow', level,
            f"ServiceNow {operation} {status}: {records_processed} records in {duration:.3f}s",
            context
        )
    
    def log_vector_operation(self, operation: str, document_count: int,
                            success: bool, duration: float, **metadata):
        """Log vector database operations"""
        context = LogContext(
            component="vector",
            operation=operation,
            metadata={
                "document_count": document_count,
                "success": success,
                "duration_ms": duration * 1000,
                **metadata
            }
        )
        
        level = 'info' if success else 'error'
        status = 'succeeded' if success else 'failed'
        
        self.log_with_context(
            'vector', level,
            f"Vector {operation} {status}: {document_count} documents in {duration:.3f}s",
            context
        )
    
    def log_llm_request(self, provider: str, model: str, tokens_used: int,
                       success: bool, duration: float, **metadata):
        """Log LLM requests"""
        context = LogContext(
            component="llm",
            operation="request",
            metadata={
                "provider": provider,
                "model": model,
                "tokens_used": tokens_used,
                "success": success,
                "duration_ms": duration * 1000,
                **metadata
            }
        )
        
        level = 'info' if success else 'error'
        status = 'succeeded' if success else 'failed'
        
        self.log_with_context(
            'llm', level,
            f"LLM request to {provider}/{model} {status}: {tokens_used} tokens in {duration:.3f}s",
            context
        )
    
    def log_embedding_request(self, provider: str, model: str, text_count: int,
                             success: bool, duration: float, **metadata):
        """Log embedding requests"""
        context = LogContext(
            component="embedding",
            operation="request",
            metadata={
                "provider": provider,
                "model": model,
                "text_count": text_count,
                "success": success,
                "duration_ms": duration * 1000,
                **metadata
            }
        )
        
        level = 'info' if success else 'error'
        status = 'succeeded' if success else 'failed'
        
        self.log_with_context(
            'embedding', level,
            f"Embedding request to {provider}/{model} {status}: {text_count} texts in {duration:.3f}s",
            context
        )
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "handlers": len(self.handlers),
            "loggers": len(self.loggers),
            "log_files": []
        }
        
        # Get log file information
        log_dir = Path(self.config.get('log_dir', 'logs'))
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                stats["log_files"].append({
                    "name": log_file.name,
                    "size_mb": log_file.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
        
        return stats
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        log_dir = Path(self.config.get('log_dir', 'logs'))
        if not log_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        cleaned_files = []
        
        for log_file in log_dir.glob("*.log.*"):  # Rotated log files
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    cleaned_files.append(log_file.name)
                except Exception as e:
                    self.log_error("logging", "cleanup", e, {"file": str(log_file)})
        
        if cleaned_files:
            self.get_logger("system").info(f"Cleaned up {len(cleaned_files)} old log files")
        
        return cleaned_files

# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None

def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager

def setup_logging(config: Dict[str, Any]):
    """Setup global logging with configuration"""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    return _logging_manager

def get_logger(name: str) -> logging.Logger:
    """Get a logger by name"""
    return get_logging_manager().get_logger(name) 