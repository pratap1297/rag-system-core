"""
Enhanced Logging System for RAG System
Provides centralized, configurable logging with structured output
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class LogContext:
    """Structured logging context"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1])
            }
        
        if hasattr(record, 'context'):
            if isinstance(record.context, LogContext):
                log_entry["context"] = asdict(record.context)
            elif isinstance(record.context, dict):
                log_entry["context"] = record.context
        
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
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        colored_levelname = f"{level_color}{record.levelname:8}{reset_color}"
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        
        message = f"{timestamp} | {colored_levelname} | {record.name:20} | {record.getMessage()}"
        
        if hasattr(record, 'context') and isinstance(record.context, LogContext):
            context_str = f"[{record.context.component}:{record.context.operation}]"
            message = f"{timestamp} | {colored_levelname} | {record.name:20} | {context_str} {record.getMessage()}"
        
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return message

class LoggingManager:
    """Centralized logging manager"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = self.config.get('log_level', 'INFO')
        log_format = self.config.get('log_format', 'json')
        log_dir = Path(self.config.get('log_dir', 'logs'))
        environment = self.config.get('environment', 'development')
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if environment == 'development':
            formatter = ColoredConsoleFormatter()
        else:
            formatter = StructuredFormatter()
        
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
        # File handler
        app_log_file = log_dir / "rag_system.log"
        file_handler = logging.handlers.RotatingFileHandler(
            app_log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        
        # Error handler
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=20*1024*1024, backupCount=10, encoding='utf-8'
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name"""
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(f'rag_system.{name}')
        return self.loggers[name]
    
    def log_with_context(self, logger_name: str, level: str, message: str, 
                        context: Optional[LogContext] = None, **kwargs):
        """Log message with structured context"""
        logger = self.get_logger(logger_name)
        log_method = getattr(logger, level.lower())
        
        extra = kwargs.copy()
        if context:
            extra['context'] = context
        
        log_method(message, extra=extra)
    
    def log_api_request(self, method: str, path: str, status_code: int, 
                       response_time: float, **kwargs):
        """Log API request"""
        context = LogContext(
            component="api",
            operation="request",
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
    
    def log_error(self, component: str, operation: str, error: Exception,
                  context_data: Optional[Dict[str, Any]] = None):
        """Log error with context"""
        context = LogContext(
            component=component,
            operation=operation,
            metadata=context_data or {}
        )
        
        logger = self.get_logger(component)
        logger.error(
            f"Error in {component}.{operation}: {str(error)}",
            exc_info=True,
            extra={'context': context}
        )

# Global instance
_logging_manager: Optional[LoggingManager] = None

def get_logging_manager() -> LoggingManager:
    """Get the global logging manager"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager

def setup_logging(config: Dict[str, Any]):
    """Setup global logging"""
    global _logging_manager
    _logging_manager = LoggingManager(config)
    return _logging_manager

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
    
    return logger 