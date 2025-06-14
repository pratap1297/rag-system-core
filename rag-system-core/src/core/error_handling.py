"""
Comprehensive Error Handling System for RAG System
Provides custom exceptions, error recovery, and monitoring integration
"""

import time
import functools
from typing import Dict, Any, Optional, Callable, Type, Union, List
from dataclasses import dataclass
from enum import Enum
import traceback
import asyncio

from .logging_system import get_logging_manager, LogContext

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification"""
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    AZURE_SERVICE = "azure_service"
    SERVICENOW = "servicenow"
    DATABASE = "database"
    LLM_PROVIDER = "llm_provider"
    EMBEDDING_PROVIDER = "embedding_provider"
    FILE_PROCESSING = "file_processing"
    VALIDATION = "validation"
    SYSTEM = "system"
    USER_INPUT = "user_input"

@dataclass
class ErrorContext:
    """Error context information"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RAGSystemError(Exception):
    """Base exception for RAG System"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 original_error: Optional[Exception] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context
        self.original_error = original_error
        self.recoverable = recoverable
        self.timestamp = time.time()

class ConfigurationError(RAGSystemError):
    """Configuration-related errors"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message, 
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.config_key = config_key

class AuthenticationError(RAGSystemError):
    """Authentication and authorization errors"""
    
    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.provider = provider

class NetworkError(RAGSystemError):
    """Network connectivity errors"""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, 
                 status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.endpoint = endpoint
        self.status_code = status_code

class AzureServiceError(RAGSystemError):
    """Azure service-specific errors"""
    
    def __init__(self, message: str, service: str, operation: Optional[str] = None,
                 error_code: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.AZURE_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.service = service
        self.operation = operation
        self.error_code = error_code

class ServiceNowError(RAGSystemError):
    """ServiceNow integration errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 instance: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SERVICENOW,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.operation = operation
        self.instance = instance

class DatabaseError(RAGSystemError):
    """Database and vector store errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.operation = operation

class LLMProviderError(RAGSystemError):
    """LLM provider errors"""
    
    def __init__(self, message: str, provider: str, model: Optional[str] = None,
                 error_code: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.LLM_PROVIDER,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.provider = provider
        self.model = model
        self.error_code = error_code

class EmbeddingProviderError(RAGSystemError):
    """Embedding provider errors"""
    
    def __init__(self, message: str, provider: str, model: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EMBEDDING_PROVIDER,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.provider = provider
        self.model = model

class FileProcessingError(RAGSystemError):
    """File processing errors"""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 file_type: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.FILE_PROCESSING,
            severity=ErrorSeverity.LOW,
            **kwargs
        )
        self.file_path = file_path
        self.file_type = file_type

class ValidationError(RAGSystemError):
    """Input validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            recoverable=False,
            **kwargs
        )
        self.field = field
        self.value = value

class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

class ErrorHandler:
    """Centralized error handling and recovery"""
    
    def __init__(self):
        self.logging_manager = get_logging_manager()
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None,
                    notify: bool = True) -> RAGSystemError:
        """Handle and classify errors"""
        
        # Convert to RAGSystemError if needed
        if isinstance(error, RAGSystemError):
            rag_error = error
        else:
            rag_error = self._classify_error(error, context)
        
        # Log the error
        self._log_error(rag_error)
        
        # Update error metrics
        self._update_error_metrics(rag_error)
        
        # Check circuit breaker
        self._check_circuit_breaker(rag_error)
        
        # Send notifications if needed
        if notify and rag_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_error_notification(rag_error)
        
        return rag_error
    
    def _classify_error(self, error: Exception, context: Optional[ErrorContext] = None) -> RAGSystemError:
        """Classify generic exceptions into RAGSystemError"""
        
        error_message = str(error)
        error_type = type(error).__name__
        
        # Network-related errors
        if any(keyword in error_message.lower() for keyword in 
               ['connection', 'timeout', 'network', 'unreachable', 'dns']):
            return NetworkError(
                f"Network error: {error_message}",
                context=context,
                original_error=error
            )
        
        # Authentication errors
        if any(keyword in error_message.lower() for keyword in 
               ['unauthorized', 'authentication', 'invalid key', 'forbidden']):
            return AuthenticationError(
                f"Authentication error: {error_message}",
                context=context,
                original_error=error
            )
        
        # File processing errors
        if any(keyword in error_message.lower() for keyword in 
               ['file not found', 'permission denied', 'invalid file']):
            return FileProcessingError(
                f"File processing error: {error_message}",
                context=context,
                original_error=error
            )
        
        # Configuration errors
        if any(keyword in error_message.lower() for keyword in 
               ['configuration', 'config', 'missing required']):
            return ConfigurationError(
                f"Configuration error: {error_message}",
                context=context,
                original_error=error
            )
        
        # Default to system error
        return RAGSystemError(
            f"System error ({error_type}): {error_message}",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            original_error=error
        )
    
    def _log_error(self, error: RAGSystemError):
        """Log error with structured context"""
        
        log_context = LogContext(
            component=error.context.component if error.context else "unknown",
            operation=error.context.operation if error.context else "unknown",
            user_id=error.context.user_id if error.context else None,
            session_id=error.context.session_id if error.context else None,
            metadata={
                "category": error.category.value,
                "severity": error.severity.value,
                "recoverable": error.recoverable,
                "timestamp": error.timestamp,
                **(error.context.metadata if error.context and error.context.metadata else {})
            }
        )
        
        self.logging_manager.log_error(
            component=log_context.component,
            operation=log_context.operation,
            error=error.original_error or error,
            context_data=log_context.metadata
        )
    
    def _update_error_metrics(self, error: RAGSystemError):
        """Update error metrics for monitoring"""
        
        error_key = f"{error.category.value}:{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Component-specific metrics
        if error.context:
            component_key = f"{error.context.component}:errors"
            self.error_counts[component_key] = self.error_counts.get(component_key, 0) + 1
    
    def _check_circuit_breaker(self, error: RAGSystemError):
        """Check and update circuit breaker status"""
        
        if not error.context:
            return
        
        component = error.context.component
        operation = error.context.operation
        breaker_key = f"{component}:{operation}"
        
        if breaker_key not in self.circuit_breakers:
            self.circuit_breakers[breaker_key] = {
                "failure_count": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
        
        breaker = self.circuit_breakers[breaker_key]
        breaker["failure_count"] += 1
        breaker["last_failure"] = time.time()
        
        # Open circuit breaker if too many failures
        if breaker["failure_count"] >= 5 and breaker["state"] == "closed":
            breaker["state"] = "open"
            self.logging_manager.get_logger("circuit_breaker").warning(
                f"Circuit breaker opened for {breaker_key}"
            )
    
    def _send_error_notification(self, error: RAGSystemError):
        """Send error notifications for critical errors"""
        
        # This would integrate with alerting systems
        # For now, just log the critical error
        if error.severity == ErrorSeverity.CRITICAL:
            self.logging_manager.get_logger("alerts").critical(
                f"CRITICAL ERROR: {error.message}",
                extra={'context': error.context}
            )
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        
        return {
            "error_counts": self.error_counts.copy(),
            "circuit_breakers": {
                key: {
                    "failure_count": breaker["failure_count"],
                    "state": breaker["state"],
                    "last_failure": breaker["last_failure"]
                }
                for key, breaker in self.circuit_breakers.items()
            }
        }

def with_error_handling(component: str, operation: str, 
                       retry_config: Optional[RetryConfig] = None,
                       fallback: Optional[Callable] = None):
    """Decorator for automatic error handling and retry logic"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            context = ErrorContext(component=component, operation=operation)
            
            retry_cfg = retry_config or RetryConfig()
            last_error = None
            
            for attempt in range(retry_cfg.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    last_error = error_handler.handle_error(e, context)
                    
                    # Don't retry if not recoverable
                    if not last_error.recoverable:
                        break
                    
                    # Don't retry on last attempt
                    if attempt == retry_cfg.max_attempts - 1:
                        break
                    
                    # Calculate delay
                    delay = min(
                        retry_cfg.base_delay * (retry_cfg.exponential_base ** attempt),
                        retry_cfg.max_delay
                    )
                    
                    if retry_cfg.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    time.sleep(delay)
            
            # Try fallback if available
            if fallback and last_error.recoverable:
                try:
                    return fallback(*args, **kwargs)
                except Exception as fallback_error:
                    error_handler.handle_error(fallback_error, context)
            
            # Re-raise the last error
            raise last_error
        
        return wrapper
    return decorator

def with_async_error_handling(component: str, operation: str,
                             retry_config: Optional[RetryConfig] = None,
                             fallback: Optional[Callable] = None):
    """Async version of error handling decorator"""
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            context = ErrorContext(component=component, operation=operation)
            
            retry_cfg = retry_config or RetryConfig()
            last_error = None
            
            for attempt in range(retry_cfg.max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except Exception as e:
                    last_error = error_handler.handle_error(e, context)
                    
                    if not last_error.recoverable:
                        break
                    
                    if attempt == retry_cfg.max_attempts - 1:
                        break
                    
                    delay = min(
                        retry_cfg.base_delay * (retry_cfg.exponential_base ** attempt),
                        retry_cfg.max_delay
                    )
                    
                    if retry_cfg.jitter:
                        import random
                        delay *= (0.5 + random.random() * 0.5)
                    
                    await asyncio.sleep(delay)
            
            if fallback and last_error.recoverable:
                try:
                    return await fallback(*args, **kwargs)
                except Exception as fallback_error:
                    error_handler.handle_error(fallback_error, context)
            
            raise last_error
        
        return wrapper
    return decorator

# Global error handler instance
_error_handler: Optional[ErrorHandler] = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler 