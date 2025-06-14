"""
Error Handler with Retry Logic
"""

import time
import functools
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .exceptions import RAGSystemError, ErrorContext
from .logging_system import get_logging_manager, get_logger

@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0

class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self):
        self.logging_manager = get_logging_manager()
        self.error_counts: Dict[str, int] = {}
    
    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> RAGSystemError:
        """Handle and classify errors"""
        
        if isinstance(error, RAGSystemError):
            rag_error = error
        else:
            rag_error = RAGSystemError(
                f"System error: {str(error)}",
                context=context,
                original_error=error
            )
        
        self._log_error(rag_error)
        return rag_error
    
    def _log_error(self, error: RAGSystemError):
        """Log error"""
        component = error.context.component if error.context else "unknown"
        operation = error.context.operation if error.context else "unknown"
        
        self.logging_manager.log_error(
            component=component,
            operation=operation,
            error=error.original_error or error
        )

def with_error_handling(component: str, operation: str) -> Callable:
    """
    Decorator for handling errors in component operations
    
    Args:
        component: Name of the component
        operation: Name of the operation
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger(f"{component}.{operation}")
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {component}.{operation}: {str(e)}")
                raise
        return wrapper
    return decorator

_error_handler: Optional[ErrorHandler] = None

def get_error_handler() -> ErrorHandler:
    """Get the global error handler"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler 