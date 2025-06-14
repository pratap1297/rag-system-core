"""
Custom Exception Classes for RAG System
Provides structured error handling with categorization and context
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

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

@dataclass
class ErrorContext:
    """Error context information"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
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

class ValidationError(RAGSystemError):
    """Input validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            recoverable=False,
            **kwargs
        )
        self.field = field

class ProcessingError(RAGSystemError):
    """General processing errors"""
    
    def __init__(self, message: str, component: Optional[str] = None, 
                 operation: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.component = component
        self.operation = operation

class FileProcessingError(ProcessingError):
    """Exception for file processing errors"""
    pass

class ExtractionError(ProcessingError):
    """Exception for text extraction errors"""
    pass

class ChunkingError(ProcessingError):
    """Exception for text chunking errors"""
    pass

class EmbeddingError(ProcessingError):
    """Exception for embedding generation errors"""
    pass 