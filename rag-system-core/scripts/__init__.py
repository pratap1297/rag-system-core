"""
RAG System Scripts Package
Organized entry points and utilities for the RAG system
"""

__version__ = "1.0.0"
__author__ = "RAG System Team"

# Entry point modules
from . import start_api
from . import start_ui  
from . import start_system

# Utility modules
from .utilities import health_check
from .utilities import diagnostics
from .utilities import migration
from .utilities import folder_manager

__all__ = [
    'start_api',
    'start_ui', 
    'start_system',
    'health_check',
    'diagnostics', 
    'migration',
    'folder_manager'
] 