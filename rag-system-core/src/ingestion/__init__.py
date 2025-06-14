"""
Ingestion module for RAG System
Handles document ingestion streams and folder monitoring
"""

# Phase 5.1: Folder Scanner Module
from .folder_scanner import (
    FolderScanner,
    FileStatus,
    ChangeType,
    FileMetadata,
    ScannerStats,
    create_folder_scanner
)

__all__ = [
    # Phase 5.1: Folder Scanner
    'FolderScanner',
    'FileStatus',
    'ChangeType', 
    'FileMetadata',
    'ScannerStats',
    'create_folder_scanner'
] 