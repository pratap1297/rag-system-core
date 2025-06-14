"""
Metadata Extractor for RAG System
Extracts and enriches document metadata
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from core.logging_system import get_logger
from core.exceptions import FileProcessingError
from core.error_handler import with_error_handling

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    mime_type: str
    created_date: datetime
    modified_date: datetime
    file_hash: str
    
    # Content metadata
    text_length: Optional[int] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    
    # Processing metadata
    processing_date: datetime = None
    extraction_method: str = ""
    extraction_confidence: float = 0.0
    
    # Custom metadata
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: List[str] = None
    
    # Additional metadata
    custom_fields: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.processing_date is None:
            self.processing_date = datetime.now()
        if self.keywords is None:
            self.keywords = []
        if self.custom_fields is None:
            self.custom_fields = {}

class MetadataExtractor:
    """Extracts metadata from documents"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger("metadata_extractor")
        
        self.logger.info("Metadata extractor initialized")
    
    @with_error_handling("metadata_extractor", "extract_metadata")
    def extract_metadata(self, file_path: str, text_content: str = "", 
                        extraction_result: Optional[Dict[str, Any]] = None) -> DocumentMetadata:
        """Extract comprehensive metadata from document"""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileProcessingError(f"File not found: {file_path}")
            
            # Basic file metadata
            file_stats = file_path.stat()
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Create base metadata
            metadata = DocumentMetadata(
                file_path=str(file_path),
                file_name=file_path.name,
                file_size=file_stats.st_size,
                file_type=file_path.suffix.lower(),
                mime_type=mime_type or 'unknown',
                created_date=datetime.fromtimestamp(file_stats.st_ctime),
                modified_date=datetime.fromtimestamp(file_stats.st_mtime),
                file_hash=file_hash
            )
            
            # Add content metadata
            if text_content:
                metadata.text_length = len(text_content)
                metadata.language = self._detect_language(text_content)
            
            # Add extraction metadata
            if extraction_result:
                metadata.extraction_method = extraction_result.get('extraction_method', '')
                metadata.extraction_confidence = extraction_result.get('confidence', 0.0)
                metadata.page_count = extraction_result.get('page_count')
                
                # Extract document-specific metadata
                doc_metadata = extraction_result.get('metadata', {})
                if doc_metadata:
                    metadata.title = doc_metadata.get('title', '')
                    metadata.author = doc_metadata.get('author', '')
                    metadata.subject = doc_metadata.get('subject', '')
                    
                    # Extract keywords if available
                    keywords = doc_metadata.get('keywords', '')
                    if keywords:
                        metadata.keywords = [k.strip() for k in keywords.split(',') if k.strip()]
            
            # Extract additional metadata based on file type
            self._extract_type_specific_metadata(metadata, file_path, text_content)
            
            self.logger.info(f"Metadata extracted for {file_path.name}: "
                           f"size={metadata.file_size}, type={metadata.file_type}")
            
            return metadata
            
        except Exception as e:
            raise FileProcessingError(f"Failed to extract metadata: {str(e)}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        
        try:
            hash_sha256 = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            
            return hash_sha256.hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate file hash: {e}")
            return ""
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        
        # Simplified language detection
        # In a full implementation, you might use libraries like langdetect
        
        if not text.strip():
            return "unknown"
        
        # Simple heuristics for common languages
        text_lower = text.lower()
        
        # English indicators
        english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if english_count >= 3:
            return "en"
        
        return "unknown"
    
    def _extract_type_specific_metadata(self, metadata: DocumentMetadata, 
                                      file_path: Path, text_content: str):
        """Extract metadata specific to file type"""
        
        file_type = metadata.file_type
        
        try:
            if file_type == '.pdf':
                self._extract_pdf_metadata(metadata, file_path)
            elif file_type in ['.docx', '.doc']:
                self._extract_word_metadata(metadata, file_path)
            elif file_type == '.txt':
                self._extract_text_metadata(metadata, text_content)
            elif file_type in ['.html', '.htm']:
                self._extract_html_metadata(metadata, text_content)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract {file_type} specific metadata: {e}")
    
    def _extract_pdf_metadata(self, metadata: DocumentMetadata, file_path: Path):
        """Extract PDF-specific metadata"""
        
        try:
            import PyPDF2
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    
                    # Update metadata with PDF info
                    if not metadata.title and pdf_meta.get('/Title'):
                        metadata.title = str(pdf_meta.get('/Title'))
                    
                    if not metadata.author and pdf_meta.get('/Author'):
                        metadata.author = str(pdf_meta.get('/Author'))
                    
                    if not metadata.subject and pdf_meta.get('/Subject'):
                        metadata.subject = str(pdf_meta.get('/Subject'))
                    
                    # Add PDF-specific fields
                    metadata.custom_fields.update({
                        'pdf_creator': str(pdf_meta.get('/Creator', '')),
                        'pdf_producer': str(pdf_meta.get('/Producer', '')),
                        'pdf_creation_date': str(pdf_meta.get('/CreationDate', '')),
                        'pdf_mod_date': str(pdf_meta.get('/ModDate', ''))
                    })
                
                # Update page count if not already set
                if not metadata.page_count:
                    metadata.page_count = len(pdf_reader.pages)
                    
        except ImportError:
            self.logger.warning("PyPDF2 not available for PDF metadata extraction")
        except Exception as e:
            self.logger.warning(f"Failed to extract PDF metadata: {e}")
    
    def _extract_word_metadata(self, metadata: DocumentMetadata, file_path: Path):
        """Extract Word document metadata"""
        
        try:
            import docx
            
            doc = docx.Document(file_path)
            
            if hasattr(doc, 'core_properties'):
                props = doc.core_properties
                
                # Update metadata with Word properties
                if not metadata.title and props.title:
                    metadata.title = props.title
                
                if not metadata.author and props.author:
                    metadata.author = props.author
                
                if not metadata.subject and props.subject:
                    metadata.subject = props.subject
                
                # Add Word-specific fields
                metadata.custom_fields.update({
                    'word_category': props.category or '',
                    'word_comments': props.comments or '',
                    'word_keywords': props.keywords or '',
                    'word_language': props.language or '',
                    'word_last_modified_by': props.last_modified_by or '',
                    'word_revision': str(props.revision) if props.revision else '',
                    'word_version': props.version or ''
                })
                
                # Update keywords
                if props.keywords and not metadata.keywords:
                    metadata.keywords = [k.strip() for k in props.keywords.split(',') if k.strip()]
                    
        except ImportError:
            self.logger.warning("python-docx not available for Word metadata extraction")
        except Exception as e:
            self.logger.warning(f"Failed to extract Word metadata: {e}")
    
    def _extract_text_metadata(self, metadata: DocumentMetadata, text_content: str):
        """Extract text file metadata"""
        
        if text_content:
            # Simple text analysis
            lines = text_content.split('\n')
            words = text_content.split()
            
            metadata.custom_fields.update({
                'line_count': len(lines),
                'word_count': len(words),
                'character_count': len(text_content),
                'paragraph_count': len([line for line in lines if line.strip()])
            })
            
            # Try to extract title from first line
            if not metadata.title and lines:
                first_line = lines[0].strip()
                if first_line and len(first_line) < 100:
                    metadata.title = first_line
    
    def _extract_html_metadata(self, metadata: DocumentMetadata, text_content: str):
        """Extract HTML metadata"""
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(text_content, 'html.parser')
            
            # Extract title
            if not metadata.title and soup.title:
                metadata.title = soup.title.string.strip()
            
            # Extract meta tags
            meta_tags = soup.find_all('meta')
            html_meta = {}
            
            for tag in meta_tags:
                if tag.get('name') and tag.get('content'):
                    name = tag.get('name').lower()
                    content = tag.get('content')
                    
                    if name == 'author' and not metadata.author:
                        metadata.author = content
                    elif name == 'description' and not metadata.subject:
                        metadata.subject = content
                    elif name == 'keywords' and not metadata.keywords:
                        metadata.keywords = [k.strip() for k in content.split(',') if k.strip()]
                    
                    html_meta[f'html_{name}'] = content
            
            metadata.custom_fields.update(html_meta)
            
        except ImportError:
            self.logger.warning("BeautifulSoup not available for HTML metadata extraction")
        except Exception as e:
            self.logger.warning(f"Failed to extract HTML metadata: {e}")
    
    def enrich_metadata(self, metadata: DocumentMetadata, 
                       additional_data: Dict[str, Any]) -> DocumentMetadata:
        """Enrich metadata with additional information"""
        
        try:
            # Update custom fields
            metadata.custom_fields.update(additional_data.get('custom_fields', {}))
            
            # Update standard fields if not already set
            if not metadata.title and additional_data.get('title'):
                metadata.title = additional_data['title']
            
            if not metadata.author and additional_data.get('author'):
                metadata.author = additional_data['author']
            
            if not metadata.subject and additional_data.get('subject'):
                metadata.subject = additional_data['subject']
            
            # Add keywords
            if additional_data.get('keywords'):
                new_keywords = additional_data['keywords']
                if isinstance(new_keywords, str):
                    new_keywords = [k.strip() for k in new_keywords.split(',') if k.strip()]
                
                # Merge with existing keywords
                all_keywords = set(metadata.keywords + new_keywords)
                metadata.keywords = list(all_keywords)
            
            self.logger.info(f"Metadata enriched for {metadata.file_name}")
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to enrich metadata: {e}")
            return metadata
    
    def to_dict(self, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        
        return {
            'file_path': metadata.file_path,
            'file_name': metadata.file_name,
            'file_size': metadata.file_size,
            'file_type': metadata.file_type,
            'mime_type': metadata.mime_type,
            'created_date': metadata.created_date.isoformat(),
            'modified_date': metadata.modified_date.isoformat(),
            'file_hash': metadata.file_hash,
            'text_length': metadata.text_length,
            'page_count': metadata.page_count,
            'language': metadata.language,
            'processing_date': metadata.processing_date.isoformat(),
            'extraction_method': metadata.extraction_method,
            'extraction_confidence': metadata.extraction_confidence,
            'title': metadata.title,
            'author': metadata.author,
            'subject': metadata.subject,
            'keywords': metadata.keywords,
            'custom_fields': metadata.custom_fields
        } 