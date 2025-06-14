"""
Persistent JSON Metadata Store
Stores metadata in JSON files on disk for persistence across restarts
"""
import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class PersistentJSONMetadataStore:
    """Persistent metadata store using JSON files"""
    
    def __init__(self, base_path: str = "data/metadata"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # JSON file paths
        self.files_metadata_path = self.base_path / "files_metadata.json"
        self.chunks_metadata_path = self.base_path / "chunks_metadata.json"
        self.vector_mappings_path = self.base_path / "vector_mappings.json"
        
        # In-memory caches for performance
        self._files_cache = None
        self._chunks_cache = None
        self._vector_mappings_cache = None
        
        # Load existing data
        self._load_all_data()
        
        print(f"🔧 PersistentJSONMetadataStore initialized at {base_path}")
        print(f"✅ Persistent JSON metadata store ready")
    
    def _load_all_data(self):
        """Load all data from JSON files"""
        self._files_cache = self._load_json(self.files_metadata_path, {})
        self._chunks_cache = self._load_json(self.chunks_metadata_path, {})
        self._vector_mappings_cache = self._load_json(self.vector_mappings_path, {})
    
    def _load_json(self, file_path: Path, default: Any = None) -> Any:
        """Load JSON file with error handling"""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
        return default or {}
    
    def _save_json(self, file_path: Path, data: Any):
        """Save data to JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
    
    def add_file_metadata(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Add file metadata"""
        file_id = str(uuid.uuid4())
        
        file_metadata = {
            'file_id': file_id,
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'created_at': datetime.now().isoformat(),
            'type': 'file',
            **metadata
        }
        
        # Add to cache and save
        self._files_cache[file_id] = file_metadata
        self._save_json(self.files_metadata_path, self._files_cache)
        
        return file_id
    
    def add_chunk_metadata(self, chunk_data: Dict[str, Any]) -> str:
        """Add chunk metadata with vector linking"""
        chunk_id = chunk_data.get('chunk_id') or str(uuid.uuid4())
        
        chunk_metadata = {
            'chunk_id': chunk_id,
            'created_at': datetime.now().isoformat(),
            'type': 'chunk',
            **chunk_data
        }
        
        # Add to cache and save
        self._chunks_cache[chunk_id] = chunk_metadata
        self._save_json(self.chunks_metadata_path, self._chunks_cache)
        
        # If there's a vector_id, create mapping
        vector_id = chunk_data.get('vector_id')
        if vector_id:
            self._vector_mappings_cache[str(vector_id)] = {
                'chunk_id': chunk_id,
                'doc_id': chunk_data.get('doc_id', 'unknown'),
                'filename': chunk_data.get('filename', 'unknown'),
                'created_at': datetime.now().isoformat()
            }
            self._save_json(self.vector_mappings_path, self._vector_mappings_cache)
        
        return chunk_id
    
    def get_metadata_by_vector_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata using vector ID"""
        mapping = self._vector_mappings_cache.get(str(vector_id))
        if mapping:
            chunk_id = mapping.get('chunk_id')
            if chunk_id:
                return self._chunks_cache.get(chunk_id)
        return None
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """Get all files with chunk count"""
        files = []
        for file_id, file_metadata in self._files_cache.items():
            # Count chunks for this file
            chunk_count = sum(1 for chunk in self._chunks_cache.values() 
                            if chunk.get('doc_id') == file_id)
            
            file_info = {
                **file_metadata,
                'chunk_count': chunk_count
            }
            files.append(file_info)
        
        return files
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks"""
        return list(self._chunks_cache.values())
    
    def get_file_chunks(self, file_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a file"""
        return [chunk for chunk in self._chunks_cache.values() 
                if chunk.get('doc_id') == file_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metadata store statistics"""
        return {
            'total_files': len(self._files_cache),
            'total_chunks': len(self._chunks_cache),
            'total_vector_mappings': len(self._vector_mappings_cache),
            'storage_path': str(self.base_path),
            'files_file_size': self.files_metadata_path.stat().st_size if self.files_metadata_path.exists() else 0,
            'chunks_file_size': self.chunks_metadata_path.stat().st_size if self.chunks_metadata_path.exists() else 0,
            'mappings_file_size': self.vector_mappings_path.stat().st_size if self.vector_mappings_path.exists() else 0
        }
    
    def clear_all_data(self):
        """Clear all metadata (for testing)"""
        self._files_cache = {}
        self._chunks_cache = {}
        self._vector_mappings_cache = {}
        
        # Remove files
        for file_path in [self.files_metadata_path, self.chunks_metadata_path, self.vector_mappings_path]:
            if file_path.exists():
                file_path.unlink()
    
    def backup_metadata(self, backup_path: str) -> str:
        """Create backup of all metadata"""
        backup_dir = Path(backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Backup each file
        for source_file in [self.files_metadata_path, self.chunks_metadata_path, self.vector_mappings_path]:
            if source_file.exists():
                backup_file = backup_dir / f"{source_file.stem}_{timestamp}.json"
                backup_file.write_text(source_file.read_text(encoding='utf-8'), encoding='utf-8')
        
        return str(backup_dir) 