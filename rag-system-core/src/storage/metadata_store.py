"""
Metadata Store
Manages metadata for documents and chunks using JSON storage
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..core.json_store import JSONStore
from ..core.error_handling import StorageError

class MetadataStore:
    """Metadata storage and management"""
    
    def __init__(self, config_manager, json_store_factory):
        self.config = config_manager.get_config()
        self.metadata_path = Path(self.config.database.metadata_path)
        
        # Initialize JSON stores for different collections
        self.files_store = json_store_factory(
            self.metadata_path / "files_metadata.json"
        )
        self.chunks_store = json_store_factory(
            self.metadata_path / "chunks_metadata.json"
        )
        self.collections_store = json_store_factory(
            self.metadata_path / "collections.json"
        )
        
        # Initialize collections registry
        self._initialize_collections()
        
        logging.info("Metadata store initialized")
    
    def _initialize_collections(self):
        """Initialize collections registry"""
        collections = self.collections_store.load()
        if not collections:
            collections = {
                'files_metadata': {
                    'created_at': datetime.now().isoformat(),
                    'description': 'File metadata collection',
                    'count': 0
                },
                'chunks_metadata': {
                    'created_at': datetime.now().isoformat(),
                    'description': 'Chunk metadata collection',
                    'count': 0
                }
            }
            self.collections_store.save(collections)
    
    def add_file_metadata(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Add metadata for a file"""
        try:
            file_id = str(uuid.uuid4())
            
            file_metadata = {
                'file_id': file_id,
                'file_path': file_path,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                **metadata
            }
            
            # Load existing data
            files_data = self.files_store.load()
            files_data[file_id] = file_metadata
            
            # Save updated data
            self.files_store.save(files_data)
            
            # Update collection count
            self._update_collection_count('files_metadata', len(files_data))
            
            logging.info(f"Added file metadata: {file_id}")
            return file_id
            
        except Exception as e:
            raise StorageError(f"Failed to add file metadata: {e}")
    
    def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific file"""
        try:
            files_data = self.files_store.load()
            return files_data.get(file_id)
        except Exception as e:
            raise StorageError(f"Failed to get file metadata: {e}")
    
    def update_file_metadata(self, file_id: str, updates: Dict[str, Any]) -> bool:
        """Update metadata for a file"""
        try:
            files_data = self.files_store.load()
            
            if file_id not in files_data:
                return False
            
            files_data[file_id].update(updates)
            files_data[file_id]['updated_at'] = datetime.now().isoformat()
            
            self.files_store.save(files_data)
            
            logging.info(f"Updated file metadata: {file_id}")
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to update file metadata: {e}")
    
    def delete_file_metadata(self, file_id: str) -> bool:
        """Delete metadata for a file"""
        try:
            files_data = self.files_store.load()
            
            if file_id not in files_data:
                return False
            
            del files_data[file_id]
            self.files_store.save(files_data)
            
            # Update collection count
            self._update_collection_count('files_metadata', len(files_data))
            
            logging.info(f"Deleted file metadata: {file_id}")
            return True
            
        except Exception as e:
            raise StorageError(f"Failed to delete file metadata: {e}")
    
    def add_chunk_metadata(self, chunk_id: str, metadata: Dict[str, Any]) -> str:
        """Add metadata for a chunk"""
        try:
            chunk_metadata = {
                'chunk_id': chunk_id,
                'created_at': datetime.now().isoformat(),
                **metadata
            }
            
            # Load existing data
            chunks_data = self.chunks_store.load()
            chunks_data[chunk_id] = chunk_metadata
            
            # Save updated data
            self.chunks_store.save(chunks_data)
            
            # Update collection count
            self._update_collection_count('chunks_metadata', len(chunks_data))
            
            logging.debug(f"Added chunk metadata: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            raise StorageError(f"Failed to add chunk metadata: {e}")
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific chunk"""
        try:
            chunks_data = self.chunks_store.load()
            return chunks_data.get(chunk_id)
        except Exception as e:
            raise StorageError(f"Failed to get chunk metadata: {e}")
    
    def search_files(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search files by metadata filters"""
        try:
            files_data = self.files_store.load()
            results = []
            
            for file_id, metadata in files_data.items():
                if self._matches_filters(metadata, filters):
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            raise StorageError(f"Failed to search files: {e}")
    
    def search_chunks(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search chunks by metadata filters"""
        try:
            chunks_data = self.chunks_store.load()
            results = []
            
            for chunk_id, metadata in chunks_data.items():
                if self._matches_filters(metadata, filters):
                    results.append(metadata)
            
            return results
            
        except Exception as e:
            raise StorageError(f"Failed to search chunks: {e}")
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        
        return True
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            collections = self.collections_store.load()
            return list(collections.keys())
        except Exception as e:
            raise StorageError(f"Failed to list collections: {e}")
    
    def collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        try:
            collections = self.collections_store.load()
            return collections.get(collection_name, {})
        except Exception as e:
            raise StorageError(f"Failed to get collection stats: {e}")
    
    def _update_collection_count(self, collection_name: str, count: int):
        """Update the count for a collection"""
        try:
            collections = self.collections_store.load()
            if collection_name in collections:
                collections[collection_name]['count'] = count
                collections[collection_name]['updated_at'] = datetime.now().isoformat()
                self.collections_store.save(collections)
        except Exception as e:
            logging.error(f"Failed to update collection count: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall metadata store statistics"""
        try:
            files_data = self.files_store.load()
            collections = self.collections_store.load()
            
            return {
                'total_files': len(files_data),
                'collections': len(collections),
                'collections_info': collections
            }
            
        except Exception as e:
            raise StorageError(f"Failed to get stats: {e}")
    
    def backup(self, backup_path: str) -> bool:
        """Create a backup of all metadata"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup each store
            self.files_store.backup(backup_dir / f"files_metadata_{timestamp}.json")
            self.chunks_store.backup(backup_dir / f"chunks_metadata_{timestamp}.json")
            self.collections_store.backup(backup_dir / f"collections_{timestamp}.json")
            
            logging.info(f"Metadata backup created: {backup_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create backup: {e}")
            return False
    
    def restore(self, backup_path: str) -> bool:
        """Restore metadata from backup"""
        try:
            backup_dir = Path(backup_path)
            
            if not backup_dir.exists():
                raise StorageError(f"Backup path does not exist: {backup_path}")
            
            # Find the latest backup files
            files_backup = max(backup_dir.glob("files_metadata_*.json"), default=None)
            chunks_backup = max(backup_dir.glob("chunks_metadata_*.json"), default=None)
            collections_backup = max(backup_dir.glob("collections_*.json"), default=None)
            
            if files_backup:
                self.files_store.restore(files_backup)
            if chunks_backup:
                self.chunks_store.restore(chunks_backup)
            if collections_backup:
                self.collections_store.restore(collections_backup)
            
            logging.info(f"Metadata restored from: {backup_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to restore backup: {e}")
            return False 