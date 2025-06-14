#!/usr/bin/env python3
"""
Data Migration Script for RAG System Core
Migrate data from old RAG system to new core system
"""
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def setup_logging():
    """Setup logging for migration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('migration.log'),
            logging.StreamHandler()
        ]
    )

def migrate_documents(old_data_dir: str, new_data_dir: str) -> int:
    """Migrate documents from old system to new system"""
    old_path = Path(old_data_dir)
    new_path = Path(new_data_dir)
    
    if not old_path.exists():
        logging.warning(f"Old data directory not found: {old_path}")
        return 0
    
    # Create new directories
    new_uploads = new_path / "uploads"
    new_uploads.mkdir(parents=True, exist_ok=True)
    
    migrated_count = 0
    
    # Find and copy documents
    for file_path in old_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt', '.md']:
            try:
                # Create relative path structure
                rel_path = file_path.relative_to(old_path)
                new_file_path = new_uploads / rel_path
                
                # Create parent directories
                new_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(file_path, new_file_path)
                logging.info(f"Migrated: {rel_path}")
                migrated_count += 1
                
            except Exception as e:
                logging.error(f"Failed to migrate {file_path}: {e}")
    
    return migrated_count

def migrate_vector_index(old_vector_path: str, new_vector_path: str) -> bool:
    """Migrate FAISS vector index if compatible"""
    old_index = Path(old_vector_path)
    new_vector_dir = Path(new_vector_path).parent
    
    if not old_index.exists():
        logging.warning(f"Old vector index not found: {old_index}")
        return False
    
    try:
        # Create new vector directory
        new_vector_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy index files
        if old_index.is_file():
            shutil.copy2(old_index, new_vector_dir / "index.faiss")
            logging.info("Migrated FAISS index")
        
        # Look for metadata files
        old_metadata = old_index.parent / "vector_metadata.pkl"
        if old_metadata.exists():
            shutil.copy2(old_metadata, new_vector_dir / "vector_metadata.pkl")
            logging.info("Migrated vector metadata")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to migrate vector index: {e}")
        return False

def migrate_configuration(old_config_dir: str, new_config_dir: str) -> bool:
    """Migrate configuration files"""
    old_path = Path(old_config_dir)
    new_path = Path(new_config_dir)
    
    if not old_path.exists():
        logging.warning(f"Old config directory not found: {old_path}")
        return False
    
    try:
        new_path.mkdir(parents=True, exist_ok=True)
        
        # Copy .env file if exists
        old_env = old_path.parent / ".env"
        new_env = new_path.parent / ".env"
        
        if old_env.exists() and not new_env.exists():
            shutil.copy2(old_env, new_env)
            logging.info("Migrated .env configuration")
        
        # Copy other config files
        for config_file in old_path.glob("*.json"):
            new_config_file = new_path / config_file.name
            if not new_config_file.exists():
                shutil.copy2(config_file, new_config_file)
                logging.info(f"Migrated config: {config_file.name}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to migrate configuration: {e}")
        return False

def main():
    """Main migration function"""
    setup_logging()
    
    print("🔄 RAG System Data Migration")
    print("=" * 50)
    
    # Get paths
    old_system_path = input("Enter path to old RAG system (e.g., ../rag-system): ").strip()
    if not old_system_path:
        old_system_path = "../rag-system"
    
    old_path = Path(old_system_path)
    new_path = Path(".")  # Current directory (rag-system-core)
    
    if not old_path.exists():
        print(f"❌ Old system path not found: {old_path}")
        return
    
    print(f"📂 Old system: {old_path.absolute()}")
    print(f"📂 New system: {new_path.absolute()}")
    print()
    
    # Migration options
    print("🔧 Migration Options:")
    print("1. Documents only (recommended for fresh start)")
    print("2. Documents + Vector Index (if compatible)")
    print("3. Documents + Configuration")
    print("4. Full migration (Documents + Vectors + Config)")
    print("5. Custom migration")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    migrated_docs = 0
    migrated_vectors = False
    migrated_config = False
    
    if choice in ['1', '2', '3', '4']:
        # Migrate documents
        print("\n📄 Migrating documents...")
        old_data = old_path / "data"
        new_data = new_path / "data"
        migrated_docs = migrate_documents(str(old_data), str(new_data))
        
        if choice in ['2', '4']:
            # Migrate vector index
            print("\n🔍 Migrating vector index...")
            old_vectors = old_path / "data" / "vectors" / "index.faiss"
            new_vectors = new_path / "data" / "vectors" / "index.faiss"
            migrated_vectors = migrate_vector_index(str(old_vectors), str(new_vectors))
        
        if choice in ['3', '4']:
            # Migrate configuration
            print("\n⚙️ Migrating configuration...")
            old_config = old_path / "data" / "config"
            new_config = new_path / "data" / "config"
            migrated_config = migrate_configuration(str(old_config), str(new_config))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Migration Summary:")
    print(f"📄 Documents migrated: {migrated_docs}")
    print(f"🔍 Vector index migrated: {'✅' if migrated_vectors else '❌'}")
    print(f"⚙️ Configuration migrated: {'✅' if migrated_config else '❌'}")
    
    if migrated_docs > 0:
        print("\n✅ Migration completed!")
        print("\n🚀 Next steps:")
        print("1. Start the new RAG system: python launch_fixed_ui.py")
        print("2. If you migrated documents only, they will be auto-processed")
        print("3. If you migrated vectors, verify they work with: python check_vector_stats.py")
        print("4. Check system health: python health_check_cli.py")
    else:
        print("\n⚠️ No documents were migrated.")
        print("You'll need to upload new documents to the system.")

if __name__ == "__main__":
    main() 