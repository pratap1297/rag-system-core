#!/usr/bin/env python3
"""
Debug script to test folder scanner functionality
"""

import tempfile
from pathlib import Path
import sys
import shutil

# Add src to path
sys.path.insert(0, 'src')

def debug_scanner():
    """Debug the folder scanner"""
    print("🔍 DEBUGGING FOLDER SCANNER")
    print("=" * 40)
    
    # Create test directory
    test_dir = Path(tempfile.mkdtemp(prefix='debug_test_'))
    docs_dir = test_dir / 'documents'
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create test files
        test_files = [
            ('test1.txt', 'Test content 1'),
            ('test2.md', '# Test Document 2'),
            ('subdir/test3.txt', 'Test content 3')
        ]
        
        for file_path, content in test_files:
            full_path = docs_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
            print(f"✅ Created: {full_path}")
        
        print(f"\n📁 Test directory: {test_dir}")
        print(f"📁 Docs directory: {docs_dir}")
        print(f"📁 Directory exists: {docs_dir.exists()}")
        
        # List files in directory
        print(f"\n📋 Files in directory:")
        for file_path in docs_dir.rglob('*'):
            if file_path.is_file():
                print(f"  - {file_path} (size: {file_path.stat().st_size})")
        
        # Test scanner
        print(f"\n🔧 Testing scanner...")
        from ingestion.folder_scanner import create_folder_scanner
        
        config = {
            'monitored_directories': [str(docs_dir)],
            'scan_interval': 5,
            'max_depth': 3,
            'enable_content_hashing': True,
            'supported_extensions': ['.txt', '.md'],
            'max_file_size_mb': 10,
            'exclude_patterns': ['.*', '*.tmp'],
            'max_concurrent_files': 2,
            'retry_attempts': 2,
            'retry_delay': 5,
            'processing_timeout': 30,
            'path_metadata_rules': {},
            'auto_categorization': True,
            'enable_parallel_scanning': True,
            'scan_batch_size': 50,
            'memory_limit_mb': 256
        }
        
        scanner = create_folder_scanner(config)
        print(f"✅ Scanner created")
        
        # Check scanner config
        print(f"📋 Scanner config:")
        print(f"  - Monitored directories: {scanner.config.monitored_directories}")
        print(f"  - Supported extensions: {scanner.config.supported_extensions}")
        print(f"  - Exclude patterns: {scanner.config.exclude_patterns}")
        
        # Perform scan
        print(f"\n🔍 Performing scan...")
        result = scanner.force_scan()
        print(f"📊 Scan result: {result}")
        
        # Get file states
        file_states = scanner.get_file_states()
        print(f"📊 File states: {len(file_states)} files tracked")
        
        if file_states:
            for path, state in file_states.items():
                print(f"  - {path}")
                print(f"    Status: {state.get('status', 'unknown')}")
                if 'metadata' in state:
                    metadata = state['metadata']
                    print(f"    Size: {metadata.get('size', 'unknown')}")
                    print(f"    Extension: {metadata.get('extension', 'unknown')}")
        else:
            print("  ❌ No files tracked!")
            
            # Debug why no files were found
            print(f"\n🔍 Debugging file detection...")
            
            # Check if files meet criteria
            for file_path in docs_dir.rglob('*'):
                if file_path.is_file():
                    print(f"\n  Checking: {file_path}")
                    
                    # Check extension
                    extension = file_path.suffix.lower()
                    print(f"    Extension: {extension}")
                    print(f"    Extension supported: {extension in config['supported_extensions']}")
                    
                    # Check size
                    size = file_path.stat().st_size
                    max_size = config['max_file_size_mb'] * 1024 * 1024
                    print(f"    Size: {size} bytes")
                    print(f"    Size OK: {size <= max_size}")
                    
                    # Check exclude patterns
                    excluded = False
                    for pattern in config['exclude_patterns']:
                        if pattern.startswith('*') and str(file_path).endswith(pattern[1:]):
                            excluded = True
                            break
                        elif pattern.endswith('*') and str(file_path).startswith(pattern[:-1]):
                            excluded = True
                            break
                        elif pattern in str(file_path):
                            excluded = True
                            break
                    print(f"    Excluded: {excluded}")
        
        # Get statistics
        stats = scanner.get_statistics()
        print(f"\n📊 Scanner statistics:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"✅ Cleanup complete")

if __name__ == "__main__":
    debug_scanner() 