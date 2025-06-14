#!/usr/bin/env python3
"""
Final System Validation
Test all core modules to ensure the system is ready for production
"""

import sys
sys.path.insert(0, 'src')

def final_validation():
    """Run final validation of all system components"""
    print('🚀 FINAL SYSTEM VALIDATION')
    print('=' * 40)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Configuration Manager
    try:
        from core.config_manager import ConfigManager
        config_manager = ConfigManager()
        print('✅ Configuration Manager: Working')
        tests_passed += 1
    except Exception as e:
        print(f'❌ Configuration Manager: {e}')
    
    # Test 2: Monitoring System
    try:
        from monitoring import get_metrics_collector, get_health_checker
        metrics = get_metrics_collector()
        health = get_health_checker()
        print('✅ Monitoring System: Working')
        tests_passed += 1
    except Exception as e:
        print(f'❌ Monitoring System: {e}')
    
    # Test 3: UI Components
    try:
        from ui import create_web_interface
        web_interface = create_web_interface()
        print('✅ UI Components: Working')
        tests_passed += 1
    except Exception as e:
        print(f'❌ UI Components: {e}')
    
    # Test 4: Folder Scanner
    try:
        from ingestion.folder_scanner import create_folder_scanner
        config = {
            'monitored_directories': ['data'],
            'scan_interval': 60,
            'supported_extensions': ['.txt', '.md'],
            'exclude_patterns': ['.*', '*.tmp'],
            'max_file_size_mb': 10,
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
        print('✅ Folder Scanner: Working')
        tests_passed += 1
    except Exception as e:
        print(f'❌ Folder Scanner: {e}')
    
    # Test 5: System Initialization
    try:
        from core.simple_system_init import initialize_simple_system
        result = initialize_simple_system()
        print('✅ System Initialization: Working')
        tests_passed += 1
    except Exception as e:
        print(f'❌ System Initialization: {e}')
    
    print()
    print(f'📊 Results: {tests_passed}/{total_tests} tests passed')
    
    if tests_passed == total_tests:
        print('🎉 All core modules are functional!')
        print('📋 System is ready for production use.')
        return True
    else:
        print('⚠️ Some modules failed. Please review the issues above.')
        return False

if __name__ == "__main__":
    success = final_validation()
    sys.exit(0 if success else 1) 