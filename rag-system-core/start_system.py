#!/usr/bin/env python3
"""
RAG System Startup Script
Interactive system startup and testing
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def print_banner():
    """Print system banner"""
    print("🚀 RAG SYSTEM - ENTERPRISE DOCUMENT PROCESSING")
    print("=" * 60)
    print("Version: 1.0.0")
    print("Status: Production Ready")
    print("=" * 60)

def test_system_components():
    """Test all system components"""
    print("\n🧪 TESTING SYSTEM COMPONENTS")
    print("-" * 40)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Configuration Manager
    try:
        from core.config_manager import ConfigManager
        config_manager = ConfigManager()
        print("✅ Configuration Manager: Working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Configuration Manager: {e}")
    
    # Test 2: System Initialization
    try:
        from core.simple_system_init import initialize_simple_system
        result = initialize_simple_system()
        print("✅ System Initialization: Working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ System Initialization: {e}")
    
    # Test 3: Monitoring System
    try:
        from monitoring import get_metrics_collector, get_health_checker
        metrics = get_metrics_collector()
        health = get_health_checker()
        print("✅ Monitoring System: Working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Monitoring System: {e}")
    
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
        print("✅ Folder Scanner: Working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Folder Scanner: {e}")
    
    # Test 5: Web Interface
    try:
        from ui.web_interface import create_web_interface
        web_interface = create_web_interface()
        print("✅ Web Interface: Working")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Web Interface: {e}")
    
    print(f"\n📊 Results: {tests_passed}/{total_tests} components working")
    return tests_passed == total_tests

def create_sample_documents():
    """Create sample documents for testing"""
    print("\n📄 CREATING SAMPLE DOCUMENTS")
    print("-" * 40)
    
    docs_dir = Path('sample_documents')
    docs_dir.mkdir(exist_ok=True)
    
    sample_files = [
        ('maintenance_manual.txt', '''Equipment Maintenance Manual

1. Daily Inspections
   - Check oil levels
   - Inspect belts and chains
   - Verify safety systems

2. Weekly Maintenance
   - Lubricate moving parts
   - Check fluid levels
   - Test emergency stops

3. Monthly Procedures
   - Replace filters
   - Calibrate instruments
   - Update maintenance logs'''),
        
        ('safety_protocol.md', '''# Safety Protocol

## Emergency Procedures

### Fire Emergency
1. Sound alarm
2. Evacuate personnel
3. Call emergency services
4. Use appropriate extinguisher

### Medical Emergency
1. Assess situation
2. Provide first aid
3. Call medical assistance
4. Document incident

## Personal Protective Equipment
- Hard hats required in all areas
- Safety glasses mandatory
- Steel-toed boots required
- High-visibility vests in designated zones'''),
        
        ('technical_specifications.txt', '''Technical Specifications - Industrial Equipment

Model: XYZ-2000 Processing Unit
Capacity: 500 units/hour
Power Requirements: 480V, 3-phase, 60Hz
Operating Temperature: 10°C to 40°C
Dimensions: 2.5m x 1.8m x 2.2m
Weight: 1,200 kg

Performance Specifications:
- Processing Speed: 8.3 units/minute
- Accuracy: ±0.1mm
- Repeatability: ±0.05mm
- Cycle Time: 7.2 seconds

Maintenance Schedule:
- Daily: Visual inspection
- Weekly: Lubrication check
- Monthly: Calibration verification
- Quarterly: Full system audit'''),
        
        ('daily_report.md', '''# Daily Production Report

**Date:** 2024-01-15  
**Shift:** Day Shift (06:00 - 18:00)  
**Supervisor:** John Smith

## Production Summary
- **Target Output:** 1,200 units
- **Actual Output:** 1,250 units
- **Efficiency:** 104.2%
- **Quality Rate:** 99.8%

## Equipment Status
- Line 1: Operational (100%)
- Line 2: Operational (98% - minor adjustment needed)
- Line 3: Maintenance (scheduled downtime)

## Issues and Actions
1. **Minor vibration on Line 2**
   - Action: Scheduled bearing replacement
   - Timeline: Next maintenance window

2. **Quality check delay**
   - Action: Additional QC staff assigned
   - Resolution: Completed

## Recommendations
- Continue monitoring Line 2 performance
- Schedule preventive maintenance for Line 1
- Review staffing levels for peak periods''')
    ]
    
    for filename, content in sample_files:
        file_path = docs_dir / filename
        file_path.write_text(content)
        print(f"✅ Created: {filename} ({len(content)} chars)")
    
    print(f"\n📁 Sample documents created in: {docs_dir.absolute()}")
    return docs_dir

def test_folder_scanner(docs_dir):
    """Test folder scanner with sample documents"""
    print("\n🔍 TESTING FOLDER SCANNER")
    print("-" * 40)
    
    try:
        from ingestion.folder_scanner import create_folder_scanner
        
        config = {
            'monitored_directories': [str(docs_dir)],
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
        
        print("🔧 Creating folder scanner...")
        scanner = create_folder_scanner(config)
        
        print("🔍 Performing scan...")
        result = scanner.force_scan()
        
        print(f"✅ Scan completed successfully")
        print(f"📊 Files tracked: {result['files_tracked']}")
        print(f"📊 Queue size: {result['queue_size']}")
        
        # Get file states
        file_states = scanner.get_file_states()
        print(f"\n📋 Detected Files:")
        for file_path, state in file_states.items():
            filename = Path(file_path).name
            status = state.get('status', 'unknown')
            size = state.get('metadata', {}).get('size', 0)
            print(f"  - {filename} ({size} bytes, {status})")
        
        # Get statistics
        stats = scanner.get_statistics()
        print(f"\n📈 Scanner Statistics:")
        print(f"  - Total scans: {stats['total_scans']}")
        print(f"  - Last scan duration: {stats['last_scan_duration']:.2f}s")
        print(f"  - Files tracked: {stats['total_files_tracked']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Folder scanner test failed: {e}")
        return False

def start_web_interface():
    """Start the web interface"""
    print("\n🌐 STARTING WEB INTERFACE")
    print("-" * 40)
    
    try:
        from ui.web_interface import create_web_interface
        from core.config_manager import ConfigManager
        
        # Check if FastAPI is available
        try:
            import uvicorn
            fastapi_available = True
        except ImportError:
            fastapi_available = False
        
        config_manager = ConfigManager()
        web_interface = create_web_interface(config_manager)
        
        if fastapi_available and web_interface.get_app():
            print("🚀 Starting FastAPI server...")
            print("📍 URL: http://localhost:8000")
            print("📍 API Docs: http://localhost:8000/docs")
            print("📍 Dashboard: http://localhost:8000/dashboard")
            print("\nPress Ctrl+C to stop the server")
            
            import uvicorn
            uvicorn.run(web_interface.get_app(), host="0.0.0.0", port=8000)
        else:
            print("⚠️ FastAPI not available, starting simple server...")
            from ui.web_interface import create_simple_status_server
            import http.server
            import socketserver
            
            handler_class, port = create_simple_status_server(8080)
            print(f"🚀 Starting simple server on http://localhost:{port}")
            print("Press Ctrl+C to stop the server")
            
            with socketserver.TCPServer(("", port), handler_class) as httpd:
                httpd.serve_forever()
                
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")

def show_cli_examples():
    """Show CLI command examples"""
    print("\n💻 CLI COMMAND EXAMPLES")
    print("-" * 40)
    
    examples = [
        ("System Status", "python rag-system status"),
        ("Scanner Status", "python rag-system scanner status"),
        ("Scanner Config", "python rag-system scanner config --show"),
        ("Force Scan", "python rag-system scanner scan --directory sample_documents"),
        ("List Files", "python rag-system scanner files --limit 10"),
        ("Health Check", "python rag-system health"),
        ("View Logs", "python rag-system logs --tail 20")
    ]
    
    for description, command in examples:
        print(f"📋 {description}:")
        print(f"   {command}")
        print()

def interactive_menu():
    """Show interactive menu"""
    while True:
        print("\n🎯 RAG SYSTEM - INTERACTIVE MENU")
        print("-" * 40)
        print("1. Test System Components")
        print("2. Create Sample Documents")
        print("3. Test Folder Scanner")
        print("4. Start Web Interface")
        print("5. Show CLI Examples")
        print("6. Run Complete Integration Test")
        print("0. Exit")
        print("-" * 40)
        
        try:
            choice = input("Select option (0-6): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                test_system_components()
            elif choice == "2":
                docs_dir = create_sample_documents()
                input("\nPress Enter to continue...")
            elif choice == "3":
                docs_dir = Path('sample_documents')
                if not docs_dir.exists():
                    print("⚠️ Sample documents not found. Creating them first...")
                    docs_dir = create_sample_documents()
                test_folder_scanner(docs_dir)
                input("\nPress Enter to continue...")
            elif choice == "4":
                start_web_interface()
            elif choice == "5":
                show_cli_examples()
                input("\nPress Enter to continue...")
            elif choice == "6":
                print("\n🧪 RUNNING COMPLETE INTEGRATION TEST")
                print("-" * 40)
                
                # Run all tests
                if test_system_components():
                    docs_dir = create_sample_documents()
                    if test_folder_scanner(docs_dir):
                        print("\n🎉 INTEGRATION TEST COMPLETED SUCCESSFULLY!")
                        print("📋 System is ready for production use.")
                    else:
                        print("\n❌ Integration test failed at folder scanner")
                else:
                    print("\n❌ Integration test failed at component testing")
                
                input("\nPress Enter to continue...")
            else:
                print("❌ Invalid option. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main startup function"""
    print_banner()
    
    # Quick system check
    print("\n🔍 QUICK SYSTEM CHECK")
    print("-" * 40)
    
    if test_system_components():
        print("\n✅ All systems operational!")
        interactive_menu()
    else:
        print("\n❌ Some components failed. Please check the errors above.")
        print("💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 