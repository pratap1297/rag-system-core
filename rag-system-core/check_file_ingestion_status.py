#!/usr/bin/env python3
"""
Check File Ingestion Status
Check why files are being tracked but not ingested
"""

import requests
import json
from datetime import datetime

def check_file_ingestion_status():
    """Check detailed file ingestion status"""
    
    print("🔍 Checking File Ingestion Status")
    print("=" * 50)
    
    api_url = "http://localhost:8000"
    
    # Check folder monitor status
    print("\n1️⃣ Folder Monitor Status:")
    try:
        response = requests.get(f"{api_url}/folder-monitor/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                status = data.get('status', {})
                print(f"   🔄 Status: {'🟢 Running' if status.get('is_running') else '🔴 Stopped'}")
                print(f"   📁 Monitored Folders: {len(status.get('monitored_folders', []))}")
                print(f"   📄 Files Tracked: {status.get('total_files_tracked', 0)}")
                print(f"   ✅ Files Ingested: {status.get('files_ingested', 0)}")
                print(f"   ❌ Files Failed: {status.get('files_failed', 0)}")
                print(f"   ⏳ Files Pending: {status.get('files_pending', 0)}")
                print(f"   📊 Total Scans: {status.get('scan_count', 0)}")
                print(f"   ⏱️ Check Interval: {status.get('check_interval', 0)} seconds")
                print(f"   🔄 Auto-Ingest: {'✅ Enabled' if status.get('auto_ingest', False) else '❌ Disabled'}")
            else:
                print(f"   ❌ Error: {data.get('error')}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
        return
    
    # Check detailed file status
    print("\n2️⃣ Detailed File Status:")
    try:
        response = requests.get(f"{api_url}/folder-monitor/files", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                files = data.get('files', {})
                
                if not files:
                    print("   📄 No files are currently being tracked")
                    return
                
                print(f"   📄 Total Files Tracked: {len(files)}")
                
                # Group files by status
                status_groups = {}
                for file_path, file_info in files.items():
                    status = file_info.get('ingestion_status', 'unknown')
                    if status not in status_groups:
                        status_groups[status] = []
                    status_groups[status].append((file_path, file_info))
                
                # Display by status
                for status, file_list in status_groups.items():
                    icon = {'success': '✅', 'pending': '⏳', 'failed': '❌', 'unknown': '❓'}.get(status, '❓')
                    print(f"\n   {icon} {status.upper()} ({len(file_list)} files):")
                    
                    for file_path, file_info in file_list:
                        filename = file_path.split('\\')[-1] if '\\' in file_path else file_path.split('/')[-1]
                        print(f"      📁 {filename}")
                        print(f"         Path: {file_path}")
                        print(f"         Size: {file_info.get('size', 0)} bytes")
                        print(f"         Last Modified: {file_info.get('last_modified', 'Unknown')}")
                        
                        if file_info.get('error_message'):
                            print(f"         ❌ Error: {file_info.get('error_message')}")
                        
                        if file_info.get('ingestion_attempts'):
                            print(f"         🔄 Ingestion Attempts: {file_info.get('ingestion_attempts')}")
                        
                        if file_info.get('last_ingestion_attempt'):
                            print(f"         🕐 Last Attempt: {file_info.get('last_ingestion_attempt')}")
                        
                        print()
            else:
                print(f"   ❌ Error: {data.get('error')}")
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection Error: {e}")
    
    # Check if auto-ingest is working
    print("\n3️⃣ Auto-Ingest Configuration:")
    try:
        # Check system configuration
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            components = health_data.get('components', {})
            
            folder_monitor_status = components.get('folder_monitor', {})
            print(f"   📁 Folder Monitor Component: {folder_monitor_status.get('status', 'Unknown')}")
            
            ingestion_engine_status = components.get('ingestion_engine', {})
            print(f"   ⚙️ Ingestion Engine Component: {ingestion_engine_status.get('status', 'Unknown')}")
            
        else:
            print(f"   ❌ Could not check component status: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error checking components: {e}")
    
    # Force a scan to see if that helps
    print("\n4️⃣ Forcing Manual Scan:")
    try:
        response = requests.post(f"{api_url}/folder-monitor/scan", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   🔍 Scan completed successfully")
                print(f"   📊 Changes detected: {data.get('changes_detected', 0)}")
                print(f"   📄 Files tracked: {data.get('files_tracked', 0)}")
                
                if data.get('ingestion_results'):
                    results = data.get('ingestion_results')
                    print(f"   ✅ Files ingested: {results.get('success', 0)}")
                    print(f"   ❌ Files failed: {results.get('failed', 0)}")
                    
                    if results.get('errors'):
                        print(f"   🚨 Ingestion errors:")
                        for error in results.get('errors', []):
                            print(f"      - {error}")
            else:
                print(f"   ❌ Scan failed: {data.get('error')}")
        else:
            print(f"   ❌ Scan request failed: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error forcing scan: {e}")

if __name__ == "__main__":
    check_file_ingestion_status() 