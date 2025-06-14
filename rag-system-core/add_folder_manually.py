#!/usr/bin/env python3
"""
Add Folder Manually
Script to manually add a folder to monitoring if UI has issues
"""
import requests
import json

def add_folder_manually():
    """Manually add a folder to monitoring"""
    api_url = "http://localhost:8000"
    
    # Get folder path from user
    folder_path = input("Enter the full path to the folder you want to monitor: ").strip()
    
    if not folder_path:
        print("❌ No folder path provided")
        return
    
    print(f"📁 Adding folder to monitoring: {folder_path}")
    
    try:
        # Add folder
        response = requests.post(
            f"{api_url}/folder-monitor/add",
            json={"folder_path": folder_path},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Folder added successfully!")
                print(f"📊 Files found: {result.get('files_found', 0)}")
                
                if result.get('immediate_scan'):
                    print(f"📊 Immediate scan: {result.get('changes_detected', 0)} changes, {result.get('files_tracked', 0)} files tracked")
                
                # Show updated status
                print("\n📊 Updated Status:")
                status_response = requests.get(f"{api_url}/folder-monitor/status")
                if status_response.status_code == 200:
                    status = status_response.json()
                    if status.get('success'):
                        folders = status.get('status', {}).get('monitored_folders', [])
                        print(f"📁 Total folders monitored: {len(folders)}")
                        for i, folder in enumerate(folders, 1):
                            print(f"   {i}. {folder}")
                            
            else:
                print(f"❌ Failed to add folder: {result}")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    add_folder_manually() 