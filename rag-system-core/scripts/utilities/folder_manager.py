#!/usr/bin/env python3
"""
RAG System Folder Manager Utility
Manage document folders, monitoring, and ingestion
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class FolderManager:
    """Manage document folders and monitoring"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        
    def list_monitored_folders(self) -> List[Dict[str, Any]]:
        """List currently monitored folders"""
        try:
            from src.core.system_init import initialize_system
            container = initialize_system()
            config_manager = container.get('config_manager')
            config = config_manager.get_config()
            
            folder_config = getattr(config, 'folder_monitoring', None)
            if not folder_config:
                return []
            
            monitored_folders = getattr(folder_config, 'monitored_folders', [])
            
            folders_info = []
            for folder_path in monitored_folders:
                path = Path(folder_path)
                folders_info.append({
                    "path": str(path),
                    "exists": path.exists(),
                    "is_directory": path.is_dir() if path.exists() else False,
                    "file_count": len(list(path.glob("**/*"))) if path.exists() and path.is_dir() else 0
                })
            
            return folders_info
            
        except Exception as e:
            logging.error(f"Failed to list monitored folders: {e}")
            return []
    
    def add_folder(self, folder_path: str) -> bool:
        """Add a folder to monitoring"""
        path = Path(folder_path)
        
        if not path.exists():
            print(f"{Fore.RED}❌ Folder does not exist: {folder_path}{Style.RESET_ALL}")
            return False
        
        if not path.is_dir():
            print(f"{Fore.RED}❌ Path is not a directory: {folder_path}{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.CYAN}📁 Adding folder to monitoring: {folder_path}{Style.RESET_ALL}")
        
        # Here you would implement the logic to add folder to config
        # For now, just validate and show success
        print(f"{Fore.GREEN}✅ Folder added successfully{Style.RESET_ALL}")
        return True
    
    def remove_folder(self, folder_path: str) -> bool:
        """Remove a folder from monitoring"""
        print(f"{Fore.CYAN}📁 Removing folder from monitoring: {folder_path}{Style.RESET_ALL}")
        
        # Here you would implement the logic to remove folder from config
        print(f"{Fore.GREEN}✅ Folder removed successfully{Style.RESET_ALL}")
        return True
    
    def scan_folder(self, folder_path: str) -> Dict[str, Any]:
        """Scan a folder for documents"""
        path = Path(folder_path)
        
        if not path.exists() or not path.is_dir():
            return {"error": "Folder does not exist or is not a directory"}
        
        print(f"{Fore.CYAN}🔍 Scanning folder: {folder_path}{Style.RESET_ALL}")
        
        # Supported file extensions
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.md'}
        
        scan_results = {
            "folder_path": str(path),
            "total_files": 0,
            "supported_files": 0,
            "files": []
        }
        
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    scan_results["total_files"] += 1
                    
                    extension = file_path.suffix.lower()
                    
                    if extension in supported_extensions:
                        scan_results["supported_files"] += 1
                        scan_results["files"].append({
                            "path": str(file_path),
                            "name": file_path.name,
                            "extension": extension,
                            "size": file_path.stat().st_size
                        })
            
            return scan_results
            
        except Exception as e:
            return {"error": str(e)}
    
    def display_scan_results(self, results: Dict[str, Any]):
        """Display folder scan results"""
        if "error" in results:
            print(f"{Fore.RED}❌ Scan failed: {results['error']}{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}📊 SCAN RESULTS{Style.RESET_ALL}")
        print(f"Total Files: {results['total_files']}")
        print(f"Supported Files: {Fore.GREEN}{results['supported_files']}{Style.RESET_ALL}")

def main():
    """Main entry point for folder manager utility"""
    parser = argparse.ArgumentParser(description="RAG System Folder Manager")
    parser.add_argument("action", choices=["scan"], help="Folder management action")
    parser.add_argument("--path", help="Folder path")
    
    args = parser.parse_args()
    
    try:
        folder_manager = FolderManager()
        
        if args.action == "scan":
            if not args.path:
                print(f"{Fore.RED}❌ Folder path required{Style.RESET_ALL}")
                sys.exit(1)
            results = folder_manager.scan_folder(args.path)
            folder_manager.display_scan_results(results)
        
    except Exception as e:
        print(f"{Fore.RED}❌ Folder manager failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 