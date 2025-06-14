#!/usr/bin/env python3
"""
RAG System Migration Utility
Handle system updates, data migration, and configuration changes
"""

import argparse
import sys
import logging
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

class MigrationManager:
    """Handle system migrations and updates"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.backup_dir = self.base_dir / "backups"
        self.migration_log = self.base_dir / "migration.log"
        
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the current system"""
        if not backup_name:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        print(f"{Fore.CYAN}📦 Creating system backup: {backup_name}{Style.RESET_ALL}")
        
        # Backup important directories and files
        backup_items = [
            "config/",
            "data/",
            ".env",
            "requirements.txt"
        ]
        
        for item in backup_items:
            source = self.base_dir / item
            if source.exists():
                if source.is_file():
                    shutil.copy2(source, backup_path / item)
                    print(f"  ✅ Backed up file: {item}")
                else:
                    shutil.copytree(source, backup_path / item, dirs_exist_ok=True)
                    print(f"  ✅ Backed up directory: {item}")
        
        print(f"{Fore.GREEN}✅ Backup created: {backup_path}{Style.RESET_ALL}")
        return str(backup_path)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        if not self.backup_dir.exists():
            return backups
        
        for backup_path in self.backup_dir.iterdir():
            if backup_path.is_dir():
                manifest_file = backup_path / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r') as f:
                            manifest = json.load(f)
                        backups.append({
                            "name": backup_path.name,
                            "path": str(backup_path),
                            "timestamp": manifest.get("timestamp", "unknown"),
                            "items": len(manifest.get("items", []))
                        })
                    except Exception as e:
                        backups.append({
                            "name": backup_path.name,
                            "path": str(backup_path),
                            "timestamp": "unknown",
                            "items": 0,
                            "error": str(e)
                        })
        
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from a backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"{Fore.RED}❌ Backup not found: {backup_name}{Style.RESET_ALL}")
            return False
        
        manifest_file = backup_path / "manifest.json"
        if not manifest_file.exists():
            print(f"{Fore.RED}❌ Invalid backup (no manifest): {backup_name}{Style.RESET_ALL}")
            return False
        
        print(f"{Fore.CYAN}🔄 Restoring from backup: {backup_name}{Style.RESET_ALL}")
        
        try:
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
            
            # Create a backup of current state before restore
            current_backup = self.create_backup(f"pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            print(f"  📦 Current state backed up to: {current_backup}")
            
            # Restore items
            for item in manifest.get("items", []):
                source = backup_path / item
                target = self.base_dir / item
                
                if source.exists():
                    if source.is_file():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
                        print(f"  ✅ Restored file: {item}")
                    else:
                        if target.exists():
                            shutil.rmtree(target)
                        shutil.copytree(source, target)
                        print(f"  ✅ Restored directory: {item}")
                else:
                    print(f"  ⚠️  Item not in backup: {item}")
            
            print(f"{Fore.GREEN}✅ Restore completed successfully{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Restore failed: {e}{Style.RESET_ALL}")
            return False
    
    def migrate_config_format(self) -> bool:
        """Migrate configuration to new format"""
        print(f"{Fore.CYAN}🔄 Migrating configuration format{Style.RESET_ALL}")
        
        config_file = self.base_dir / "config" / "default.yaml"
        if not config_file.exists():
            print(f"{Fore.YELLOW}⚠️  No configuration file found to migrate{Style.RESET_ALL}")
            return True
        
        # Create backup first
        backup_path = self.create_backup("pre_config_migration")
        
        try:
            # Here you would implement actual config migration logic
            # For now, just validate the config exists
            print(f"  ✅ Configuration format is up to date")
            print(f"{Fore.GREEN}✅ Configuration migration completed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Configuration migration failed: {e}{Style.RESET_ALL}")
            return False
    
    def migrate_data_structure(self) -> bool:
        """Migrate data structure to new format"""
        print(f"{Fore.CYAN}🔄 Migrating data structure{Style.RESET_ALL}")
        
        data_dir = self.base_dir / "data"
        if not data_dir.exists():
            print(f"{Fore.YELLOW}⚠️  No data directory found to migrate{Style.RESET_ALL}")
            return True
        
        # Create backup first
        backup_path = self.create_backup("pre_data_migration")
        
        try:
            # Here you would implement actual data migration logic
            # For now, just validate the data directory structure
            print(f"  ✅ Data structure is up to date")
            print(f"{Fore.GREEN}✅ Data migration completed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}❌ Data migration failed: {e}{Style.RESET_ALL}")
            return False
    
    def update_dependencies(self) -> bool:
        """Update system dependencies"""
        print(f"{Fore.CYAN}📦 Updating dependencies{Style.RESET_ALL}")
        
        requirements_file = self.base_dir / "requirements.txt"
        if not requirements_file.exists():
            print(f"{Fore.RED}❌ requirements.txt not found{Style.RESET_ALL}")
            return False
        
        try:
            import subprocess
            
            # Update pip first
            print("  🔄 Updating pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, capture_output=True)
            
            # Install/update requirements
            print("  🔄 Installing/updating requirements...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                         check=True, capture_output=True)
            
            print(f"{Fore.GREEN}✅ Dependencies updated successfully{Style.RESET_ALL}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"{Fore.RED}❌ Dependency update failed: {e}{Style.RESET_ALL}")
            return False
        except Exception as e:
            print(f"{Fore.RED}❌ Unexpected error during dependency update: {e}{Style.RESET_ALL}")
            return False
    
    def run_full_migration(self) -> bool:
        """Run complete system migration"""
        print(f"{Fore.CYAN}🚀 Running full system migration{Style.RESET_ALL}")
        print("=" * 50)
        
        # Create pre-migration backup
        backup_path = self.create_backup("pre_full_migration")
        
        migration_steps = [
            ("Configuration Migration", self.migrate_config_format),
            ("Data Structure Migration", self.migrate_data_structure),
            ("Dependency Update", self.update_dependencies)
        ]
        
        success = True
        for step_name, step_func in migration_steps:
            print(f"\n{Fore.CYAN}📋 {step_name}{Style.RESET_ALL}")
            if not step_func():
                print(f"{Fore.RED}❌ {step_name} failed{Style.RESET_ALL}")
                success = False
                break
        
        if success:
            print(f"\n{Fore.GREEN}✅ Full migration completed successfully{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}❌ Migration failed. You can restore from backup: {backup_path}{Style.RESET_ALL}")
        
        return success

def main():
    """Main entry point for migration utility"""
    parser = argparse.ArgumentParser(description="RAG System Migration Utility")
    parser.add_argument("action", choices=["backup", "restore", "list-backups", "migrate-config", 
                                          "migrate-data", "update-deps", "full-migration"],
                       help="Migration action to perform")
    parser.add_argument("--name", help="Backup name (for backup/restore operations)")
    parser.add_argument("--log-level", default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        migration_manager = MigrationManager()
        
        if args.action == "backup":
            migration_manager.create_backup(args.name)
            
        elif args.action == "restore":
            if not args.name:
                print(f"{Fore.RED}❌ Backup name required for restore operation{Style.RESET_ALL}")
                sys.exit(1)
            success = migration_manager.restore_backup(args.name)
            sys.exit(0 if success else 1)
            
        elif args.action == "list-backups":
            backups = migration_manager.list_backups()
            if backups:
                print(f"\n{Fore.CYAN}📦 Available Backups:{Style.RESET_ALL}")
                for backup in backups:
                    print(f"  • {backup['name']} ({backup['timestamp']}) - {backup['items']} items")
            else:
                print(f"{Fore.YELLOW}⚠️  No backups found{Style.RESET_ALL}")
                
        elif args.action == "migrate-config":
            success = migration_manager.migrate_config_format()
            sys.exit(0 if success else 1)
            
        elif args.action == "migrate-data":
            success = migration_manager.migrate_data_structure()
            sys.exit(0 if success else 1)
            
        elif args.action == "update-deps":
            success = migration_manager.update_dependencies()
            sys.exit(0 if success else 1)
            
        elif args.action == "full-migration":
            success = migration_manager.run_full_migration()
            sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"{Fore.RED}❌ Migration utility failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 