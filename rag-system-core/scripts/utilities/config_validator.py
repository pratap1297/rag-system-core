#!/usr/bin/env python3
"""
Configuration Validator and Migration Utility
Part of Phase 1.2: Standardize Configuration Management

This utility validates the current configuration setup and provides
migration tools for the enhanced YAML-based configuration system.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import argparse

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables from .env file won't be loaded.")

class ConfigValidator:
    """Configuration validation and migration utility"""
    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.root_dir / "config"
        self.env_file = self.root_dir / ".env"
        
    def validate_current_setup(self) -> Dict[str, Any]:
        """Validate the current configuration setup"""
        print("🔍 Validating Current Configuration Setup")
        print("=" * 50)
        
        validation_results = {
            "env_file": self._validate_env_file(),
            "yaml_configs": self._validate_yaml_configs(),
            "azure_services": self._validate_azure_services(),
            "servicenow": self._validate_servicenow(),
            "api_keys": self._validate_api_keys(),
            "directories": self._validate_directories()
        }
        
        return validation_results
    
    def _validate_env_file(self) -> Dict[str, Any]:
        """Validate .env file"""
        result = {"status": "unknown", "issues": [], "found_vars": []}
        
        if not self.env_file.exists():
            result["status"] = "missing"
            result["issues"].append(".env file not found")
            return result
        
        try:
            with open(self.env_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check for key environment variables
            required_vars = [
                'AZURE_CHATAPI_KEY', 'AZURE_CHAT_ENDPOINT',
                'AZURE_EMBEDDINGS_KEY', 'AZURE_EMBEDDINGS_ENDPOINT',
                'GROQ_API_KEY', 'RAG_API_KEY'
            ]
            
            found_vars = []
            for var in required_vars:
                if f"{var}=" in content:
                    found_vars.append(var)
                    # Check if it has a value
                    value = os.getenv(var, "")
                    if not value:
                        result["issues"].append(f"{var} is defined but empty")
            
            result["found_vars"] = found_vars
            result["status"] = "good" if len(found_vars) >= 4 else "partial"
            
            if len(found_vars) < 4:
                result["issues"].append(f"Only {len(found_vars)}/{len(required_vars)} required variables found")
            
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(f"Error reading .env file: {e}")
        
        return result
    
    def _validate_yaml_configs(self) -> Dict[str, Any]:
        """Validate YAML configuration files"""
        result = {"status": "unknown", "configs": {}, "issues": []}
        
        environments_dir = self.config_dir / "environments"
        
        if not environments_dir.exists():
            result["status"] = "missing"
            result["issues"].append("config/environments directory not found")
            return result
        
        # Check for environment-specific configs
        env_configs = list(environments_dir.glob("*.yaml"))
        
        for config_file in env_configs:
            env_name = config_file.stem
            config_result = {"status": "unknown", "issues": []}
            
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Validate structure
                if not isinstance(config_data, dict):
                    config_result["issues"].append("Invalid YAML structure")
                else:
                    # Check for key sections
                    expected_sections = ['azure', 'llm', 'embedding', 'api']
                    missing_sections = [s for s in expected_sections if s not in config_data]
                    if missing_sections:
                        config_result["issues"].append(f"Missing sections: {missing_sections}")
                
                config_result["status"] = "good" if not config_result["issues"] else "issues"
                
            except Exception as e:
                config_result["status"] = "error"
                config_result["issues"].append(f"Error loading YAML: {e}")
            
            result["configs"][env_name] = config_result
        
        result["status"] = "good" if env_configs else "missing"
        if not env_configs:
            result["issues"].append("No environment-specific YAML configs found")
        
        return result
    
    def _validate_azure_services(self) -> Dict[str, Any]:
        """Validate Azure AI services configuration"""
        result = {"status": "unknown", "services": {}, "issues": []}
        
        # Check Azure Chat API
        chat_key = os.getenv('AZURE_CHATAPI_KEY')
        chat_endpoint = os.getenv('AZURE_CHAT_ENDPOINT')
        chat_model = os.getenv('CHAT_MODEL')
        
        result["services"]["chat"] = {
            "has_key": bool(chat_key),
            "has_endpoint": bool(chat_endpoint),
            "has_model": bool(chat_model),
            "status": "configured" if all([chat_key, chat_endpoint, chat_model]) else "incomplete"
        }
        
        # Check Azure Embeddings
        emb_key = os.getenv('AZURE_EMBEDDINGS_KEY')
        emb_endpoint = os.getenv('AZURE_EMBEDDINGS_ENDPOINT')
        emb_model = os.getenv('EMBEDDING_MODEL')
        
        result["services"]["embeddings"] = {
            "has_key": bool(emb_key),
            "has_endpoint": bool(emb_endpoint),
            "has_model": bool(emb_model),
            "status": "configured" if all([emb_key, emb_endpoint, emb_model]) else "incomplete"
        }
        
        # Check Azure Computer Vision
        vision_key = os.getenv('AZURE_COMPUTER_VISION_KEY')
        vision_endpoint = os.getenv('AZURE_COMPUTER_VISION_ENDPOINT')
        
        result["services"]["computer_vision"] = {
            "has_key": bool(vision_key),
            "has_endpoint": bool(vision_endpoint),
            "status": "configured" if all([vision_key, vision_endpoint]) else "incomplete"
        }
        
        # Overall status
        configured_services = sum(1 for s in result["services"].values() if s["status"] == "configured")
        result["status"] = "good" if configured_services >= 2 else "partial"
        
        if configured_services < 3:
            result["issues"].append(f"Only {configured_services}/3 Azure services fully configured")
        
        return result
    
    def _validate_servicenow(self) -> Dict[str, Any]:
        """Validate ServiceNow configuration"""
        result = {"status": "unknown", "issues": []}
        
        instance = os.getenv('SERVICENOW_INSTANCE')
        username = os.getenv('SERVICENOW_USERNAME')
        password = os.getenv('SERVICENOW_PASSWORD')
        enabled = os.getenv('SERVICENOW_SYNC_ENABLED', 'true').lower() == 'true'
        
        result["enabled"] = enabled
        result["has_instance"] = bool(instance)
        result["has_credentials"] = bool(username and password)
        
        if enabled:
            if not instance:
                result["issues"].append("ServiceNow instance not configured")
            if not (username and password):
                result["issues"].append("ServiceNow credentials not configured")
            
            result["status"] = "good" if not result["issues"] else "incomplete"
        else:
            result["status"] = "disabled"
        
        return result
    
    def _validate_api_keys(self) -> Dict[str, Any]:
        """Validate API keys"""
        result = {"status": "unknown", "keys": {}, "issues": []}
        
        # Check various API keys
        api_keys = {
            "groq": os.getenv('GROQ_API_KEY'),
            "openai": os.getenv('OPENAI_API_KEY'),
            "cohere": os.getenv('COHERE_API_KEY'),
            "rag_api": os.getenv('RAG_API_KEY')
        }
        
        for key_name, key_value in api_keys.items():
            result["keys"][key_name] = {
                "configured": bool(key_value),
                "length": len(key_value) if key_value else 0
            }
        
        configured_keys = sum(1 for k in result["keys"].values() if k["configured"])
        result["status"] = "good" if configured_keys >= 2 else "partial"
        
        if not result["keys"]["rag_api"]["configured"]:
            result["issues"].append("RAG API key not configured (security risk)")
        
        return result
    
    def _validate_directories(self) -> Dict[str, Any]:
        """Validate required directories"""
        result = {"status": "unknown", "directories": {}, "issues": []}
        
        required_dirs = {
            "data": self.root_dir / "data",
            "logs": self.root_dir / "logs",
            "config": self.config_dir,
            "scripts": self.root_dir / "scripts",
            "src": self.root_dir / "src"
        }
        
        for dir_name, dir_path in required_dirs.items():
            exists = dir_path.exists()
            result["directories"][dir_name] = {
                "exists": exists,
                "path": str(dir_path),
                "writable": dir_path.is_dir() and os.access(dir_path, os.W_OK) if exists else False
            }
            
            if not exists:
                result["issues"].append(f"Directory missing: {dir_name} ({dir_path})")
        
        missing_dirs = sum(1 for d in result["directories"].values() if not d["exists"])
        result["status"] = "good" if missing_dirs == 0 else "issues"
        
        return result
    
    def generate_migration_plan(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate a migration plan based on validation results"""
        plan = []
        
        # Check .env file issues
        env_result = validation_results.get("env_file", {})
        if env_result.get("status") != "good":
            plan.append("1. Fix .env file configuration")
            for issue in env_result.get("issues", []):
                plan.append(f"   - {issue}")
        
        # Check YAML configs
        yaml_result = validation_results.get("yaml_configs", {})
        if yaml_result.get("status") != "good":
            plan.append("2. Create environment-specific YAML configurations")
            plan.append("   - Generate development.yaml from current .env")
            plan.append("   - Create production.yaml template")
        
        # Check Azure services
        azure_result = validation_results.get("azure_services", {})
        if azure_result.get("status") != "good":
            plan.append("3. Complete Azure AI services configuration")
            for service, config in azure_result.get("services", {}).items():
                if config.get("status") != "configured":
                    plan.append(f"   - Configure Azure {service} service")
        
        # Check directories
        dir_result = validation_results.get("directories", {})
        if dir_result.get("status") != "good":
            plan.append("4. Create missing directories")
            for issue in dir_result.get("issues", []):
                plan.append(f"   - {issue}")
        
        if not plan:
            plan.append("✅ Configuration is already well-structured!")
            plan.append("   - Consider running config optimization")
        
        return plan
    
    def migrate_env_to_yaml(self, environment: str = "development") -> str:
        """Migrate current .env configuration to YAML format"""
        print(f"🔄 Migrating .env to {environment}.yaml")
        
        # Create YAML config from current environment variables
        config = {
            "environment": environment,
            "debug": os.getenv('RAG_DEBUG', 'true').lower() == 'true',
            "data_dir": "data",
            "log_dir": "logs",
            "azure": {
                "chat_api_key": f"${{AZURE_CHATAPI_KEY}}",
                "chat_endpoint": f"${{AZURE_CHAT_ENDPOINT}}",
                "embeddings_endpoint": f"${{AZURE_EMBEDDINGS_ENDPOINT}}",
                "embeddings_key": f"${{AZURE_EMBEDDINGS_KEY}}",
                "computer_vision_endpoint": f"${{AZURE_COMPUTER_VISION_ENDPOINT}}",
                "computer_vision_key": f"${{AZURE_COMPUTER_VISION_KEY}}",
                "chat_model": f"${{CHAT_MODEL}}",
                "embedding_model": f"${{EMBEDDING_MODEL}}",
                "foundry": {
                    "enabled": True,
                    "workspace_name": "azurehub1910875317"
                },
                "vision": {
                    "enabled": True,
                    "api_version": "2024-02-01"
                }
            },
            "llm": {
                "provider": "groq",
                "groq": {
                    "api_key": "${GROQ_API_KEY}",
                    "model_name": "meta-llama/llama-4-maverick-17b-128e-instruct"
                },
                "azure_llm": {
                    "api_key": "${AZURE_CHATAPI_KEY}",
                    "endpoint": "${AZURE_CHAT_ENDPOINT}",
                    "model_name": "${CHAT_MODEL}"
                }
            },
            "embedding": {
                "provider": "sentence-transformers",
                "sentence_transformers": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "dimension": 384
                },
                "azure": {
                    "api_key": "${AZURE_EMBEDDINGS_KEY}",
                    "endpoint": "${AZURE_EMBEDDINGS_ENDPOINT}",
                    "model_name": "${EMBEDDING_MODEL}"
                }
            },
            "servicenow": {
                "enabled": "${SERVICENOW_SYNC_ENABLED:-true}",
                "instance": "${SERVICENOW_INSTANCE}",
                "username": "${SERVICENOW_USERNAME}",
                "password": "${SERVICENOW_PASSWORD}",
                "sync": {
                    "interval_minutes": "${SERVICENOW_SYNC_INTERVAL:-120}",
                    "max_records": "${SERVICENOW_MAX_RECORDS:-1000}"
                }
            },
            "api": {
                "host": "${RAG_API_HOST:-0.0.0.0}",
                "port": "${RAG_API_PORT:-8000}",
                "api_key": "${RAG_API_KEY}",
                "cors_origins": ["*"]
            }
        }
        
        # Save to YAML file
        output_path = self.config_dir / "environments" / f"{environment}.yaml"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(f"# RAG System - {environment.title()} Environment Configuration\n")
            f.write(f"# Generated from .env file on {os.popen('date').read().strip()}\n\n")
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration migrated to: {output_path}")
        return str(output_path)
    
    def create_config_summary(self) -> str:
        """Create a configuration summary report"""
        validation_results = self.validate_current_setup()
        
        summary = {
            "timestamp": os.popen('date').read().strip(),
            "environment": os.getenv('RAG_ENVIRONMENT', 'development'),
            "validation_results": validation_results,
            "migration_plan": self.generate_migration_plan(validation_results)
        }
        
        output_path = self.root_dir / f"config_validation_report.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False, indent=2)
        
        return str(output_path)
    
    def print_validation_report(self, validation_results: Dict[str, Any]):
        """Print a formatted validation report"""
        print("\n📊 Configuration Validation Report")
        print("=" * 50)
        
        # .env file status
        env_result = validation_results.get("env_file", {})
        status_icon = "✅" if env_result.get("status") == "good" else "⚠️" if env_result.get("status") == "partial" else "❌"
        print(f"{status_icon} .env File: {env_result.get('status', 'unknown').upper()}")
        if env_result.get("issues"):
            for issue in env_result["issues"]:
                print(f"   - {issue}")
        
        # YAML configs status
        yaml_result = validation_results.get("yaml_configs", {})
        status_icon = "✅" if yaml_result.get("status") == "good" else "❌"
        print(f"{status_icon} YAML Configs: {yaml_result.get('status', 'unknown').upper()}")
        if yaml_result.get("issues"):
            for issue in yaml_result["issues"]:
                print(f"   - {issue}")
        
        # Azure services status
        azure_result = validation_results.get("azure_services", {})
        status_icon = "✅" if azure_result.get("status") == "good" else "⚠️"
        print(f"{status_icon} Azure Services: {azure_result.get('status', 'unknown').upper()}")
        for service, config in azure_result.get("services", {}).items():
            service_icon = "✅" if config.get("status") == "configured" else "⚠️"
            print(f"   {service_icon} {service.title()}: {config.get('status', 'unknown')}")
        
        # ServiceNow status
        sn_result = validation_results.get("servicenow", {})
        status_icon = "✅" if sn_result.get("status") == "good" else "⚠️" if sn_result.get("status") == "disabled" else "❌"
        print(f"{status_icon} ServiceNow: {sn_result.get('status', 'unknown').upper()}")
        
        # API keys status
        keys_result = validation_results.get("api_keys", {})
        status_icon = "✅" if keys_result.get("status") == "good" else "⚠️"
        print(f"{status_icon} API Keys: {keys_result.get('status', 'unknown').upper()}")
        configured_keys = sum(1 for k in keys_result.get("keys", {}).values() if k.get("configured"))
        total_keys = len(keys_result.get("keys", {}))
        print(f"   - {configured_keys}/{total_keys} keys configured")
        
        # Directories status
        dir_result = validation_results.get("directories", {})
        status_icon = "✅" if dir_result.get("status") == "good" else "❌"
        print(f"{status_icon} Directories: {dir_result.get('status', 'unknown').upper()}")
        
        print("\n🔧 Migration Plan:")
        print("-" * 30)
        migration_plan = self.generate_migration_plan(validation_results)
        for step in migration_plan:
            print(step)

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="RAG System Configuration Validator")
    parser.add_argument("--validate", action="store_true", help="Validate current configuration")
    parser.add_argument("--migrate", action="store_true", help="Migrate .env to YAML")
    parser.add_argument("--environment", default="development", help="Environment name for migration")
    parser.add_argument("--summary", action="store_true", help="Create configuration summary")
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    if args.validate or not any([args.migrate, args.summary]):
        # Default action: validate
        validation_results = validator.validate_current_setup()
        validator.print_validation_report(validation_results)
    
    if args.migrate:
        validator.migrate_env_to_yaml(args.environment)
    
    if args.summary:
        summary_path = validator.create_config_summary()
        print(f"📄 Configuration summary saved to: {summary_path}")

if __name__ == "__main__":
    main() 