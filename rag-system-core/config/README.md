# RAG System Configuration Management

## Overview

The RAG System uses a sophisticated configuration management system that supports:

- **Environment-specific YAML configurations** for different deployment environments
- **Environment variable expansion** for secure credential management
- **Azure AI Services integration** with comprehensive service configuration
- **ServiceNow integration** with sync and connection settings
- **Multi-provider LLM and embedding support** (Groq, Azure, OpenAI, Cohere)
- **Backward compatibility** with existing `.env` files

## Configuration Structure

### Directory Layout

```
config/
├── environments/           # Environment-specific configurations
│   ├── development.yaml   # Development environment
│   ├── production.yaml    # Production environment
│   └── staging.yaml       # Staging environment (optional)
├── config_manager.py      # Configuration management utilities
└── README.md              # This documentation
```

### Configuration Hierarchy

1. **YAML Configuration Files** (Primary)
   - Environment-specific settings
   - Structured, hierarchical configuration
   - Environment variable expansion support

2. **Environment Variables** (Override)
   - Direct environment variable overrides
   - Backward compatibility with existing `.env` setup
   - Secure credential management

3. **Default Values** (Fallback)
   - Built-in defaults for all configuration options
   - Ensures system can start with minimal configuration

## Environment-Specific Configurations

### Development Environment (`development.yaml`)

```yaml
environment: development
debug: true
data_dir: data
log_dir: logs

azure:
  chat_api_key: "${AZURE_CHATAPI_KEY}"
  chat_endpoint: "${AZURE_CHAT_ENDPOINT}"
  # ... other Azure settings

llm:
  provider: groq  # Primary provider for development
  groq:
    api_key: "${GROQ_API_KEY}"
    model_name: "meta-llama/llama-4-maverick-17b-128e-instruct"
  # ... other LLM providers

# ... other configurations
```

### Production Environment (`production.yaml`)

```yaml
environment: production
debug: false
data_dir: /app/data
log_dir: /app/logs

azure:
  # Production Azure configuration
  # Enhanced security and performance settings

llm:
  provider: azure  # Prefer Azure in production
  # ... production-optimized settings

# ... production-specific configurations
```

## Configuration Sections

### 1. Azure AI Services Configuration

```yaml
azure:
  # API Keys and Endpoints
  chat_api_key: "${AZURE_CHATAPI_KEY}"
  chat_endpoint: "${AZURE_CHAT_ENDPOINT}"
  embeddings_key: "${AZURE_EMBEDDINGS_KEY}"
  embeddings_endpoint: "${AZURE_EMBEDDINGS_ENDPOINT}"
  computer_vision_key: "${AZURE_COMPUTER_VISION_KEY}"
  computer_vision_endpoint: "${AZURE_COMPUTER_VISION_ENDPOINT}"
  
  # Model Names
  chat_model: "${CHAT_MODEL}"
  embedding_model: "${EMBEDDING_MODEL}"
  
  # Azure AI Foundry Settings
  foundry:
    enabled: true
    workspace_name: "azurehub1910875317"
    resource_group: "default"
    subscription_id: "${AZURE_SUBSCRIPTION_ID:-}"
    
  # Azure Vision Read 4.0 (OCR)
  vision:
    enabled: true
    api_version: "2024-02-01"
    read_timeout: 30
    max_retries: 3
```

### 2. LLM Provider Configuration

```yaml
llm:
  provider: groq  # Active provider: groq, azure, openai, cohere
  
  # Groq Configuration
  groq:
    api_key: "${GROQ_API_KEY}"
    model_name: "meta-llama/llama-4-maverick-17b-128e-instruct"
    temperature: 0.1
    max_tokens: 1000
    timeout: 30
  
  # Azure LLM Configuration
  azure_llm:
    api_key: "${AZURE_CHATAPI_KEY}"
    endpoint: "${AZURE_CHAT_ENDPOINT}"
    model_name: "${CHAT_MODEL}"
    api_version: "2024-02-15-preview"
    temperature: 0.1
    max_tokens: 1000
```

### 3. Embedding Provider Configuration

```yaml
embedding:
  provider: sentence-transformers  # Active provider
  
  # Sentence Transformers (Local)
  sentence_transformers:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    dimension: 384
    device: cpu
    batch_size: 32
  
  # Azure Embeddings
  azure:
    api_key: "${AZURE_EMBEDDINGS_KEY}"
    endpoint: "${AZURE_EMBEDDINGS_ENDPOINT}"
    model_name: "${EMBEDDING_MODEL}"
    dimension: 1024
    api_version: "2024-02-15-preview"
```

### 4. ServiceNow Integration

```yaml
servicenow:
  enabled: "${SERVICENOW_SYNC_ENABLED:-true}"
  instance: "${SERVICENOW_INSTANCE}"
  username: "${SERVICENOW_USERNAME}"
  password: "${SERVICENOW_PASSWORD}"
  table: "${SERVICENOW_TABLE:-incident}"
  
  # Sync Configuration
  sync:
    enabled: true
    interval_minutes: "${SERVICENOW_SYNC_INTERVAL:-120}"
    max_records: "${SERVICENOW_MAX_RECORDS:-1000}"
    fields: "${SERVICENOW_FIELDS}"
    query_filter: "${SERVICENOW_QUERY_FILTER:-state!=7}"
    date_field: "${SERVICENOW_DATE_FIELD:-sys_updated_on}"
  
  # Connection Settings
  connection:
    timeout: 30
    max_retries: 3
    verify_ssl: true
```

### 5. API Server Configuration

```yaml
api:
  host: "${RAG_API_HOST:-0.0.0.0}"
  port: "${RAG_API_PORT:-8000}"
  workers: 1
  reload: true  # Development mode
  cors_origins: ["*"]  # Development - allow all origins
  api_key: "${RAG_API_KEY}"
  
  # Rate Limiting
  rate_limit:
    enabled: false  # Development
    requests_per_minute: 1000
```

## Environment Variable Expansion

The configuration system supports environment variable expansion using the syntax:

- `${VARIABLE_NAME}` - Required variable (will be empty string if not set)
- `${VARIABLE_NAME:-default_value}` - Optional variable with default value

### Examples

```yaml
# Required environment variable
api_key: "${GROQ_API_KEY}"

# Optional with default
port: "${RAG_API_PORT:-8000}"

# Optional with empty default
optional_setting: "${OPTIONAL_VAR:-}"
```

## Configuration Management Commands

### Using the Unified Entry Point

```bash
# Validate current configuration
./rag-system config --validate

# Migrate .env to YAML
./rag-system config --migrate --environment development

# Create configuration summary
./rag-system config --summary
```

### Using Direct Scripts

```bash
# Validate configuration
python scripts/utilities/config_validator.py --validate

# Migrate configuration
python scripts/utilities/config_validator.py --migrate --environment production

# Create summary report
python scripts/utilities/config_validator.py --summary
```

## Configuration Validation

The system includes comprehensive configuration validation that checks:

### ✅ .env File Validation
- Presence and readability of `.env` file
- Required environment variables
- Variable value validation

### ✅ YAML Configuration Validation
- YAML syntax and structure
- Required configuration sections
- Environment-specific configurations

### ✅ Azure Services Validation
- API key presence and format
- Endpoint configuration
- Model name validation
- Service-specific settings

### ✅ ServiceNow Integration Validation
- Instance configuration
- Credential validation
- Sync settings validation

### ✅ API Keys Validation
- Provider-specific API keys
- Security key configuration
- Key format validation

### ✅ Directory Structure Validation
- Required directories existence
- Write permissions
- Path validation

## Migration from .env to YAML

The system provides automatic migration from existing `.env` files to structured YAML configurations:

### Migration Process

1. **Analyze Current .env**: Reads and validates existing environment variables
2. **Generate YAML Structure**: Creates structured YAML with proper sections
3. **Preserve Environment Variables**: Uses `${VAR}` syntax to maintain .env compatibility
4. **Create Environment-Specific Files**: Generates development, production configurations
5. **Validate Generated Config**: Ensures the new configuration is valid

### Migration Command

```bash
# Migrate to development environment
./rag-system config --migrate --environment development

# Migrate to production environment
./rag-system config --migrate --environment production
```

## Best Practices

### 1. Environment Management

- **Development**: Use `development.yaml` with debug enabled, relaxed security
- **Production**: Use `production.yaml` with enhanced security, performance optimization
- **Staging**: Create `staging.yaml` for testing production-like settings

### 2. Security

- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Rotate API keys** regularly
- **Use different keys** for different environments

### 3. Configuration Organization

- **Group related settings** in logical sections
- **Use descriptive names** for configuration options
- **Document complex settings** with comments
- **Validate configurations** before deployment

### 4. Deployment

- **Set environment variables** before starting the system
- **Validate configuration** in CI/CD pipelines
- **Use environment-specific** YAML files
- **Monitor configuration** changes

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   ```bash
   # Check which variables are missing
   ./rag-system config --validate
   ```

2. **YAML Syntax Errors**
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('config/environments/development.yaml'))"
   ```

3. **Azure Service Configuration**
   ```bash
   # Test Azure connectivity
   ./rag-system diagnostics
   ```

4. **ServiceNow Connection Issues**
   ```bash
   # Validate ServiceNow settings
   ./rag-system config --validate
   ```

### Configuration Recovery

If configuration becomes corrupted:

1. **Backup Current Config**: Always backup before changes
2. **Regenerate from .env**: Use migration tool to recreate YAML
3. **Validate After Recovery**: Run validation to ensure correctness
4. **Test System Startup**: Verify system starts with new configuration

## Integration with Existing Code

The configuration system integrates seamlessly with existing code:

```python
# Using the enhanced configuration manager
from config.config_manager import ConfigManager

# Initialize configuration
config_manager = ConfigManager(environment='development')

# Get Azure configuration
azure_config = config_manager.get_azure_config()
chat_api_key = azure_config.get('chat_api_key')

# Get LLM configuration
llm_config = config_manager.get_llm_config()
provider = llm_config.get('provider')

# Validate configuration
issues = config_manager.validate_config()
if issues:
    print("Configuration issues:", issues)
```

## Future Enhancements

- **Configuration Templates**: Pre-built templates for common deployments
- **Dynamic Configuration**: Runtime configuration updates
- **Configuration Encryption**: Encrypted configuration files
- **Configuration Versioning**: Track configuration changes over time
- **Configuration UI**: Web-based configuration management interface 