# RAG System Scripts

This directory contains organized entry points and utilities for the RAG system.

## Structure

```
scripts/
├── __init__.py              # Package initialization
├── start_api.py             # API server startup
├── start_ui.py              # UI interface startup  
├── start_system.py          # Full system startup
├── utilities/               # System utilities
│   ├── __init__.py
│   ├── health_check.py      # Health monitoring
│   ├── diagnostics.py       # System diagnostics
│   ├── migration.py         # Data migration tools
│   └── folder_manager.py    # Folder management
└── README.md               # This file
```

## Entry Points

### 1. API Server (`start_api.py`)
Start only the API server without UI components.

```bash
python scripts/start_api.py --host 0.0.0.0 --port 8000
```

**Options:**
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--workers`: Number of worker processes (default: 1)
- `--reload`: Enable auto-reload for development
- `--log-level`: Log level (debug, info, warning, error)
- `--log-file`: Log file path (optional)

### 2. UI Interface (`start_ui.py`)
Start various UI interfaces.

```bash
python scripts/start_ui.py --ui gradio --port 7860
```

**Options:**
- `--ui`: UI type (gradio, servicenow, comprehensive, enhanced)
- `--port`: Port to bind to (default: 7860)
- `--share`: Share UI publicly
- `--multiple`: Launch multiple UIs
- `--interactive`: Show interactive menu

**Examples:**
```bash
# Start Gradio UI
python scripts/start_ui.py --ui gradio --port 7860

# Start multiple UIs
python scripts/start_ui.py --multiple gradio servicenow --ports 7860 7861

# Interactive menu
python scripts/start_ui.py --interactive
```

### 3. Full System (`start_system.py`)
Start complete system with API and UI.

```bash
python scripts/start_system.py --mode full
```

**Options:**
- `--mode`: Startup mode (full, api, ui)
- `--api-port`: API server port (default: 8000)
- `--ui-type`: UI interface type (gradio, servicenow)
- `--ui-port`: UI port (default: 7860)
- `--share`: Share UI publicly
- `--interactive`: Show interactive menu

**Examples:**
```bash
# Full system with defaults
python scripts/start_system.py --mode full

# API only
python scripts/start_system.py --mode api --api-port 8000

# Interactive startup menu
python scripts/start_system.py --interactive
```

## Utilities

### 1. Health Check (`utilities/health_check.py`)
Monitor system health and performance.

```bash
python scripts/utilities/health_check.py --mode basic
```

**Options:**
- `--url`: Base URL of the RAG system API
- `--mode`: Health check mode (basic)
- `--api-key`: API key for authentication

### 2. System Diagnostics (`utilities/diagnostics.py`)
Comprehensive system diagnostics and troubleshooting.

```bash
python scripts/utilities/diagnostics.py --output console
```

**Options:**
- `--output`: Output format (console, json)
- `--save`: Save diagnostics to file
- `--log-level`: Log level

**Features:**
- Environment configuration check
- Python dependencies verification
- System resource monitoring
- Port availability check
- File system validation

### 3. Migration Tools (`utilities/migration.py`)
Handle system updates and data migration.

```bash
python scripts/utilities/migration.py backup --name my-backup
```

**Actions:**
- `backup`: Create system backup
- `restore`: Restore from backup (future)
- `migrate-config`: Migrate configuration (future)
- `update-deps`: Update dependencies (future)

### 4. Folder Manager (`utilities/folder_manager.py`)
Manage document folders and monitoring.

```bash
python scripts/utilities/folder_manager.py scan --path /path/to/documents
```

**Actions:**
- `scan`: Scan folder for supported documents

**Features:**
- Document type detection
- File count and size analysis
- Supported format validation

## Main Entry Point

The main `rag-system` script provides a unified command-line interface:

```bash
# Start full system
./rag-system start --mode full

# Health check
./rag-system health --mode basic

# System diagnostics
./rag-system diagnostics

# Scan documents
./rag-system folder scan --path /path/to/docs

# Create backup
./rag-system migrate backup --name my-backup
```

## Migration from Old Scripts

The new structure replaces these scattered files:
- `main.py` → `scripts/start_system.py`
- `launch_ui.py` → `scripts/start_ui.py`
- `start_system.py` → `scripts/start_system.py`
- `health_check_cli.py` → `scripts/utilities/health_check.py`

## Benefits

1. **Organized Structure**: Clear separation of concerns
2. **Unified Interface**: Single entry point with subcommands
3. **Modular Design**: Independent utilities
4. **Better Documentation**: Clear usage examples
5. **Enhanced Features**: More options and flexibility
6. **Easier Maintenance**: Centralized script management

## Usage Examples

### Development Workflow
```bash
# Start API server with auto-reload
python scripts/start_api.py --reload --log-level debug

# Start UI for testing
python scripts/start_ui.py --ui gradio --share
```

### Production Deployment
```bash
# Start full system
python scripts/start_system.py --mode full --api-port 8000 --ui-port 7860

# Health monitoring
python scripts/utilities/health_check.py --mode basic
```

### System Maintenance
```bash
# Create backup before updates
python scripts/utilities/migration.py backup --name pre-update

# Run diagnostics
python scripts/utilities/diagnostics.py --save diagnostics.json

# Scan new document folders
python scripts/utilities/folder_manager.py scan --path /new/documents
``` 