# 🏗️ RAG System Core - Production Ready

## 📋 **Overview**

This is a **production-ready RAG (Retrieval-Augmented Generation) System** with comprehensive features including:

- ✅ **Multi-Provider AI/ML Support**: OpenAI, Groq, Cohere, Sentence Transformers
- ✅ **Advanced Document Processing**: PDF, DOCX, TXT with semantic chunking
- ✅ **Vector Database**: FAISS with persistent metadata storage
- ✅ **Web UI**: Multiple Gradio interfaces (Basic, Enhanced, ServiceNow)
- ✅ **API Layer**: FastAPI with comprehensive endpoints
- ✅ **Monitoring**: Real-time health checks and system monitoring
- ✅ **ServiceNow Integration**: Enterprise ticket processing
- ✅ **Folder Monitoring**: Automatic document ingestion

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Clone or copy this directory
cd rag-system-core

# Install dependencies
pip install -r requirements.txt

# Optional: Install UI dependencies
pip install -r requirements_ui.txt
```

### **2. Configuration**

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your API keys
# Required: COHERE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY
```

### **3. Launch Options**

#### **Option A: Fixed UI (Recommended)**
```bash
python launch_fixed_ui.py
```

#### **Option B: Comprehensive UI**
```bash
python launch_comprehensive_ui.py
```

#### **Option C: Basic System**
```bash
python main.py
```

#### **Option D: Enhanced UI v2**
```bash
python launch_enhanced_ui_v2.py
```

## 📁 **Directory Structure**

```
rag-system-core/
├── 📁 src/                          # Core source code
│   ├── 📁 api/                      # FastAPI application & UI
│   │   ├── main.py                  # Main API server (52KB)
│   │   ├── gradio_ui.py             # Main Gradio interface
│   │   ├── servicenow_ui.py         # ServiceNow integration UI
│   │   ├── management_api.py        # Management endpoints
│   │   ├── 📁 models/               # Request/Response models
│   │   └── 📁 routes/               # API route handlers
│   ├── 📁 core/                     # Core system components
│   │   ├── dependency_container.py  # Dependency injection
│   │   ├── config_manager.py        # Configuration management
│   │   ├── system_init.py           # System initialization
│   │   ├── error_handling.py        # Error handling framework
│   │   └── constants.py             # System constants
│   ├── 📁 storage/                  # Data storage layer
│   │   ├── faiss_store.py           # Vector database operations
│   │   ├── metadata_store.py        # Metadata persistence
│   │   └── persistent_metadata_store.py
│   ├── 📁 ingestion/                # Document processing
│   │   ├── ingestion_engine.py      # Main ingestion orchestrator
│   │   ├── embedder.py              # Text embedding
│   │   ├── chunker.py               # Text chunking
│   │   └── semantic_chunker.py      # Semantic chunking
│   ├── 📁 retrieval/                # Query & retrieval
│   │   ├── query_engine.py          # Main query processor
│   │   ├── query_enhancer.py        # Query enhancement
│   │   ├── llm_client.py            # LLM integration
│   │   └── reranker.py              # Result reranking
│   ├── 📁 monitoring/               # System monitoring
│   │   ├── heartbeat_monitor.py     # Health monitoring
│   │   └── folder_monitor.py        # Folder monitoring
│   ├── 📁 integrations/             # External integrations
│   │   └── 📁 servicenow/           # ServiceNow integration
│   └── 📁 ui/                       # UI components
├── 📁 data/                         # Data storage
├── 📁 logs/                         # System logs
├── 📁 config/                       # Configuration files
├── main.py                          # Main entry point
├── start.py                         # System startup
├── requirements.txt                 # Python dependencies
├── .env.template                    # Environment template
└── README.md                        # This file
```

## 🔧 **Core Features**

### **1. Document Processing**
- **Supported Formats**: PDF, DOCX, TXT, MD
- **Chunking Strategies**: Fixed-size, semantic, sentence-based
- **Metadata Extraction**: Title, author, creation date, file info

### **2. Vector Search**
- **FAISS Integration**: High-performance similarity search
- **Multiple Embeddings**: Cohere, Sentence Transformers
- **Persistent Storage**: Metadata and vector persistence

### **3. Query Processing**
- **Query Enhancement**: Automatic query expansion and refinement
- **Multi-Provider LLM**: OpenAI, Groq support
- **Result Reranking**: Improved relevance scoring

### **4. Web Interfaces**
- **Fixed UI**: Stable, production-ready interface
- **Comprehensive UI**: Full-featured management interface
- **ServiceNow UI**: Enterprise integration interface

### **5. System Monitoring**
- **Health Checks**: Real-time system health monitoring
- **Performance Metrics**: Response times, resource usage
- **Folder Monitoring**: Automatic document detection

## 🛠️ **API Endpoints**

### **Core Endpoints**
- `GET /health` - System health check
- `POST /query` - Process queries
- `POST /ingest` - Ingest text content
- `POST /upload` - Upload files
- `GET /documents` - List documents
- `GET /stats` - System statistics

### **Management Endpoints**
- `GET /manage/documents` - Document management
- `GET /manage/vectors` - Vector management
- `POST /manage/cleanup` - System cleanup
- `DELETE /documents/{path}` - Delete documents

### **Monitoring Endpoints**
- `GET /heartbeat` - Heartbeat status
- `POST /heartbeat/start` - Start monitoring
- `GET /folder-monitor/status` - Folder monitoring status

## 🔑 **Environment Variables**

```bash
# AI/ML Provider API Keys
COHERE_API_KEY=your_cohere_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

# System Configuration
EMBEDDING_PROVIDER=cohere
LLM_PROVIDER=groq
DEBUG=true

# ServiceNow Integration (Optional)
SERVICENOW_INSTANCE=your_instance
SERVICENOW_USERNAME=your_username
SERVICENOW_PASSWORD=your_password
```

## 🚀 **Usage Examples**

### **1. Basic Query**
```python
import requests

response = requests.post("http://localhost:8000/query", 
    json={"query": "What is network troubleshooting?"})
print(response.json())
```

### **2. Document Upload**
```python
files = {"file": open("document.pdf", "rb")}
response = requests.post("http://localhost:8000/upload", files=files)
print(response.json())
```

### **3. System Health Check**
```python
response = requests.get("http://localhost:8000/health")
print(response.json())
```

## 🔧 **Utility Scripts**

### **Health & Diagnostics**
- `health_check_cli.py` - Command-line health checker
- `diagnose.py` - System diagnostics
- `check_vector_stats.py` - Vector database statistics

### **Management**
- `restart_api.py` - Restart API server
- `add_folder_manually.py` - Add monitored folders
- `check_file_ingestion_status.py` - Check ingestion status

## 📊 **System Requirements**

### **Minimum Requirements**
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: 2 cores

### **Recommended Requirements**
- **Python**: 3.10+
- **RAM**: 8GB+
- **Storage**: 10GB+ free space
- **CPU**: 4+ cores
- **GPU**: Optional (for faster embeddings)

## 🔒 **Security Features**

- **API Key Management**: Secure environment variable storage
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error messages
- **CORS Configuration**: Configurable cross-origin policies

## 📈 **Performance Optimization**

- **Async Processing**: Non-blocking API operations
- **Connection Pooling**: Efficient database connections
- **Caching**: Intelligent result caching
- **Batch Processing**: Optimized bulk operations

## 🐛 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd rag-system-core
   python -m pip install -r requirements.txt
   ```

2. **API Key Errors**
   ```bash
   # Check your .env file
   cat .env | grep API_KEY
   ```

3. **Port Conflicts**
   ```bash
   # Check if port 8000 is in use
   netstat -an | grep 8000
   ```

### **Health Check**
```bash
# Run comprehensive health check
python health_check_cli.py

# Check system diagnostics
python diagnose.py
```

## 📝 **Development**

### **Adding New Features**
1. Follow the modular architecture
2. Add appropriate error handling
3. Update API documentation
4. Add tests for new functionality

### **Code Style**
- Use Black for formatting: `black .`
- Use flake8 for linting: `flake8 .`
- Use mypy for type checking: `mypy .`

## 🤝 **Support**

For issues and questions:
1. Check the troubleshooting section
2. Run diagnostic scripts
3. Check system logs in `logs/` directory
4. Review API documentation at `http://localhost:8000/docs`

## 📄 **License**

This RAG System Core is designed for enterprise use with comprehensive features and production-ready architecture.

---

**🎉 Ready to use! Start with `python launch_fixed_ui.py` for the best experience.** 