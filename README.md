# RAG System Core

A robust Retrieval-Augmented Generation (RAG) system for document processing, search, and retrieval.

## Features

- Document processing for multiple formats (PDF, DOCX, XLSX, images)
- OCR capabilities with Azure integration
- Advanced text chunking and vector embedding
- Semantic and hybrid search functionality
- Comprehensive monitoring and health checks
- RESTful API endpoints
- User interface components

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd rag-system-core
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
# Edit .env with your configuration
```

## Usage

1. Start the API server:
```bash
uvicorn main_api_extract:app --reload
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## Project Structure

```
src/
├── api/            # API endpoints and request handling
├── core/           # Core system functionality
├── document_processing/  # Document handling and processing
├── ingestion/      # Document ingestion pipeline
├── integrations/   # External system integrations
├── monitoring/     # System monitoring and metrics
├── retrieval/      # Search and retrieval functionality
├── storage/        # Data storage and management
├── ui/            # User interface components
└── utils/         # Utility functions and helpers
```

## Configuration

The system can be configured through:
- Environment variables (.env file)
- Configuration files in the config/ directory
- API parameters

## Development

1. Set up development environment:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your License]

## Contact

[Your Contact Information] 