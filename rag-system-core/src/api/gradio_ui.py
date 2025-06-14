"""
Gradio UI for RAG System Management
Provides a web interface for managing documents, vectors, and system operations
"""
import gradio as gr
import requests
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import os
from pathlib import Path

# Import ServiceNow UI
try:
    from .servicenow_ui import create_servicenow_interface, ServiceNowUI
    SERVICENOW_AVAILABLE = True
except ImportError:
    try:
        # Try absolute import as fallback
        import sys
        sys.path.append(os.path.dirname(__file__))
        from servicenow_ui import create_servicenow_interface, ServiceNowUI
        SERVICENOW_AVAILABLE = True
    except ImportError:
        SERVICENOW_AVAILABLE = False

class RAGSystemUI:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request with error handling"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.request(method, url, **kwargs)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def get_system_stats(self) -> Tuple[str, str]:
        """Get system statistics"""
        try:
            # Basic stats
            stats = self._make_request("GET", "/stats")
            detailed_stats = self._make_request("GET", "/manage/stats/detailed")
            
            if "error" in stats:
                return f"Error: {stats['error']}", ""
            
            # Format basic stats
            basic_info = f"""
## System Statistics

**FAISS Store:**
- Total Vectors: {stats.get('faiss_store', {}).get('vector_count', 0)}
- Active Vectors: {stats.get('faiss_store', {}).get('active_vectors', 0)}
- Dimension: {stats.get('faiss_store', {}).get('dimension', 0)}
- Index Size: {stats.get('faiss_store', {}).get('index_size_mb', 0):.2f} MB

**Metadata Store:**
- Total Files: {stats.get('metadata_store', {}).get('total_files', 0)}
- Collections: {stats.get('metadata_store', {}).get('collections', 0)}
"""
            
            # Format detailed stats
            detailed_info = ""
            if "error" not in detailed_stats:
                detailed_info = f"""
## Detailed Statistics

**Documents:**
- Total Documents: {detailed_stats.get('total_documents', 0)}
- Unknown Documents: {detailed_stats.get('unknown_documents', 0)}
- Avg Chunks per Document: {detailed_stats.get('avg_chunks_per_document', 0):.1f}

**Content:**
- Avg Text Length per Chunk: {detailed_stats.get('avg_text_length_per_chunk', 0):.0f} chars
- Largest Document: {detailed_stats.get('largest_document_chunks', 0)} chunks
- Smallest Document: {detailed_stats.get('smallest_document_chunks', 0)} chunks
"""
            
            return basic_info, detailed_info
            
        except Exception as e:
            return f"Error getting stats: {str(e)}", ""
    
    def list_documents(self, limit: int = 20, title_filter: str = "") -> str:
        """List documents in the system"""
        try:
            params = {"limit": limit}
            if title_filter.strip():
                params["title_filter"] = title_filter.strip()
            
            response = self._make_request("GET", "/manage/documents", params=params)
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            if not response:
                return "No documents found."
            
            # Format as table
            table_data = []
            for doc in response:
                table_data.append([
                    doc.get('doc_id', 'N/A'),
                    doc.get('title', 'N/A'),
                    doc.get('filename', 'N/A'),
                    doc.get('chunk_count', 0),
                    doc.get('total_text_length', 0),
                    doc.get('created_at', 'N/A')[:19] if doc.get('created_at') else 'N/A'
                ])
            
            df = pd.DataFrame(table_data, columns=[
                'Document ID', 'Title', 'Filename', 'Chunks', 'Text Length', 'Created'
            ])
            
            return df.to_string(index=False)
            
        except Exception as e:
            return f"Error listing documents: {str(e)}"
    
    def list_vectors(self, limit: int = 20, doc_id_filter: str = "", text_search: str = "") -> str:
        """List vectors in the system"""
        try:
            params = {"limit": limit}
            if doc_id_filter.strip():
                params["doc_id_filter"] = doc_id_filter.strip()
            if text_search.strip():
                params["text_search"] = text_search.strip()
            
            response = self._make_request("GET", "/manage/vectors", params=params)
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            if not response:
                return "No vectors found."
            
            # Format as table
            table_data = []
            for vector in response:
                table_data.append([
                    vector.get('vector_id', 'N/A'),
                    vector.get('doc_id', 'N/A'),
                    vector.get('text_preview', 'N/A')[:100] + "..." if len(vector.get('text_preview', '')) > 100 else vector.get('text_preview', 'N/A'),
                    vector.get('metadata', {}).get('chunk_index', 'N/A'),
                    vector.get('metadata', {}).get('added_at', 'N/A')[:19] if vector.get('metadata', {}).get('added_at') else 'N/A'
                ])
            
            df = pd.DataFrame(table_data, columns=[
                'Vector ID', 'Document ID', 'Text Preview', 'Chunk Index', 'Created'
            ])
            
            return df.to_string(index=False)
            
        except Exception as e:
            return f"Error listing vectors: {str(e)}"
    
    def get_document_details(self, doc_id: str) -> str:
        """Get detailed information about a document"""
        try:
            if not doc_id.strip():
                return "Please enter a document ID."
            
            response = self._make_request("GET", f"/manage/document/{doc_id.strip()}")
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            # Format document details
            details = f"""
## Document Details: {doc_id}

**Metadata:**
- Title: {response.get('title', 'N/A')}
- Filename: {response.get('filename', 'N/A')}
- Description: {response.get('description', 'N/A')}
- Chunk Count: {response.get('chunk_count', 0)}
- Total Text Length: {response.get('total_text_length', 0)} characters
- Created: {response.get('created_at', 'N/A')[:19] if response.get('created_at') else 'N/A'}

**Chunks:**
"""
            
            for chunk in response.get('chunks', []):
                details += f"""
### Chunk {chunk.get('chunk_index', 0)} (Vector ID: {chunk.get('vector_id', 'N/A')})
{chunk.get('text', 'No text')[:300]}{'...' if len(chunk.get('text', '')) > 300 else ''}

---
"""
            
            return details
            
        except Exception as e:
            return f"Error getting document details: {str(e)}"
    
    def cleanup_unknown_documents(self) -> str:
        """Clean up documents with unknown IDs"""
        try:
            response = self._make_request("POST", "/manage/cleanup/unknown")
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            return f"""
## Cleanup Results

**Action:** {response.get('action', 'N/A')}
**Affected Count:** {response.get('affected_count', 0)}

**Details:**
{chr(10).join(response.get('details', []))}
"""
            
        except Exception as e:
            return f"Error during cleanup: {str(e)}"
    
    def cleanup_duplicates(self) -> str:
        """Clean up duplicate documents"""
        try:
            response = self._make_request("POST", "/manage/cleanup/duplicates")
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            return f"""
## Duplicate Cleanup Results

**Action:** {response.get('action', 'N/A')}
**Affected Count:** {response.get('affected_count', 0)}

**Details:**
{chr(10).join(response.get('details', []))}
"""
            
        except Exception as e:
            return f"Error during duplicate cleanup: {str(e)}"
    
    def reindex_document_ids(self) -> str:
        """Reindex document IDs with improved naming"""
        try:
            response = self._make_request("POST", "/manage/reindex/doc_ids")
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            return f"""
## Reindexing Results

**Action:** {response.get('action', 'N/A')}
**Affected Count:** {response.get('affected_count', 0)}

**Details:**
{chr(10).join(response.get('details', []))}
"""
            
        except Exception as e:
            return f"Error during reindexing: {str(e)}"
    
    def delete_documents(self, doc_ids: str) -> str:
        """Delete specified documents"""
        try:
            if not doc_ids.strip():
                return "Please enter document IDs to delete."
            
            # Parse document IDs (comma-separated)
            doc_id_list = [doc_id.strip() for doc_id in doc_ids.split(',') if doc_id.strip()]
            
            if not doc_id_list:
                return "No valid document IDs provided."
            
            response = self._make_request("DELETE", "/manage/documents", json=doc_id_list)
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            return f"""
## Deletion Results

**Action:** {response.get('action', 'N/A')}
**Affected Count:** {response.get('affected_count', 0)}

**Details:**
{chr(10).join(response.get('details', []))}
"""
            
        except Exception as e:
            return f"Error during deletion: {str(e)}"
    
    def query_system(self, query: str, max_results: int = 3) -> str:
        """Query the RAG system"""
        try:
            if not query.strip():
                return "Please enter a query."
            
            response = self._make_request("POST", "/query", json={
                "query": query.strip(),
                "max_results": max_results
            })
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            # Format query results
            result = f"""
## Query: {query}

**Response:**
{response.get('response', 'No response')}

**Sources ({response.get('context_used', 0)} used):**
"""
            
            for i, source in enumerate(response.get('sources', []), 1):
                result += f"""
### Source {i} (Score: {source.get('score', 0):.3f})
**Document:** {source.get('doc_id', 'Unknown')}
**Text:** {source.get('text', 'No text')}

---
"""
            
            return result
            
        except Exception as e:
            return f"Error during query: {str(e)}"
    
    def ingest_text(self, text: str, title: str = "", description: str = "", author: str = "") -> str:
        """Ingest new text into the system"""
        try:
            if not text.strip():
                return "Please enter text to ingest."
            
            metadata = {}
            if title.strip():
                metadata["title"] = title.strip()
            if description.strip():
                metadata["description"] = description.strip()
            if author.strip():
                metadata["author"] = author.strip()
            
            response = self._make_request("POST", "/ingest", json={
                "text": text.strip(),
                "metadata": metadata
            })
            
            if "error" in response:
                return f"Error: {response['error']}"
            
            return f"""
## Ingestion Results

**Status:** {response.get('status', 'Unknown')}
**File ID:** {response.get('file_id', 'N/A')}
**Chunks Created:** {response.get('chunks_created', 0)}
**Embeddings Generated:** {response.get('embeddings_generated', 0)}

Text has been successfully ingested into the system!
"""
            
        except Exception as e:
            return f"Error during ingestion: {str(e)}"

    def upload_file(self, file_path: str, title: str = "", description: str = "", author: str = "") -> str:
        """Upload and ingest a file"""
        try:
            if not file_path or not os.path.exists(file_path):
                return "Please select a valid file to upload."
            
            # Prepare metadata
            metadata = {}
            if title.strip():
                metadata["title"] = title.strip()
            if description.strip():
                metadata["description"] = description.strip()
            if author.strip():
                metadata["author"] = author.strip()
            
            # Upload file
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
                data = {'metadata': json.dumps(metadata)} if metadata else {}
                
                response = requests.post(
                    f"{self.api_base_url}/upload",
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout for large files
                )
            
            if response.status_code == 200:
                result = response.json()
                return f"""
## File Upload Results

**Status:** {result.get('status', 'Unknown')}
**File:** {os.path.basename(file_path)}
**File ID:** {result.get('file_id', 'N/A')}
**Chunks Created:** {result.get('chunks_created', 0)}
**Embeddings Generated:** {result.get('embeddings_generated', 0)}

File has been successfully uploaded and ingested!
"""
            else:
                return f"Error: Upload failed with status {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"Error during file upload: {str(e)}"

    def process_directory(self, directory_path: str, file_extensions: str = "pdf,txt,docx,md", 
                         recursive: bool = True) -> str:
        """Process all files in a directory"""
        try:
            if not directory_path or not os.path.exists(directory_path):
                return "Please enter a valid directory path."
            
            # Parse file extensions
            extensions = [ext.strip().lower() for ext in file_extensions.split(',') if ext.strip()]
            if not extensions:
                extensions = ['pdf', 'txt', 'docx', 'md']
            
            # Find files
            directory = Path(directory_path)
            files_found = []
            
            if recursive:
                for ext in extensions:
                    files_found.extend(directory.rglob(f"*.{ext}"))
            else:
                for ext in extensions:
                    files_found.extend(directory.glob(f"*.{ext}"))
            
            if not files_found:
                return f"No files found with extensions: {', '.join(extensions)}"
            
            # Process files
            results = []
            successful = 0
            failed = 0
            
            for file_path in files_found:
                try:
                    # Extract metadata from filename/path
                    metadata = {
                        "title": file_path.stem,
                        "filename": file_path.name,
                        "directory": str(file_path.parent)
                    }
                    
                    # Upload file
                    with open(file_path, 'rb') as f:
                        files = {'file': (file_path.name, f, 'application/octet-stream')}
                        data = {'metadata': json.dumps(metadata)}
                        
                        response = requests.post(
                            f"{self.api_base_url}/upload",
                            files=files,
                            data=data,
                            timeout=300
                        )
                    
                    if response.status_code == 200:
                        result = response.json()
                        results.append(f"✅ {file_path.name}: {result.get('chunks_created', 0)} chunks")
                        successful += 1
                    else:
                        results.append(f"❌ {file_path.name}: Upload failed ({response.status_code})")
                        failed += 1
                        
                except Exception as e:
                    results.append(f"❌ {file_path.name}: Error - {str(e)}")
                    failed += 1
            
            return f"""
## Directory Processing Results

**Directory:** {directory_path}
**Files Found:** {len(files_found)}
**Successful:** {successful}
**Failed:** {failed}

**Details:**
{chr(10).join(results)}
"""
            
        except Exception as e:
            return f"Error processing directory: {str(e)}"

    def configure_scheduler(self, watch_directory: str = "", schedule_time: str = "02:00", 
                          file_extensions: str = "pdf,txt,docx,md", enabled: bool = False) -> str:
        """Configure the document processing scheduler"""
        try:
            config = {
                "enabled": enabled,
                "watch_directory": watch_directory.strip(),
                "schedule_time": schedule_time.strip(),
                "file_extensions": file_extensions.strip(),
                "recursive": True
            }
            
            # Note: This would need a backend endpoint to actually configure the scheduler
            # For now, we'll return a configuration summary
            
            if enabled and not watch_directory:
                return "Error: Watch directory is required when scheduler is enabled."
            
            status = "Enabled" if enabled else "Disabled"
            
            return f"""
## Scheduler Configuration

**Status:** {status}
**Watch Directory:** {watch_directory or 'Not set'}
**Schedule Time:** {schedule_time} (daily)
**File Extensions:** {file_extensions}
**Recursive Processing:** Yes

{f"⚠️ Note: Scheduler configuration saved but requires backend implementation to be active." if enabled else ""}

To implement this feature, the backend needs:
1. A scheduler configuration endpoint
2. File system monitoring capability
3. Automated processing triggers
"""
            
        except Exception as e:
            return f"Error configuring scheduler: {str(e)}"

def create_gradio_interface(api_base_url: str = "http://localhost:8000") -> gr.Blocks:
    """Create the Gradio interface"""
    ui = RAGSystemUI(api_base_url)
    
    with gr.Blocks(title="RAG System Management", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔧 RAG System Management Dashboard")
        gr.Markdown("Manage documents, vectors, and system operations for your RAG system.")
        
        with gr.Tabs():
            # System Overview Tab
            with gr.TabItem("📊 System Overview"):
                gr.Markdown("## System Statistics and Health")
                
                with gr.Row():
                    refresh_stats_btn = gr.Button("🔄 Refresh Stats", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        basic_stats = gr.Markdown("Loading basic stats...")
                    with gr.Column():
                        detailed_stats = gr.Markdown("Loading detailed stats...")
                
                refresh_stats_btn.click(
                    ui.get_system_stats,
                    outputs=[basic_stats, detailed_stats]
                )
            
            # Document Management Tab
            with gr.TabItem("📄 Document Management"):
                gr.Markdown("## Browse and Manage Documents")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        doc_limit = gr.Slider(5, 100, value=20, step=5, label="Number of documents to show")
                        doc_title_filter = gr.Textbox(label="Filter by title (optional)", placeholder="Enter title to search...")
                    with gr.Column(scale=1):
                        list_docs_btn = gr.Button("📋 List Documents", variant="primary")
                
                doc_list_output = gr.Textbox(label="Documents", lines=15, max_lines=20)
                
                gr.Markdown("### Document Details")
                with gr.Row():
                    doc_id_input = gr.Textbox(label="Document ID", placeholder="Enter document ID to view details...")
                    get_doc_details_btn = gr.Button("🔍 Get Details")
                
                doc_details_output = gr.Markdown("Enter a document ID to view details.")
                
                list_docs_btn.click(
                    ui.list_documents,
                    inputs=[doc_limit, doc_title_filter],
                    outputs=doc_list_output
                )
                
                get_doc_details_btn.click(
                    ui.get_document_details,
                    inputs=doc_id_input,
                    outputs=doc_details_output
                )
            
            # Vector Management Tab
            with gr.TabItem("🔢 Vector Management"):
                gr.Markdown("## Browse and Manage Vectors")
                
                with gr.Row():
                    with gr.Column():
                        vector_limit = gr.Slider(5, 100, value=20, step=5, label="Number of vectors to show")
                        vector_doc_filter = gr.Textbox(label="Filter by document ID (optional)", placeholder="Enter document ID...")
                        vector_text_search = gr.Textbox(label="Search in text (optional)", placeholder="Enter text to search...")
                    with gr.Column():
                        list_vectors_btn = gr.Button("📋 List Vectors", variant="primary")
                
                vector_list_output = gr.Textbox(label="Vectors", lines=15, max_lines=20)
                
                list_vectors_btn.click(
                    ui.list_vectors,
                    inputs=[vector_limit, vector_doc_filter, vector_text_search],
                    outputs=vector_list_output
                )
            
            # Cleanup Operations Tab
            with gr.TabItem("🧹 Cleanup Operations"):
                gr.Markdown("## System Cleanup and Maintenance")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Automated Cleanup")
                        cleanup_unknown_btn = gr.Button("🗑️ Clean Unknown Documents", variant="secondary")
                        cleanup_duplicates_btn = gr.Button("🔄 Remove Duplicates", variant="secondary")
                        reindex_btn = gr.Button("🏷️ Reindex Document IDs", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("### Manual Deletion")
                        delete_doc_ids = gr.Textbox(
                            label="Document IDs to delete (comma-separated)", 
                            placeholder="doc_example_1, doc_example_2, ..."
                        )
                        delete_docs_btn = gr.Button("❌ Delete Documents", variant="stop")
                
                cleanup_output = gr.Markdown("Cleanup results will appear here.")
                
                cleanup_unknown_btn.click(ui.cleanup_unknown_documents, outputs=cleanup_output)
                cleanup_duplicates_btn.click(ui.cleanup_duplicates, outputs=cleanup_output)
                reindex_btn.click(ui.reindex_document_ids, outputs=cleanup_output)
                delete_docs_btn.click(ui.delete_documents, inputs=delete_doc_ids, outputs=cleanup_output)
            
            # Query System Tab
            with gr.TabItem("🔍 Query System"):
                gr.Markdown("## Test System Queries")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(label="Query", placeholder="Enter your question...")
                    with gr.Column(scale=1):
                        max_results = gr.Slider(1, 10, value=3, step=1, label="Max Results")
                        query_btn = gr.Button("🔍 Query", variant="primary")
                
                query_output = gr.Markdown("Query results will appear here.")
                
                query_btn.click(
                    ui.query_system,
                    inputs=[query_input, max_results],
                    outputs=query_output
                )
            
            # Data Ingestion Tab
            with gr.TabItem("📥 Data Ingestion"):
                gr.Markdown("## Add New Content")
                
                with gr.Tabs():
                    # Text Ingestion Sub-tab
                    with gr.TabItem("📝 Text Input"):
                        with gr.Column():
                            ingest_text_input = gr.Textbox(
                                label="Text to ingest", 
                                lines=10, 
                                placeholder="Enter the text content you want to add to the system..."
                            )
                            
                            with gr.Row():
                                ingest_title = gr.Textbox(label="Title (optional)", placeholder="Document title...")
                                ingest_author = gr.Textbox(label="Author (optional)", placeholder="Author name...")
                            
                            ingest_description = gr.Textbox(
                                label="Description (optional)", 
                                placeholder="Brief description of the content..."
                            )
                            
                            ingest_btn = gr.Button("📥 Ingest Text", variant="primary")
                        
                        ingest_output = gr.Markdown("Ingestion results will appear here.")
                        
                        ingest_btn.click(
                            ui.ingest_text,
                            inputs=[ingest_text_input, ingest_title, ingest_description, ingest_author],
                            outputs=ingest_output
                        )
                    
                    # File Upload Sub-tab
                    with gr.TabItem("📄 File Upload"):
                        with gr.Column():
                            upload_file_input = gr.File(
                                label="Select file to upload",
                                file_types=[".pdf", ".txt", ".docx", ".md", ".doc", ".rtf"]
                            )
                            
                            with gr.Row():
                                upload_title = gr.Textbox(label="Title (optional)", placeholder="Document title...")
                                upload_author = gr.Textbox(label="Author (optional)", placeholder="Author name...")
                            
                            upload_description = gr.Textbox(
                                label="Description (optional)", 
                                placeholder="Brief description of the content..."
                            )
                            
                            upload_btn = gr.Button("📤 Upload File", variant="primary")
                        
                        upload_output = gr.Markdown("Upload results will appear here.")
                        
                        upload_btn.click(
                            ui.upload_file,
                            inputs=[upload_file_input, upload_title, upload_description, upload_author],
                            outputs=upload_output
                        )
                    
                    # Directory Processing Sub-tab
                    with gr.TabItem("📁 Directory Processing"):
                        with gr.Column():
                            gr.Markdown("### Bulk Process Files from Directory")
                            
                            directory_path = gr.Textbox(
                                label="Directory Path",
                                placeholder="Enter full path to directory (e.g., C:\\Documents\\MyFiles)",
                                info="All supported files in this directory will be processed"
                            )
                            
                            with gr.Row():
                                file_extensions = gr.Textbox(
                                    label="File Extensions",
                                    value="pdf,txt,docx,md",
                                    placeholder="pdf,txt,docx,md",
                                    info="Comma-separated list of file extensions to process"
                                )
                                recursive_processing = gr.Checkbox(
                                    label="Include Subdirectories",
                                    value=True,
                                    info="Process files in subdirectories recursively"
                                )
                            
                            process_dir_btn = gr.Button("📁 Process Directory", variant="primary")
                        
                        directory_output = gr.Markdown("Directory processing results will appear here.")
                        
                        process_dir_btn.click(
                            ui.process_directory,
                            inputs=[directory_path, file_extensions, recursive_processing],
                            outputs=directory_output
                        )
            
            # Scheduler Configuration Tab
            with gr.TabItem("⏰ Scheduler"):
                gr.Markdown("## Automated Document Processing")
                gr.Markdown("Configure the system to automatically process documents from a specific directory on a schedule.")
                
                with gr.Column():
                    with gr.Row():
                        scheduler_enabled = gr.Checkbox(
                            label="Enable Scheduler",
                            value=False,
                            info="Enable automatic document processing"
                        )
                        schedule_time = gr.Textbox(
                            label="Schedule Time (24h format)",
                            value="02:00",
                            placeholder="02:00",
                            info="Daily processing time (HH:MM)"
                        )
                    
                    watch_directory = gr.Textbox(
                        label="Watch Directory",
                        placeholder="Enter directory path to monitor for new documents",
                        info="Directory that will be monitored for new files"
                    )
                    
                    scheduler_extensions = gr.Textbox(
                        label="File Extensions to Process",
                        value="pdf,txt,docx,md",
                        placeholder="pdf,txt,docx,md",
                        info="Comma-separated list of file extensions"
                    )
                    
                    configure_scheduler_btn = gr.Button("⚙️ Configure Scheduler", variant="secondary")
                
                scheduler_output = gr.Markdown("""
### Current Status: Not Configured

The scheduler allows you to:
- Monitor a directory for new documents
- Automatically process files at scheduled times
- Support multiple file formats
- Process files recursively in subdirectories

**Note:** This feature requires additional backend implementation for full functionality.
""")
                
                configure_scheduler_btn.click(
                    ui.configure_scheduler,
                    inputs=[watch_directory, schedule_time, scheduler_extensions, scheduler_enabled],
                    outputs=scheduler_output
                )
            
            # ServiceNow Integration Tab
            if SERVICENOW_AVAILABLE:
                with gr.TabItem("🎫 ServiceNow"):
                    gr.Markdown("## ServiceNow Ticket Management")
                    gr.Markdown("Browse, filter, and selectively ingest ServiceNow tickets into your RAG system")
                    
                    # Import ServiceNow UI components directly
                    servicenow_ui = ServiceNowUI(api_base_url)
                    
                    with gr.Tabs():
                        # Browse Tickets Tab
                        with gr.TabItem("📋 Browse Tickets"):
                            with gr.Row():
                                with gr.Column(scale=2):
                                    gr.Markdown("### 🔍 Filters")
                                    priority_filter = gr.Dropdown(
                                        choices=["All", "1", "2", "3", "4", "5"],
                                        value="All",
                                        label="Priority Filter",
                                        info="1=Critical, 2=High, 3=Moderate, 4=Low, 5=Planning"
                                    )
                                    state_filter = gr.Dropdown(
                                        choices=["All", "1", "2", "3", "6", "7"],
                                        value="All", 
                                        label="State Filter",
                                        info="1=New, 2=In Progress, 3=On Hold, 6=Resolved, 7=Closed"
                                    )
                                    category_filter = gr.Dropdown(
                                        choices=["All", "network", "hardware", "software", "inquiry"],
                                        value="All",
                                        label="Category Filter"
                                    )
                                    
                                with gr.Column(scale=1):
                                    gr.Markdown("### 📄 Pagination")
                                    current_page = gr.Number(value=1, label="Page", precision=0, minimum=1)
                                    page_size = gr.Number(value=10, label="Items per page", precision=0, minimum=1, maximum=50)
                                    
                            fetch_btn = gr.Button("🔄 Fetch Tickets", variant="primary", size="lg")
                            
                            with gr.Row():
                                with gr.Column(scale=2):
                                    tickets_table = gr.Textbox(
                                        label="📋 ServiceNow Tickets",
                                        lines=15,
                                        max_lines=20,
                                        interactive=False,
                                        show_copy_button=True
                                    )
                                    
                                with gr.Column(scale=1):
                                    pagination_info = gr.Markdown("📄 Pagination info will appear here")
                        
                        # Select & Ingest Tab
                        with gr.TabItem("✅ Select & Ingest"):
                            gr.Markdown("### 🎯 Ticket Selection")
                            gr.Markdown("Select tickets from the list below or enter ticket IDs manually")
                            
                            ticket_checkboxes = gr.HTML(label="Select Tickets")
                            
                            with gr.Row():
                                selected_ids = gr.Textbox(
                                    label="Selected Ticket IDs (comma-separated)",
                                    placeholder="Enter ticket IDs manually or use checkboxes above",
                                    lines=2,
                                    info="Example: ticket_1,ticket_2,ticket_3"
                                )
                                
                            with gr.Row():
                                update_selection_btn = gr.Button("🔄 Update Selection", variant="secondary")
                                ingest_btn = gr.Button("🚀 Ingest Selected Tickets", variant="primary")
                            
                            selection_status = gr.Textbox(
                                label="Selection Status",
                                lines=2,
                                interactive=False
                            )
                            
                            ingestion_results = gr.Markdown("### 📊 Ingestion results will appear here")
                        
                        # Statistics Tab
                        with gr.TabItem("📊 Statistics"):
                            stats_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
                            stats_display = gr.Markdown("### 📈 Statistics will appear here")
                    
                    # Event handlers
                    fetch_btn.click(
                        fn=servicenow_ui.fetch_servicenow_tickets,
                        inputs=[current_page, page_size, priority_filter, state_filter, category_filter],
                        outputs=[tickets_table, ticket_checkboxes, pagination_info]
                    )
                    
                    update_selection_btn.click(
                        fn=servicenow_ui.update_ticket_selection,
                        inputs=[selected_ids],
                        outputs=[selection_status]
                    )
                    
                    ingest_btn.click(
                        fn=servicenow_ui.ingest_selected_tickets,
                        inputs=[],
                        outputs=[ingestion_results]
                    )
                    
                    stats_btn.click(
                        fn=servicenow_ui.get_servicenow_stats,
                        inputs=[],
                        outputs=[stats_display]
                    )
            else:
                with gr.TabItem("🎫 ServiceNow (Unavailable)"):
                    gr.Markdown("## ServiceNow Integration Not Available")
                    gr.Markdown("""
                    The ServiceNow integration module is not available. This could be due to:
                    
                    - Missing ServiceNow UI module
                    - Import errors in the ServiceNow components
                    - Missing dependencies
                    
                    To enable ServiceNow integration:
                    1. Ensure the ServiceNow UI module is properly installed
                    2. Check that all dependencies are available
                    3. Restart the application
                    """)
        
        # Auto-refresh stats on load
        interface.load(ui.get_system_stats, outputs=[basic_stats, detailed_stats])
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    ) 