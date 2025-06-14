"""
Enhanced Gradio UI for RAG System Management
Provides a web interface for managing documents, vectors, and system operations
Includes file upload and scheduler functionality
"""
import gradio as gr
import requests
import json
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging
import os
from pathlib import Path

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

    def upload_file(self, file_obj, title: str = "", description: str = "", author: str = "") -> str:
        """Upload and ingest a file"""
        try:
            if file_obj is None:
                return "Please select a file to upload."
            
            # Get file path from Gradio file object
            file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
            
            if not os.path.exists(file_path):
                return "Selected file does not exist."
            
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

def create_enhanced_interface(api_base_url: str = "http://localhost:8000") -> gr.Blocks:
    """Create the enhanced Gradio interface with file upload and scheduler"""
    ui = RAGSystemUI(api_base_url)
    
    with gr.Blocks(title="Enhanced RAG System Management", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔧 Enhanced RAG System Management Dashboard")
        gr.Markdown("Complete management interface with file upload and automated processing capabilities.")
        
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
            
            # File Upload Tab
            with gr.TabItem("📤 File Upload"):
                gr.Markdown("## Upload Documents")
                gr.Markdown("Upload individual files to be processed and added to the RAG system.")
                
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
            
            # Directory Processing Tab
            with gr.TabItem("📁 Directory Processing"):
                gr.Markdown("## Bulk Process Files from Directory")
                gr.Markdown("Process all supported files from a directory and its subdirectories.")
                
                with gr.Column():
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
        
        # Auto-refresh stats on load
        interface.load(ui.get_system_stats, outputs=[basic_stats, detailed_stats])
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_enhanced_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port to avoid conflicts
        share=False,
        debug=True
    ) 