"""
Gradio UI for RAG System Testing
Provides a user-friendly interface to test all RAG system functionality
"""
import gradio as gr
import requests
import json
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystemUI:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url.rstrip('/')
        self.session_history = []
        
    def check_api_health(self) -> Tuple[str, str]:
        """Check if the API is running and healthy"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                status = health_data.get('status', 'unknown')
                timestamp = health_data.get('timestamp', 'unknown')
                components = health_data.get('components', {})
                
                status_text = f"✅ API Status: {status.upper()}\n"
                status_text += f"🕐 Last Check: {timestamp}\n"
                status_text += f"🔧 Components: {len(components)} active\n"
                
                if health_data.get('issues'):
                    status_text += f"⚠️ Issues: {len(health_data['issues'])}"
                
                return status_text, "success"
            else:
                return f"❌ API Error: HTTP {response.status_code}", "error"
        except requests.exceptions.RequestException as e:
            return f"❌ Connection Error: {str(e)}", "error"
    
    def query_rag_system(self, query: str, max_results: int = 3) -> Tuple[str, str, str]:
        """Query the RAG system and return response with sources"""
        if not query.strip():
            return "Please enter a query.", "", ""
        
        try:
            # Add to session history
            self.session_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "query",
                "content": query
            })
            
            payload = {
                "query": query,
                "max_results": max_results
            }
            
            response = requests.post(
                f"{self.api_base_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format response
                answer = data.get('response', 'No response generated')
                
                # Format sources
                sources = data.get('sources', [])
                sources_text = ""
                if sources:
                    sources_text = "📚 **Sources:**\n\n"
                    for i, source in enumerate(sources, 1):
                        score = source.get('score', 0)
                        doc_id = source.get('doc_id', 'Unknown')
                        text_preview = source.get('text', '')[:150] + "..."
                        sources_text += f"**Source {i}** (Score: {score:.3f})\n"
                        sources_text += f"Document: `{doc_id}`\n"
                        sources_text += f"Preview: {text_preview}\n\n"
                else:
                    sources_text = "No sources found."
                
                # Format metadata
                context_used = data.get('context_used', 0)
                metadata = f"**Query Metadata:**\n"
                metadata += f"- Context chunks used: {context_used}\n"
                metadata += f"- Max results requested: {max_results}\n"
                metadata += f"- Query timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                return answer, sources_text, metadata
                
            else:
                error_msg = f"API Error: HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"\nDetails: {error_detail}"
                except:
                    error_msg += f"\nResponse: {response.text[:200]}"
                
                return error_msg, "", ""
                
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {str(e)}", "", ""
        except Exception as e:
            return f"Unexpected Error: {str(e)}", "", ""
    
    def ingest_text(self, text: str, title: str = "", description: str = "") -> str:
        """Ingest text into the RAG system"""
        if not text.strip():
            return "Please enter text to ingest."
        
        try:
            # Add to session history
            self.session_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "ingestion",
                "content": f"Text: {text[:100]}..." if len(text) > 100 else text
            })
            
            metadata = {
                "title": title or f"Text_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "description": description,
                "source": "gradio_ui",
                "ingestion_timestamp": datetime.now().isoformat()
            }
            
            payload = {
                "text": text,
                "metadata": metadata
            }
            
            response = requests.post(
                f"{self.api_base_url}/ingest",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                result = f"✅ **Ingestion Successful!**\n\n"
                result += f"📄 File ID: `{data.get('file_id', 'N/A')}`\n"
                result += f"📝 Chunks Created: {data.get('chunks_created', 0)}\n"
                result += f"🔢 Embeddings Generated: {data.get('embeddings_generated', 0)}\n"
                result += f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                return result
            else:
                error_msg = f"❌ **Ingestion Failed**\n\n"
                error_msg += f"HTTP Status: {response.status_code}\n"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"Details: {error_detail}"
                except:
                    error_msg += f"Response: {response.text[:200]}"
                return error_msg
                
        except requests.exceptions.RequestException as e:
            return f"❌ **Connection Error**\n\n{str(e)}"
        except Exception as e:
            return f"❌ **Unexpected Error**\n\n{str(e)}"
    
    def upload_file(self, file) -> str:
        """Upload a file to the RAG system"""
        if file is None:
            return "Please select a file to upload."
        
        try:
            # Add to session history
            self.session_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "file_upload",
                "content": f"File: {file.name}"
            })
            
            with open(file.name, 'rb') as f:
                files = {'file': (os.path.basename(file.name), f, 'application/octet-stream')}
                metadata = {
                    "source": "gradio_ui",
                    "upload_timestamp": datetime.now().isoformat()
                }
                data = {'metadata': json.dumps(metadata)}
                
                response = requests.post(
                    f"{self.api_base_url}/upload",
                    files=files,
                    data=data,
                    timeout=120
                )
            
            if response.status_code == 200:
                data = response.json()
                result = f"✅ **File Upload Successful!**\n\n"
                result += f"📄 File: `{os.path.basename(file.name)}`\n"
                result += f"📝 Chunks Created: {data.get('chunks_created', 0)}\n"
                result += f"🔢 Embeddings Generated: {data.get('embeddings_generated', 0)}\n"
                result += f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                return result
            else:
                error_msg = f"❌ **Upload Failed**\n\n"
                error_msg += f"HTTP Status: {response.status_code}\n"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"Details: {error_detail}"
                except:
                    error_msg += f"Response: {response.text[:200]}"
                return error_msg
                
        except Exception as e:
            return f"❌ **Upload Error**\n\n{str(e)}"
    
    def get_system_stats(self) -> str:
        """Get system statistics"""
        try:
            response = requests.get(f"{self.api_base_url}/stats", timeout=10)
            if response.status_code == 200:
                stats = response.json()
                
                result = "📊 **System Statistics**\n\n"
                
                # Document stats
                if 'documents' in stats:
                    doc_stats = stats['documents']
                    result += f"📄 **Documents:**\n"
                    result += f"  - Total: {doc_stats.get('total', 0)}\n"
                    result += f"  - Processed: {doc_stats.get('processed', 0)}\n\n"
                
                # Vector stats
                if 'vectors' in stats:
                    vec_stats = stats['vectors']
                    result += f"🔢 **Vectors:**\n"
                    result += f"  - Total: {vec_stats.get('total', 0)}\n"
                    result += f"  - Dimension: {vec_stats.get('dimension', 0)}\n\n"
                
                # Storage stats
                if 'storage' in stats:
                    storage_stats = stats['storage']
                    result += f"💾 **Storage:**\n"
                    result += f"  - Index size: {storage_stats.get('index_size', 'N/A')}\n"
                    result += f"  - Metadata entries: {storage_stats.get('metadata_entries', 0)}\n\n"
                
                result += f"🕐 **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                return result
            else:
                return f"❌ Failed to get stats: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ Error getting stats: {str(e)}"
    
    def get_session_history(self) -> str:
        """Get session history"""
        if not self.session_history:
            return "No activity in this session yet."
        
        history_text = "📋 **Session History**\n\n"
        for i, entry in enumerate(reversed(self.session_history[-10:]), 1):  # Last 10 entries
            timestamp = entry['timestamp']
            entry_type = entry['type']
            content = entry['content']
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp
            
            history_text += f"**{i}.** `{time_str}` - {entry_type.title()}\n"
            history_text += f"   {content[:100]}{'...' if len(content) > 100 else ''}\n\n"
        
        return history_text
    
    def clear_session_history(self) -> str:
        """Clear session history"""
        self.session_history.clear()
        return "Session history cleared."

def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize the UI class
    rag_ui = RAGSystemUI()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .tab-nav {
        background-color: #f8f9fa;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=css, title="RAG System Test Interface") as interface:
        
        gr.Markdown("""
        # 🤖 RAG System Test Interface
        
        Welcome to the RAG (Retrieval-Augmented Generation) System testing interface. 
        Use the tabs below to test different aspects of the system.
        """)
        
        with gr.Tabs():
            
            # Query Tab
            with gr.TabItem("🔍 Query System"):
                gr.Markdown("### Ask questions and get AI-powered responses with source citations")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Enter your question here...",
                            lines=3
                        )
                        max_results = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Maximum Results"
                        )
                        query_btn = gr.Button("🔍 Ask Question", variant="primary")
                    
                    with gr.Column(scale=1):
                        health_status = gr.Textbox(
                            label="API Status",
                            interactive=False,
                            lines=4
                        )
                        refresh_btn = gr.Button("🔄 Refresh Status")
                
                with gr.Row():
                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="AI Response",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column():
                        sources_output = gr.Markdown(
                            label="Sources & Citations"
                        )
                
                metadata_output = gr.Markdown(label="Query Metadata")
                
                # Event handlers for Query tab
                query_btn.click(
                    fn=rag_ui.query_rag_system,
                    inputs=[query_input, max_results],
                    outputs=[answer_output, sources_output, metadata_output]
                )
                
                refresh_btn.click(
                    fn=rag_ui.check_api_health,
                    outputs=[health_status, gr.State()]
                )
            
            # Ingestion Tab
            with gr.TabItem("📝 Add Content"):
                gr.Markdown("### Add text content or upload files to the knowledge base")
                
                with gr.Tabs():
                    with gr.TabItem("Text Input"):
                        text_input = gr.Textbox(
                            label="Text Content",
                            placeholder="Paste or type your text content here...",
                            lines=10
                        )
                        
                        with gr.Row():
                            title_input = gr.Textbox(
                                label="Title (optional)",
                                placeholder="Give your content a title..."
                            )
                            description_input = gr.Textbox(
                                label="Description (optional)",
                                placeholder="Brief description of the content..."
                            )
                        
                        ingest_btn = gr.Button("📝 Add Text to Knowledge Base", variant="primary")
                        ingest_output = gr.Markdown(label="Ingestion Result")
                        
                        ingest_btn.click(
                            fn=rag_ui.ingest_text,
                            inputs=[text_input, title_input, description_input],
                            outputs=[ingest_output]
                        )
                    
                    with gr.TabItem("File Upload"):
                        file_input = gr.File(
                            label="Upload Document",
                            file_types=[".txt", ".pdf", ".docx", ".md"]
                        )
                        upload_btn = gr.Button("📤 Upload File", variant="primary")
                        upload_output = gr.Markdown(label="Upload Result")
                        
                        upload_btn.click(
                            fn=rag_ui.upload_file,
                            inputs=[file_input],
                            outputs=[upload_output]
                        )
            
            # System Status Tab
            with gr.TabItem("📊 System Status"):
                gr.Markdown("### Monitor system health and statistics")
                
                with gr.Row():
                    with gr.Column():
                        stats_output = gr.Markdown(label="System Statistics")
                        stats_btn = gr.Button("📊 Get Statistics", variant="secondary")
                        
                        stats_btn.click(
                            fn=rag_ui.get_system_stats,
                            outputs=[stats_output]
                        )
                    
                    with gr.Column():
                        history_output = gr.Markdown(label="Session Activity")
                        with gr.Row():
                            history_btn = gr.Button("📋 View History", variant="secondary")
                            clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
                        
                        history_btn.click(
                            fn=rag_ui.get_session_history,
                            outputs=[history_output]
                        )
                        
                        clear_btn.click(
                            fn=rag_ui.clear_session_history,
                            outputs=[history_output]
                        )
            
            # Help Tab
            with gr.TabItem("❓ Help"):
                gr.Markdown("""
                ### How to Use This Interface
                
                #### 🔍 **Query System**
                - Enter your questions in natural language
                - The system will search the knowledge base and provide AI-generated answers
                - Sources and citations are provided for transparency
                - Adjust "Maximum Results" to control how many source documents to consider
                
                #### 📝 **Add Content**
                - **Text Input**: Paste text directly into the system
                - **File Upload**: Upload PDF, DOCX, TXT, or Markdown files
                - Added content becomes searchable immediately
                
                #### 📊 **System Status**
                - Monitor system health and performance
                - View statistics about documents and vectors
                - Track your session activity
                
                #### 🔧 **Troubleshooting**
                - If you see connection errors, make sure the RAG API server is running
                - Default API URL: `http://localhost:8000`
                - Check the API Status indicator in the Query tab
                
                #### 🚀 **Getting Started**
                1. First, check that the API status shows "✅ HEALTHY"
                2. Add some content using the "Add Content" tab
                3. Ask questions in the "Query System" tab
                4. Monitor system performance in "System Status"
                
                #### 📋 **Supported File Types**
                - **PDF**: `.pdf` files
                - **Word**: `.docx` files  
                - **Text**: `.txt` files
                - **Markdown**: `.md` files
                
                #### ⚡ **Tips for Better Results**
                - Be specific in your questions
                - Add relevant context when ingesting content
                - Use descriptive titles for uploaded content
                - Monitor the sources to understand where answers come from
                """)
        
        # Auto-refresh API status on load
        interface.load(
            fn=rag_ui.check_api_health,
            outputs=[health_status, gr.State()]
        )
    
    return interface

def main():
    """Main function to launch the Gradio interface"""
    
    # Check if API is running
    print("🚀 Starting RAG System Test Interface...")
    print("📡 Checking API connection...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy!")
        else:
            print(f"⚠️ API responded with status {response.status_code}")
    except requests.exceptions.RequestException:
        print("❌ Warning: Could not connect to API at http://localhost:8000")
        print("   Make sure the RAG system API is running before using this interface.")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    print("\n🌐 Launching Gradio interface...")
    print("📱 The interface will open in your default web browser")
    print("🔗 You can also access it at: http://localhost:7860")
    print("\n💡 Tip: Keep this terminal open while using the interface")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 