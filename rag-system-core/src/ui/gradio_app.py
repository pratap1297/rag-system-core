"""
Gradio UI Application
Simple web interface for the RAG system
"""
import logging
from typing import Optional

def create_gradio_app(container) -> Optional[object]:
    """Create Gradio application"""
    try:
        # Optional Gradio interface - only if gradio is installed
        import gradio as gr
        
        query_engine = container.get('query_engine')
        ingestion_engine = container.get('ingestion_engine')
        
        def process_query(query: str, top_k: int = 5):
            """Process a query through the RAG system"""
            try:
                result = query_engine.process_query(query=query, top_k=top_k)
                return result['response'], str(result['sources'])
            except Exception as e:
                return f"Error: {str(e)}", ""
        
        def upload_file(file):
            """Upload and ingest a file"""
            try:
                if file is None:
                    return "No file uploaded"
                
                result = ingestion_engine.ingest_file(file.name)
                return f"File ingested successfully: {result['chunks_created']} chunks created"
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Create Gradio interface
        with gr.Blocks(title="RAG System") as app:
            gr.Markdown("# RAG System Interface")
            
            with gr.Tab("Query"):
                query_input = gr.Textbox(label="Enter your query", lines=2)
                top_k_input = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of results")
                query_button = gr.Button("Submit Query")
                
                response_output = gr.Textbox(label="Response", lines=5)
                sources_output = gr.Textbox(label="Sources", lines=3)
                
                query_button.click(
                    process_query,
                    inputs=[query_input, top_k_input],
                    outputs=[response_output, sources_output]
                )
            
            with gr.Tab("Upload"):
                file_input = gr.File(label="Upload Document")
                upload_button = gr.Button("Upload and Ingest")
                upload_output = gr.Textbox(label="Upload Result")
                
                upload_button.click(
                    upload_file,
                    inputs=[file_input],
                    outputs=[upload_output]
                )
        
        logging.info("Gradio app created")
        return app
        
    except ImportError:
        logging.info("Gradio not installed, skipping UI creation")
        return None
    except Exception as e:
        logging.error(f"Failed to create Gradio app: {e}")
        return None 