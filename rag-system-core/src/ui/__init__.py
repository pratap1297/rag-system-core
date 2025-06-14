"""
UI Package
User interface components for RAG System including web interface and Gradio app
"""

from .web_interface import WebInterface, create_web_interface, create_simple_status_server

try:
    from .gradio_app import create_gradio_interface
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    def create_gradio_interface(*args, **kwargs):
        raise ImportError("Gradio not available. Install with: pip install gradio")

__all__ = [
    'WebInterface', 'create_web_interface', 'create_simple_status_server',
    'create_gradio_interface', 'GRADIO_AVAILABLE'
] 