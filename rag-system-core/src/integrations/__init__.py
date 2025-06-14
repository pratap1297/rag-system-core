"""
Integration modules for external systems
"""

try:
    from .servicenow.connector import ServiceNowConnector
    from .servicenow.processor import ServiceNowTicketProcessor
    __all__ = ['ServiceNowConnector', 'ServiceNowTicketProcessor']
except ImportError:
    # ServiceNow integration not available
    __all__ = [] 