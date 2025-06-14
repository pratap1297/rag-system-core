"""
ServiceNow Integration Main Class
Main orchestrator for ServiceNow integration with RAG system
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .connector import ServiceNowConnector
from .processor import ServiceNowTicketProcessor, ProcessedTicket
from .scheduler import ServiceNowScheduler
from ...core.error_handling import IntegrationError

class ServiceNowIntegration:
    """Main ServiceNow integration class for RAG system"""
    
    def __init__(self, config_manager=None, ingestion_engine=None):
        """Initialize ServiceNow integration"""
        self.config_manager = config_manager
        self.ingestion_engine = ingestion_engine
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.connector = ServiceNowConnector(config_manager)
        self.processor = ServiceNowTicketProcessor(config_manager)
        self.scheduler = ServiceNowScheduler(config_manager, ingestion_engine)
        
        # Integration state
        self.is_initialized = False
        self.last_sync_time = None
        
        self.logger.info("ServiceNow integration initialized")
    
    def initialize(self) -> bool:
        """Initialize and test the ServiceNow integration"""
        try:
            self.logger.info("Initializing ServiceNow integration...")
            
            # Test ServiceNow connection
            if not self.connector.test_connection():
                raise IntegrationError("ServiceNow connection test failed")
            
            self.is_initialized = True
            self.logger.info("ServiceNow integration initialized successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize ServiceNow integration: {str(e)}"
            self.logger.error(error_msg)
            raise IntegrationError(error_msg)
    
    def start_automated_sync(self) -> bool:
        """Start automated ServiceNow ticket synchronization"""
        try:
            if not self.is_initialized:
                self.initialize()
            
            self.scheduler.start_scheduler()
            self.logger.info("ServiceNow automated sync started")
            return True
            
        except Exception as e:
            error_msg = f"Failed to start ServiceNow automated sync: {str(e)}"
            self.logger.error(error_msg)
            raise IntegrationError(error_msg)
    
    def stop_automated_sync(self):
        """Stop automated ServiceNow ticket synchronization"""
        try:
            self.scheduler.stop_scheduler()
            self.logger.info("ServiceNow automated sync stopped")
        except Exception as e:
            self.logger.error(f"Error stopping ServiceNow sync: {str(e)}")
    
    def manual_sync(self, 
                   filters: Optional[Dict[str, Any]] = None,
                   ingest_immediately: bool = True) -> Dict[str, Any]:
        """Perform manual synchronization of ServiceNow tickets"""
        try:
            if not self.is_initialized:
                self.initialize()
            
            self.logger.info("Starting manual ServiceNow sync...")
            
            # Use scheduler's fetch method for consistency
            if filters:
                # Temporarily update scheduler config with custom filters
                original_config = self.scheduler.config.copy()
                self.scheduler.config.update(filters)
                
                try:
                    result = self.scheduler.fetch_and_process_incidents()
                finally:
                    # Restore original config
                    self.scheduler.config = original_config
            else:
                result = self.scheduler.fetch_and_process_incidents()
            
            self.last_sync_time = datetime.now().isoformat()
            
            self.logger.info(f"Manual ServiceNow sync completed: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Manual ServiceNow sync failed: {str(e)}"
            self.logger.error(error_msg)
            raise IntegrationError(error_msg)
    
    def sync_specific_incident(self, incident_number: str) -> Dict[str, Any]:
        """Sync a specific incident by number"""
        try:
            if not self.is_initialized:
                self.initialize()
            
            self.logger.info(f"Syncing specific incident: {incident_number}")
            
            # Fetch the specific incident
            incident = self.connector.get_incident_by_number(incident_number)
            if not incident:
                return {
                    'status': 'not_found',
                    'incident_number': incident_number,
                    'message': 'Incident not found in ServiceNow'
                }
            
            # Process the incident
            processed_ticket = self.processor.process_incident(incident)
            if not processed_ticket:
                return {
                    'status': 'processing_failed',
                    'incident_number': incident_number,
                    'message': 'Failed to process incident'
                }
            
            # Ingest if ingestion engine is available
            ingested = False
            if self.ingestion_engine:
                try:
                    ingested_count = self.scheduler._ingest_tickets([processed_ticket])
                    ingested = ingested_count > 0
                except Exception as e:
                    self.logger.error(f"Error ingesting specific incident: {e}")
            
            return {
                'status': 'success',
                'incident_number': incident_number,
                'processed': True,
                'ingested': ingested,
                'ticket_data': {
                    'title': processed_ticket.title,
                    'priority': processed_ticket.metadata.get('priority_label'),
                    'state': processed_ticket.metadata.get('state_label'),
                    'is_network_related': processed_ticket.metadata.get('is_network_related', False)
                }
            }
            
        except Exception as e:
            error_msg = f"Failed to sync specific incident {incident_number}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'incident_number': incident_number,
                'message': error_msg
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        try:
            connection_status = self.connector.test_connection() if self.is_initialized else False
            scheduler_stats = self.scheduler.get_statistics()
            
            return {
                'initialized': self.is_initialized,
                'connection_healthy': connection_status,
                'scheduler_running': scheduler_stats.get('is_running', False),
                'last_sync_time': self.last_sync_time,
                'statistics': scheduler_stats,
                'connection_info': self.connector.get_connection_info() if self.is_initialized else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting integration status: {e}")
            return {
                'initialized': self.is_initialized,
                'connection_healthy': False,
                'scheduler_running': False,
                'error': str(e)
            }
    
    def get_recent_tickets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently processed tickets from cache"""
        try:
            import sqlite3
            
            with sqlite3.connect(self.scheduler.db_path) as conn:
                cursor = conn.execute('''
                    SELECT sys_id, number, data, content_hash, fetched_at, 
                           updated_at, ingested, ingestion_result
                    FROM incidents_cache 
                    ORDER BY fetched_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                tickets = []
                
                for row in cursor.fetchall():
                    ticket_dict = dict(zip(columns, row))
                    # Parse the stored data (simplified)
                    try:
                        import ast
                        incident_data = ast.literal_eval(ticket_dict['data'])
                        ticket_dict['short_description'] = incident_data.get('short_description', '')
                        ticket_dict['priority'] = incident_data.get('priority', '')
                        ticket_dict['state'] = incident_data.get('state', '')
                    except:
                        pass
                    
                    tickets.append(ticket_dict)
                
                return tickets
                
        except Exception as e:
            self.logger.error(f"Error getting recent tickets: {e}")
            return []
    
    def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get synchronization history"""
        try:
            return self.scheduler.get_fetch_history(limit)
        except Exception as e:
            self.logger.error(f"Error getting sync history: {e}")
            return []
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> bool:
        """Update ServiceNow integration configuration"""
        try:
            # Update scheduler configuration
            self.scheduler.config.update(config_updates)
            
            # If scheduler is running and critical settings changed, restart it
            critical_settings = ['fetch_interval_minutes', 'enabled']
            if (self.scheduler.is_running and 
                any(key in config_updates for key in critical_settings)):
                
                self.logger.info("Restarting scheduler due to configuration changes")
                self.scheduler.stop_scheduler()
                if config_updates.get('enabled', True):
                    self.scheduler.start_scheduler()
            
            self.logger.info(f"ServiceNow configuration updated: {config_updates}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False
    
    def test_integration(self) -> Dict[str, Any]:
        """Test all aspects of the ServiceNow integration"""
        results = {
            'connection_test': False,
            'fetch_test': False,
            'processing_test': False,
            'ingestion_test': False,
            'overall_status': 'failed',
            'details': {}
        }
        
        try:
            # Test connection
            results['connection_test'] = self.connector.test_connection()
            results['details']['connection'] = 'Success' if results['connection_test'] else 'Failed'
            
            if results['connection_test']:
                # Test fetching a small number of incidents
                try:
                    test_incidents = self.connector.get_incidents(limit=2)
                    results['fetch_test'] = len(test_incidents) >= 0  # Even 0 is okay for test
                    results['details']['fetch'] = f"Fetched {len(test_incidents)} incidents"
                except Exception as e:
                    results['details']['fetch'] = f"Fetch failed: {str(e)}"
                
                # Test processing if we have incidents
                if results['fetch_test'] and test_incidents:
                    try:
                        processed = self.processor.process_incidents(test_incidents[:1])
                        results['processing_test'] = len(processed) > 0
                        results['details']['processing'] = f"Processed {len(processed)} tickets"
                    except Exception as e:
                        results['details']['processing'] = f"Processing failed: {str(e)}"
                    
                    # Test ingestion if we have processed tickets and ingestion engine
                    if (results['processing_test'] and processed and self.ingestion_engine):
                        try:
                            # Don't actually ingest for test, just validate the process
                            document = processed[0].to_document()
                            results['ingestion_test'] = 'content' in document and 'metadata' in document
                            results['details']['ingestion'] = 'Document format validation passed'
                        except Exception as e:
                            results['details']['ingestion'] = f"Ingestion test failed: {str(e)}"
            
            # Determine overall status
            if results['connection_test'] and results['fetch_test']:
                if results['processing_test']:
                    results['overall_status'] = 'excellent' if results['ingestion_test'] else 'good'
                else:
                    results['overall_status'] = 'partial'
            
            return results
            
        except Exception as e:
            results['details']['error'] = str(e)
            return results
    
    def cleanup(self):
        """Cleanup integration resources"""
        try:
            self.stop_automated_sync()
            self.logger.info("ServiceNow integration cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup() 