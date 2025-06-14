"""
ServiceNow Scheduler for RAG System
Automated ServiceNow incident fetching and processing for RAG ingestion
"""

import asyncio
import schedule
import time
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .connector import ServiceNowConnector
from .processor import ServiceNowTicketProcessor, ProcessedTicket
from ...core.error_handling import IntegrationError

class ServiceNowScheduler:
    """ServiceNow incident scheduler with RAG system integration"""
    
    def __init__(self, config_manager=None, ingestion_engine=None):
        """Initialize the ServiceNow scheduler"""
        self.config_manager = config_manager
        self.ingestion_engine = ingestion_engine
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.config = self._load_config()
        
        # Initialize components
        self.connector = ServiceNowConnector(config_manager)
        self.processor = ServiceNowTicketProcessor(config_manager)
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup database for caching and tracking
        self._setup_database()
        
        # Callbacks for processed tickets
        self.ticket_callbacks: List[Callable] = []
        
        # Statistics
        self.stats = {
            'total_fetched': 0,
            'total_processed': 0,
            'total_ingested': 0,
            'last_fetch_time': None,
            'last_error': None
        }
        
        self.logger.info("ServiceNow scheduler initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load ServiceNow configuration"""
        default_config = {
            'enabled': True,
            'fetch_interval_minutes': 15,
            'batch_size': 100,
            'max_incidents_per_fetch': 1000,
            'priority_filter': ['1', '2', '3'],  # Critical, High, Moderate
            'state_filter': ['1', '2', '3'],     # New, In Progress, On Hold
            'days_back': 7,
            'network_only': False,
            'auto_ingest': True,
            'cache_enabled': True,
            'cache_ttl_hours': 1
        }
        
        # Try to get ServiceNow config from system config
        if self.config_manager:
            try:
                system_config = self.config_manager.get_config()
                servicenow_config = getattr(system_config, 'servicenow', {})
                if isinstance(servicenow_config, dict):
                    default_config.update(servicenow_config)
            except Exception as e:
                self.logger.warning(f"Could not load ServiceNow config from system: {e}")
        
        return default_config
    
    def _setup_database(self):
        """Setup SQLite database for caching and tracking"""
        self.db_path = Path('data/servicenow_cache.db')
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Incidents cache table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS incidents_cache (
                    sys_id TEXT PRIMARY KEY,
                    number TEXT UNIQUE,
                    data TEXT,
                    content_hash TEXT,
                    fetched_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    ingested BOOLEAN DEFAULT FALSE,
                    ingestion_result TEXT
                )
            ''')
            
            # Fetch history table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fetch_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_time TIMESTAMP,
                    incidents_fetched INTEGER,
                    incidents_processed INTEGER,
                    incidents_ingested INTEGER,
                    new_incidents INTEGER,
                    updated_incidents INTEGER,
                    errors TEXT,
                    duration_seconds REAL
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_incidents_updated ON incidents_cache(updated_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_incidents_number ON incidents_cache(number)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_fetch_history_time ON fetch_history(fetch_time)')
    
    def add_ticket_callback(self, callback: Callable[[List[ProcessedTicket]], None]):
        """Add callback function for processed tickets"""
        self.ticket_callbacks.append(callback)
        self.logger.info(f"Added ticket callback: {callback.__name__}")
    
    def _cache_incident(self, incident: Dict[str, Any], processed_ticket: ProcessedTicket):
        """Cache incident data"""
        if not self.config['cache_enabled']:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO incidents_cache 
                    (sys_id, number, data, content_hash, fetched_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    incident['sys_id'],
                    incident['number'],
                    str(incident),  # Store as string for simplicity
                    processed_ticket.hash,
                    datetime.now().isoformat(),
                    incident.get('sys_updated_on', datetime.now().isoformat())
                ))
        except Exception as e:
            self.logger.error(f"Error caching incident {incident.get('number')}: {e}")
    
    def _is_incident_cached_and_unchanged(self, incident: Dict[str, Any]) -> bool:
        """Check if incident is cached and unchanged"""
        if not self.config['cache_enabled']:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT content_hash, updated_at FROM incidents_cache 
                    WHERE sys_id = ?
                ''', (incident['sys_id'],))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                cached_hash, cached_updated = result
                
                # Generate current hash
                current_hash = self.processor.process_incident(incident)
                if not current_hash:
                    return False
                
                return cached_hash == current_hash.hash
        except Exception as e:
            self.logger.error(f"Error checking cache for incident {incident.get('number')}: {e}")
            return False
    
    def _record_ingestion_result(self, sys_id: str, success: bool, result: str):
        """Record ingestion result in cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE incidents_cache 
                    SET ingested = ?, ingestion_result = ?
                    WHERE sys_id = ?
                ''', (success, result, sys_id))
        except Exception as e:
            self.logger.error(f"Error recording ingestion result: {e}")
    
    def fetch_and_process_incidents(self) -> Dict[str, Any]:
        """Fetch and process incidents from ServiceNow"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting ServiceNow incident fetch and processing...")
            
            # Build filters
            filters = {
                'priority': self.config['priority_filter'],
                'state': self.config['state_filter']
            }
            
            # Add date filter
            if self.config['days_back'] > 0:
                cutoff_date = datetime.now() - timedelta(days=self.config['days_back'])
                filters['updated_after'] = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add network filter if enabled
            if self.config['network_only']:
                filters['network_related'] = True
            
            # Fetch incidents
            incidents = self.connector.get_incidents(
                filters=filters,
                limit=self.config['max_incidents_per_fetch']
            )
            
            if not incidents:
                self.logger.info("No incidents fetched")
                return {
                    'incidents_fetched': 0,
                    'incidents_processed': 0,
                    'incidents_ingested': 0,
                    'new_incidents': 0,
                    'updated_incidents': 0,
                    'duration': time.time() - start_time
                }
            
            self.logger.info(f"Fetched {len(incidents)} incidents from ServiceNow")
            
            # Process incidents
            processed_tickets = []
            new_count = 0
            updated_count = 0
            
            for incident in incidents:
                # Check if incident is cached and unchanged
                if self._is_incident_cached_and_unchanged(incident):
                    continue
                
                processed_ticket = self.processor.process_incident(incident)
                if processed_ticket:
                    processed_tickets.append(processed_ticket)
                    
                    # Cache the incident
                    self._cache_incident(incident, processed_ticket)
                    
                    # Check if it's new or updated
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.execute(
                            'SELECT COUNT(*) FROM incidents_cache WHERE sys_id = ?',
                            (incident['sys_id'],)
                        )
                        if cursor.fetchone()[0] == 1:  # Just inserted
                            new_count += 1
                        else:
                            updated_count += 1
            
            self.logger.info(f"Processed {len(processed_tickets)} tickets")
            
            # Execute callbacks
            if processed_tickets and self.ticket_callbacks:
                for callback in self.ticket_callbacks:
                    try:
                        callback(processed_tickets)
                    except Exception as e:
                        self.logger.error(f"Error in ticket callback {callback.__name__}: {e}")
            
            # Auto-ingest if enabled
            ingested_count = 0
            if self.config['auto_ingest'] and self.ingestion_engine and processed_tickets:
                ingested_count = self._ingest_tickets(processed_tickets)
            
            # Update statistics
            self.stats.update({
                'total_fetched': self.stats['total_fetched'] + len(incidents),
                'total_processed': self.stats['total_processed'] + len(processed_tickets),
                'total_ingested': self.stats['total_ingested'] + ingested_count,
                'last_fetch_time': datetime.now().isoformat(),
                'last_error': None
            })
            
            # Record fetch history
            duration = time.time() - start_time
            self._record_fetch_history(
                len(incidents), len(processed_tickets), ingested_count,
                new_count, updated_count, "", duration
            )
            
            result = {
                'incidents_fetched': len(incidents),
                'incidents_processed': len(processed_tickets),
                'incidents_ingested': ingested_count,
                'new_incidents': new_count,
                'updated_incidents': updated_count,
                'duration': duration
            }
            
            self.logger.info(f"ServiceNow fetch completed: {result}")
            return result
            
        except Exception as e:
            error_msg = f"Error in ServiceNow fetch and process: {str(e)}"
            self.logger.error(error_msg)
            self.stats['last_error'] = error_msg
            
            # Record error in fetch history
            self._record_fetch_history(0, 0, 0, 0, 0, error_msg, time.time() - start_time)
            
            raise IntegrationError(error_msg)
    
    def _ingest_tickets(self, processed_tickets: List[ProcessedTicket]) -> int:
        """Ingest processed tickets into RAG system"""
        ingested_count = 0
        
        for ticket in processed_tickets:
            try:
                # Convert ticket to document format
                document = ticket.to_document()
                
                # Create a temporary file-like object for ingestion
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                    temp_file.write(document['content'])
                    temp_file_path = temp_file.name
                
                try:
                    # Ingest using the ingestion engine
                    result = self.ingestion_engine.ingest_file(
                        temp_file_path,
                        metadata=document['metadata']
                    )
                    
                    if result.get('status') == 'success':
                        ingested_count += 1
                        self._record_ingestion_result(ticket.ticket_id, True, "Success")
                        self.logger.debug(f"Successfully ingested ticket {ticket.ticket_number}")
                    else:
                        self._record_ingestion_result(ticket.ticket_id, False, str(result))
                        self.logger.warning(f"Failed to ingest ticket {ticket.ticket_number}: {result}")
                
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)
                    
            except Exception as e:
                error_msg = f"Error ingesting ticket {ticket.ticket_number}: {str(e)}"
                self.logger.error(error_msg)
                self._record_ingestion_result(ticket.ticket_id, False, error_msg)
        
        self.logger.info(f"Ingested {ingested_count} out of {len(processed_tickets)} tickets")
        return ingested_count
    
    def _record_fetch_history(self, fetched: int, processed: int, ingested: int,
                             new: int, updated: int, errors: str, duration: float):
        """Record fetch history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO fetch_history 
                    (fetch_time, incidents_fetched, incidents_processed, incidents_ingested,
                     new_incidents, updated_incidents, errors, duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    fetched, processed, ingested, new, updated, errors, duration
                ))
        except Exception as e:
            self.logger.error(f"Error recording fetch history: {e}")
    
    def start_scheduler(self):
        """Start the ServiceNow scheduler"""
        if not self.config['enabled']:
            self.logger.info("ServiceNow scheduler is disabled")
            return
        
        if self.is_running:
            self.logger.warning("ServiceNow scheduler is already running")
            return
        
        self.logger.info("Starting ServiceNow scheduler...")
        
        # Test connection first
        if not self.connector.test_connection():
            raise IntegrationError("Cannot start scheduler - ServiceNow connection failed")
        
        # Schedule the job
        schedule.every(self.config['fetch_interval_minutes']).minutes.do(
            self.fetch_and_process_incidents
        )
        
        # Start scheduler thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info(f"ServiceNow scheduler started (interval: {self.config['fetch_interval_minutes']} minutes)")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5)
    
    def stop_scheduler(self):
        """Stop the ServiceNow scheduler"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping ServiceNow scheduler...")
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        self.executor.shutdown(wait=True)
        
        self.logger.info("ServiceNow scheduler stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            **self.stats,
            'is_running': self.is_running,
            'config': self.config
        }
    
    def get_fetch_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent fetch history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT * FROM fetch_history 
                    ORDER BY fetch_time DESC 
                    LIMIT ?
                ''', (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error getting fetch history: {e}")
            return [] 