"""
ServiceNow Connector for RAG System
Enhanced connector with RAG system integration capabilities
"""

import os
import json
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from ...core.error_handling import IntegrationError

class ServiceNowConnector:
    """ServiceNow API connector optimized for RAG system integration"""
    
    def __init__(self, config_manager=None):
        """Initialize ServiceNow connector with RAG system configuration"""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Get ServiceNow credentials from environment (matching your .env format)
        self.instance = os.getenv('SERVICENOW_INSTANCE')
        self.username = os.getenv('SERVICENOW_USERNAME') 
        self.password = os.getenv('SERVICENOW_PASSWORD')
        self.table = os.getenv('SERVICENOW_TABLE', 'incident')
        
        # Validate required credentials
        if not all([self.instance, self.username, self.password]):
            missing = []
            if not self.instance: missing.append('SERVICENOW_INSTANCE')
            if not self.username: missing.append('SERVICENOW_USERNAME')
            if not self.password: missing.append('SERVICENOW_PASSWORD')
            
            raise ValueError(f"Missing ServiceNow credentials: {', '.join(missing)}")
        
        # Build base URL - your instance format: dev319029.service-now.com
        if not self.instance.startswith('https://'):
            self.base_url = f"https://{self.instance}"
        else:
            self.base_url = self.instance
        
        # API endpoints
        self.incident_endpoint = f"{self.base_url}/api/now/table/{self.table}"
        
        # Session setup with enhanced configuration
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'RAG-System-ServiceNow-Integration/1.0'
        })
        
        # Connection timeout settings
        self.timeout = 30
        
        self.logger.info(f"ServiceNow connector initialized for instance: {self.instance}")
    
    def test_connection(self) -> bool:
        """Test connection to ServiceNow instance with enhanced error reporting"""
        try:
            self.logger.info("Testing ServiceNow connection...")
            
            response = self.session.get(
                self.incident_endpoint,
                params={'sysparm_limit': 1},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully connected to ServiceNow: {self.base_url}")
                return True
            else:
                self.logger.error(f"Connection failed. Status: {response.status_code}, Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.error("Connection timeout - ServiceNow instance may be slow or unreachable")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error - Check network connectivity and ServiceNow instance URL")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected connection error: {str(e)}")
            return False
    
    def get_incidents(self, 
                     filters: Optional[Dict[str, Any]] = None, 
                     limit: int = 100,
                     offset: int = 0,
                     fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get incidents from ServiceNow with comprehensive filtering and field selection
        
        Args:
            filters: Dictionary of filters to apply
            limit: Maximum number of incidents to retrieve
            offset: Number of records to skip
            fields: Specific fields to retrieve (None for all fields)
        
        Returns:
            List of incident dictionaries
        """
        try:
            params = {
                'sysparm_limit': limit,
                'sysparm_offset': offset
            }
            
            # Add field selection - use your .env SERVICENOW_FIELDS if available
            if fields:
                params['sysparm_fields'] = ','.join(fields)
            else:
                # Use fields from environment or default set
                env_fields = os.getenv('SERVICENOW_FIELDS')
                if env_fields:
                    params['sysparm_fields'] = env_fields
                else:
                    # Default essential fields
                    params['sysparm_fields'] = 'sys_id,number,short_description,description,state,priority,category,subcategory,assigned_to,sys_created_on,sys_updated_on'
            
            # Build query string from filters
            query_parts = []
            
            if filters:
                # Priority filter
                if 'priority' in filters:
                    priorities = filters['priority'] if isinstance(filters['priority'], list) else [filters['priority']]
                    priority_query = '^OR'.join([f'priority={p}' for p in priorities])
                    query_parts.append(f"({priority_query})")
                
                # State filter
                if 'state' in filters:
                    states = filters['state'] if isinstance(filters['state'], list) else [filters['state']]
                    state_query = '^OR'.join([f'state={s}' for s in states])
                    query_parts.append(f"({state_query})")
                
                # Category filter
                if 'category' in filters:
                    categories = filters['category'] if isinstance(filters['category'], list) else [filters['category']]
                    category_query = '^OR'.join([f'category={c}' for c in categories])
                    query_parts.append(f"({category_query})")
                
                # Date range filter
                if 'created_after' in filters:
                    query_parts.append(f"sys_created_on>={filters['created_after']}")
                
                if 'updated_after' in filters:
                    query_parts.append(f"sys_updated_on>={filters['updated_after']}")
                
                # Network-related filter
                if filters.get('network_related', False):
                    network_keywords = ['network', 'router', 'switch', 'firewall', 'vpn', 'wifi', 'ethernet']
                    network_query = '^OR'.join([
                        f'short_descriptionLIKE{keyword}^ORdescriptionLIKE{keyword}' 
                        for keyword in network_keywords
                    ])
                    query_parts.append(f"({network_query})")
            
            # Add environment query filter if specified
            env_query_filter = os.getenv('SERVICENOW_QUERY_FILTER')
            if env_query_filter:
                query_parts.append(env_query_filter)
            
            # Combine query parts
            if query_parts:
                params['sysparm_query'] = '^'.join(query_parts)
            
            self.logger.debug(f"Fetching incidents with params: {params}")
            
            response = self.session.get(
                self.incident_endpoint,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                incidents = response.json()['result']
                self.logger.info(f"Successfully retrieved {len(incidents)} incidents")
                return incidents
            else:
                self.logger.error(f"Failed to get incidents. Status: {response.status_code}, Response: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting incidents: {str(e)}")
            raise Exception(f"Failed to retrieve ServiceNow incidents: {str(e)}")
    
    def get_incident_by_number(self, incident_number: str) -> Optional[Dict[str, Any]]:
        """Get a specific incident by its number"""
        try:
            params = {
                'sysparm_query': f'number={incident_number}',
                'sysparm_limit': 1
            }
            
            response = self.session.get(
                self.incident_endpoint,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                incidents = response.json()['result']
                return incidents[0] if incidents else None
            else:
                self.logger.error(f"Failed to get incident {incident_number}. Status: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting incident {incident_number}: {str(e)}")
            return None
    
    def get_recent_incidents(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get incidents created or updated in the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        cutoff_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
        
        filters = {
            'updated_after': cutoff_str
        }
        
        return self.get_incidents(filters=filters, limit=1000)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information for diagnostics"""
        return {
            'instance': self.instance,
            'base_url': self.base_url,
            'username': self.username,
            'table': self.table,
            'connected': self.test_connection()
        } 