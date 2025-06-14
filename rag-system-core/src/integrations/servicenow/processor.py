"""
ServiceNow Ticket Processor
Processes ServiceNow incidents and converts them to RAG-compatible documents
"""

import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ProcessedTicket:
    """Processed ServiceNow ticket ready for RAG ingestion"""
    ticket_id: str
    ticket_number: str
    title: str
    content: str
    metadata: Dict[str, Any]
    hash: str
    
    def to_document(self) -> Dict[str, Any]:
        """Convert to document format for RAG ingestion"""
        return {
            'content': self.content,
            'metadata': {
                **self.metadata,
                'document_type': 'servicenow_ticket',
                'ticket_id': self.ticket_id,
                'ticket_number': self.ticket_number,
                'title': self.title,
                'content_hash': self.hash
            }
        }

class ServiceNowTicketProcessor:
    """Processes ServiceNow incidents for RAG system ingestion"""
    
    def __init__(self, config_manager=None):
        """Initialize the ticket processor"""
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Network-related keywords for classification
        self.network_keywords = [
            'network', 'router', 'switch', 'firewall', 'vpn', 'bgp', 'ospf', 'vlan',
            'ethernet', 'wifi', 'wireless', 'lan', 'wan', 'dns', 'dhcp', 'tcp',
            'ip', 'subnet', 'gateway', 'ping', 'traceroute', 'bandwidth',
            'latency', 'packet', 'cisco', 'juniper', 'arista', 'fortinet',
            'routing', 'switching', 'connectivity', 'interface', 'port'
        ]
        
        # Priority mappings
        self.priority_mapping = {
            '1': 'Critical',
            '2': 'High', 
            '3': 'Moderate',
            '4': 'Low',
            '5': 'Planning'
        }
        
        # State mappings
        self.state_mapping = {
            '1': 'New',
            '2': 'In Progress',
            '3': 'On Hold',
            '6': 'Resolved',
            '7': 'Closed',
            '8': 'Canceled'
        }
        
        self.logger.info("ServiceNow ticket processor initialized")
    
    def is_network_related(self, incident: Dict[str, Any]) -> bool:
        """Determine if an incident is network-related"""
        text_to_check = (
            incident.get('short_description', '').lower() + " " +
            incident.get('description', '').lower() + " " +
            incident.get('category', '').lower() + " " +
            incident.get('subcategory', '').lower() + " " +
            incident.get('u_configuration_item', '').lower() + " " +
            incident.get('business_service', '').lower()
        )
        
        return any(keyword in text_to_check for keyword in self.network_keywords)
    
    def extract_technical_details(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical details from incident"""
        details = {}
        
        # Extract IP addresses
        import re
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        text_content = f"{incident.get('description', '')} {incident.get('work_notes', '')}"
        
        ip_addresses = re.findall(ip_pattern, text_content)
        if ip_addresses:
            details['ip_addresses'] = list(set(ip_addresses))
        
        # Extract device names/hostnames
        hostname_pattern = r'\b[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+\b'
        hostnames = re.findall(hostname_pattern, text_content)
        if hostnames:
            details['hostnames'] = list(set(hostnames))
        
        # Extract error codes
        error_pattern = r'\b(?:error|err|code)\s*:?\s*([a-zA-Z0-9-_]+)\b'
        errors = re.findall(error_pattern, text_content, re.IGNORECASE)
        if errors:
            details['error_codes'] = list(set(errors))
        
        return details
    
    def process_incident(self, incident: Dict[str, Any]) -> Optional[ProcessedTicket]:
        """Process a single ServiceNow incident into a RAG document"""
        try:
            # Extract basic information
            sys_id = incident.get('sys_id', '')
            number = incident.get('number', '')
            short_description = incident.get('short_description', '')
            description = incident.get('description', '')
            
            if not all([sys_id, number, short_description]):
                self.logger.warning(f"Incomplete incident data for {number or sys_id}")
                return None
            
            # Build comprehensive content
            content_parts = [
                f"Incident Number: {number}",
                f"Title: {short_description}",
                ""
            ]
            
            # Priority and state
            priority = incident.get('priority', '')
            priority_label = self.priority_mapping.get(priority, f"Priority {priority}")
            state = incident.get('state', '')
            state_label = self.state_mapping.get(state, f"State {state}")
            
            content_parts.extend([
                f"Priority: {priority_label} ({priority})",
                f"Status: {state_label} ({state})",
                f"Category: {incident.get('category', 'N/A')}",
                f"Subcategory: {incident.get('subcategory', 'N/A')}",
                ""
            ])
            
            # Assignment information
            if incident.get('assigned_to'):
                assigned_to_display = incident.get('assigned_to', {})
                if isinstance(assigned_to_display, dict):
                    assigned_to_name = assigned_to_display.get('display_value', assigned_to_display.get('value', 'Unknown'))
                else:
                    assigned_to_name = str(assigned_to_display)
                content_parts.append(f"Assigned To: {assigned_to_name}")
            
            if incident.get('assignment_group'):
                assignment_group_display = incident.get('assignment_group', {})
                if isinstance(assignment_group_display, dict):
                    assignment_group_name = assignment_group_display.get('display_value', assignment_group_display.get('value', 'Unknown'))
                else:
                    assignment_group_name = str(assignment_group_display)
                content_parts.append(f"Assignment Group: {assignment_group_name}")
            
            content_parts.append("")
            
            # Detailed description
            if description:
                content_parts.extend([
                    "Description:",
                    description,
                    ""
                ])
            
            # Technical details
            if incident.get('u_configuration_item'):
                content_parts.append(f"Configuration Item: {incident.get('u_configuration_item')}")
            
            if incident.get('business_service'):
                content_parts.append(f"Business Service: {incident.get('business_service')}")
            
            if incident.get('location'):
                location_display = incident.get('location', {})
                if isinstance(location_display, dict):
                    location_name = location_display.get('display_value', location_display.get('value', 'Unknown'))
                else:
                    location_name = str(location_display)
                content_parts.append(f"Location: {location_name}")
            
            content_parts.append("")
            
            # Work notes and resolution
            if incident.get('work_notes'):
                content_parts.extend([
                    "Work Notes:",
                    incident.get('work_notes'),
                    ""
                ])
            
            if incident.get('close_notes'):
                content_parts.extend([
                    "Resolution Notes:",
                    incident.get('close_notes'),
                    ""
                ])
            
            # Timestamps
            content_parts.extend([
                f"Created: {incident.get('sys_created_on', '')}",
                f"Updated: {incident.get('sys_updated_on', '')}",
            ])
            
            if incident.get('resolved_at'):
                content_parts.append(f"Resolved: {incident.get('resolved_at')}")
            
            if incident.get('closed_at'):
                content_parts.append(f"Closed: {incident.get('closed_at')}")
            
            # Join content
            content = "\n".join(content_parts)
            
            # Extract technical details
            technical_details = self.extract_technical_details(incident)
            
            # Helper function to safely extract string values
            def safe_extract_string(value, default=''):
                """Safely extract string value from ServiceNow field"""
                if value is None:
                    return default
                if isinstance(value, dict):
                    return value.get('display_value', value.get('value', default))
                return str(value) if value else default
            
            # Build metadata with safe extraction
            metadata = {
                'source': 'ServiceNow',
                'source_type': 'servicenow_incident',
                'sys_id': sys_id or '',
                'number': number or '',
                'ticket_number': number or '',  # Add explicit ticket_number field
                'title': short_description or '',
                'priority': priority or '',
                'priority_label': priority_label or '',
                'state': state or '',
                'state_label': state_label or '',
                'category': safe_extract_string(incident.get('category')),
                'subcategory': safe_extract_string(incident.get('subcategory')),
                'is_network_related': self.is_network_related(incident),
                'created_on': safe_extract_string(incident.get('sys_created_on')),
                'updated_on': safe_extract_string(incident.get('sys_updated_on')),
                'resolved_at': safe_extract_string(incident.get('resolved_at')),
                'closed_at': safe_extract_string(incident.get('closed_at')),
                'assigned_to': safe_extract_string(incident.get('assigned_to')),
                'assignment_group': safe_extract_string(incident.get('assignment_group')),
                'caller': safe_extract_string(incident.get('caller_id')),
                'location': safe_extract_string(incident.get('location')),
                'business_service': safe_extract_string(incident.get('business_service')),
                'configuration_item': safe_extract_string(incident.get('u_configuration_item')),
                'impact': safe_extract_string(incident.get('impact')),
                'urgency': safe_extract_string(incident.get('urgency')),
                'ingested_at': datetime.now().isoformat()
            }
            
            # Add technical details safely
            if technical_details:
                for key, value in technical_details.items():
                    if value is not None:
                        metadata[key] = value if isinstance(value, (str, int, float, bool, list)) else str(value)
            
            # Generate content hash for change detection
            content_hash = hashlib.md5(
                f"{sys_id}{incident.get('sys_updated_on', '')}{content}".encode()
            ).hexdigest()
            
            processed_ticket = ProcessedTicket(
                ticket_id=sys_id,
                ticket_number=number,
                title=short_description,
                content=content,
                metadata=metadata,
                hash=content_hash
            )
            
            self.logger.debug(f"Processed incident {number} successfully")
            return processed_ticket
            
        except Exception as e:
            self.logger.error(f"Error processing incident {incident.get('number', 'unknown')}: {str(e)}")
            return None
    
    def process_incidents(self, incidents: List[Dict[str, Any]], 
                         filter_network_only: bool = False) -> List[ProcessedTicket]:
        """Process multiple incidents"""
        processed_tickets = []
        
        for incident in incidents:
            processed_ticket = self.process_incident(incident)
            if processed_ticket:
                # Apply network filter if requested
                if filter_network_only and not processed_ticket.metadata.get('is_network_related', False):
                    continue
                
                processed_tickets.append(processed_ticket)
        
        self.logger.info(f"Processed {len(processed_tickets)} tickets from {len(incidents)} incidents")
        return processed_tickets
    
    def get_processing_stats(self, processed_tickets: List[ProcessedTicket]) -> Dict[str, Any]:
        """Get statistics about processed tickets"""
        if not processed_tickets:
            return {'total': 0}
        
        network_related = sum(1 for t in processed_tickets if t.metadata.get('is_network_related', False))
        
        priorities = {}
        states = {}
        categories = {}
        
        for ticket in processed_tickets:
            # Priority stats
            priority = ticket.metadata.get('priority_label', 'Unknown')
            priorities[priority] = priorities.get(priority, 0) + 1
            
            # State stats
            state = ticket.metadata.get('state_label', 'Unknown')
            states[state] = states.get(state, 0) + 1
            
            # Category stats
            category = ticket.metadata.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total': len(processed_tickets),
            'network_related': network_related,
            'non_network': len(processed_tickets) - network_related,
            'priorities': priorities,
            'states': states,
            'categories': categories
        } 