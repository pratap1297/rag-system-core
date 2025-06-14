"""
ServiceNow UI Components
Provides Gradio interface for ServiceNow ticket management with pagination and selective ingestion
"""
import gradio as gr
import pandas as pd
import requests
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os

class ServiceNowUI:
    """ServiceNow ticket management UI"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.current_page = 1
        self.page_size = 10
        self.total_tickets = 0
        self.cached_tickets = []
        self.selected_tickets = set()
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.request(method, url, timeout=30, **kwargs)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}
    
    def fetch_servicenow_tickets(self, page: int = 1, limit: int = 10, 
                                priority_filter: str = "", state_filter: str = "",
                                category_filter: str = "") -> Tuple[str, str, str]:
        """Fetch ServiceNow tickets with pagination and filters"""
        try:
            # Build filters
            filters = {}
            if priority_filter and priority_filter != "All":
                filters['priority'] = priority_filter
            if state_filter and state_filter != "All":
                filters['state'] = state_filter
            if category_filter and category_filter != "All":
                filters['category'] = category_filter
            
            # Make API request to fetch tickets
            params = {
                'limit': limit * 2,  # Fetch more to account for filtering
                'offset': (page - 1) * limit
            }
            
            # Use ServiceNow connector to fetch tickets
            tickets = []
            try:
                from ..integrations.servicenow.connector import ServiceNowConnector
                
                connector = ServiceNowConnector()
                # Test connection first
                if connector.test_connection():
                    # Use correct method name: get_incidents instead of fetch_incidents
                    tickets = connector.get_incidents(
                        filters=filters,
                        limit=params['limit'],
                        offset=params['offset']
                    )
                else:
                    # Connection failed, use sample data
                    tickets = self._get_sample_tickets()
                    
            except (ImportError, ValueError, Exception) as e:
                # ServiceNow integration not available or credentials missing, use sample data
                print(f"ServiceNow connection failed: {e}")
                tickets = self._get_sample_tickets()
            
            # Apply additional filtering if needed
            if filters:
                tickets = self._apply_filters(tickets, filters)
            
            self.cached_tickets = tickets
            self.total_tickets = len(tickets)
            
            # Apply pagination
            start_idx = (page - 1) * limit
            end_idx = start_idx + limit
            page_tickets = tickets[start_idx:end_idx]
            
            # Format tickets table
            if not page_tickets:
                return "📭 No ServiceNow tickets found", "", self._get_pagination_info(page, limit)
            
            # Create table data
            table_data = []
            for i, ticket in enumerate(page_tickets, start=start_idx + 1):
                table_data.append([
                    f"☐ {i}",  # Checkbox placeholder
                    ticket.get('number', 'N/A'),
                    ticket.get('short_description', 'N/A')[:50] + '...' if len(ticket.get('short_description', '')) > 50 else ticket.get('short_description', 'N/A'),
                    self._get_priority_label(ticket.get('priority', '5')),
                    self._get_state_label(ticket.get('state', '1')),
                    ticket.get('category', 'N/A'),
                    ticket.get('sys_created_on', 'N/A')[:10] if ticket.get('sys_created_on') else 'N/A'
                ])
            
            df = pd.DataFrame(table_data, columns=[
                'Select', 'Ticket #', 'Description', 'Priority', 'State', 'Category', 'Created'
            ])
            
            tickets_table = df.to_string(index=False)
            
            # Create selection checkboxes
            checkboxes_html = self._create_ticket_checkboxes(page_tickets, start_idx)
            
            return tickets_table, checkboxes_html, self._get_pagination_info(page, limit)
            
        except Exception as e:
            return f"❌ Error fetching tickets: {str(e)}", "", ""
    
    def _get_priority_label(self, priority: str) -> str:
        """Convert priority number to label"""
        priority_map = {
            '1': '🔴 Critical',
            '2': '🟠 High', 
            '3': '🟡 Moderate',
            '4': '🔵 Low',
            '5': '⚪ Planning'
        }
        return priority_map.get(str(priority), f'❓ {priority}')
    
    def _get_state_label(self, state: str) -> str:
        """Convert state number to label"""
        state_map = {
            '1': '🆕 New',
            '2': '🔄 In Progress',
            '3': '⏸️ On Hold',
            '6': '✅ Resolved',
            '7': '❌ Closed'
        }
        return state_map.get(str(state), f'❓ {state}')
    
    def _create_ticket_checkboxes(self, tickets: List[Dict], start_idx: int) -> str:
        """Create HTML checkboxes for ticket selection"""
        checkboxes = []
        for i, ticket in enumerate(tickets):
            ticket_id = ticket.get('sys_id', f'ticket_{start_idx + i}')
            ticket_number = ticket.get('number', 'N/A')
            checked = 'checked' if ticket_id in self.selected_tickets else ''
            
            checkbox_html = f"""
            <div style="margin: 5px 0;">
                <input type="checkbox" id="{ticket_id}" name="ticket_selection" value="{ticket_id}" {checked}>
                <label for="{ticket_id}" style="margin-left: 8px;">
                    <strong>{ticket_number}</strong> - {ticket.get('short_description', 'N/A')[:60]}...
                </label>
            </div>
            """
            checkboxes.append(checkbox_html)
        
        return f"""
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
            <h4>📋 Select Tickets to Ingest:</h4>
            {''.join(checkboxes)}
        </div>
        """
    
    def _get_pagination_info(self, page: int, limit: int) -> str:
        """Get pagination information"""
        total_pages = (self.total_tickets + limit - 1) // limit
        start_item = (page - 1) * limit + 1
        end_item = min(page * limit, self.total_tickets)
        
        return f"""
        📄 **Page {page} of {total_pages}** | 
        Showing {start_item}-{end_item} of {self.total_tickets} tickets
        """
    
    def update_ticket_selection(self, selected_ids: str) -> str:
        """Update selected tickets based on checkbox input"""
        try:
            if selected_ids:
                # Parse selected ticket IDs (comma-separated)
                self.selected_tickets = set(selected_ids.split(','))
            else:
                self.selected_tickets = set()
            
            return f"✅ Selected {len(self.selected_tickets)} tickets for ingestion"
        except Exception as e:
            return f"❌ Error updating selection: {str(e)}"
    
    def ingest_selected_tickets(self) -> str:
        """Ingest selected ServiceNow tickets into the RAG system"""
        try:
            if not self.selected_tickets:
                return "⚠️ No tickets selected for ingestion"
            
            # Find selected tickets from cached data
            tickets_to_ingest = []
            for ticket in self.cached_tickets:
                if ticket.get('sys_id') in self.selected_tickets:
                    tickets_to_ingest.append(ticket)
            
            if not tickets_to_ingest:
                return "❌ Selected tickets not found in cache"
            
            # Process and ingest tickets
            try:
                from ..integrations.servicenow.processor import ServiceNowProcessor
                processor = ServiceNowProcessor()
            except ImportError:
                return "❌ ServiceNow processor not available"
            
            ingestion_results = []
            
            for ticket in tickets_to_ingest:
                try:
                    # Process ticket into document format
                    document = processor.process_incident(ticket)
                    
                    # Ingest into RAG system
                    response = self._make_request(
                        "POST", 
                        "/ingest/text",
                        json={
                            "text": document['content'],
                            "metadata": document['metadata']
                        }
                    )
                    
                    if "error" not in response:
                        ingestion_results.append(f"✅ {ticket.get('number', 'Unknown')}")
                    else:
                        ingestion_results.append(f"❌ {ticket.get('number', 'Unknown')}: {response['error']}")
                        
                except Exception as e:
                    ingestion_results.append(f"❌ {ticket.get('number', 'Unknown')}: {str(e)}")
            
            # Clear selection after ingestion
            self.selected_tickets = set()
            
            success_count = len([r for r in ingestion_results if r.startswith('✅')])
            total_count = len(ingestion_results)
            
            result_summary = f"""
## 🎯 Ingestion Results

**Summary:** {success_count}/{total_count} tickets successfully ingested

**Details:**
{chr(10).join(ingestion_results)}

**Next Steps:**
- Query the ingested tickets using the main chat interface
- Use filters like "ServiceNow ticket INC0000039" or "network incidents"
- Check the document management section for ingested tickets
"""
            
            return result_summary
            
        except Exception as e:
            return f"❌ Error during ingestion: {str(e)}"
    
    def get_servicenow_stats(self) -> str:
        """Get ServiceNow integration statistics"""
        try:
            # Get overall system stats
            stats_response = self._make_request("GET", "/stats")
            
            if "error" in stats_response:
                return f"❌ Error getting stats: {stats_response['error']}"
            
            # Get ServiceNow specific stats
            servicenow_response = self._make_request("GET", "/api/servicenow/status")
            
            servicenow_stats = "📊 ServiceNow integration status not available"
            if "error" not in servicenow_response:
                servicenow_stats = f"""
**ServiceNow Integration:**
- Status: {servicenow_response.get('status', 'Unknown')}
- Last Sync: {servicenow_response.get('last_sync', 'Never')}
- Tickets Synced: {servicenow_response.get('tickets_synced', 0)}
"""
            
            # Count ServiceNow documents in vector store
            docs_response = self._make_request("GET", "/manage/documents", params={"limit": 100})
            servicenow_docs = 0
            
            if "error" not in docs_response:
                for doc in docs_response:
                    metadata = doc.get('metadata', {})
                    if metadata.get('source') == 'servicenow':
                        servicenow_docs += 1
            
            return f"""
## 📈 ServiceNow Integration Statistics

**System Overview:**
- Total Documents: {stats_response.get('documents', {}).get('total', 0)}
- Total Vectors: {stats_response.get('vectors', {}).get('total', 0)}
- ServiceNow Documents: {servicenow_docs}

{servicenow_stats}

**Available Actions:**
- Fetch and browse ServiceNow tickets
- Select specific tickets for ingestion
- Query ingested tickets through main interface
"""
            
        except Exception as e:
            return f"❌ Error getting ServiceNow stats: {str(e)}"

    def _get_sample_tickets(self) -> List[Dict[str, Any]]:
        """Get comprehensive sample ServiceNow tickets for testing"""
        from datetime import datetime, timedelta
        
        base_date = datetime.now()
        
        sample_tickets = [
            {
                'sys_id': 'sample_001',
                'number': 'INC0000039',
                'short_description': 'Trouble getting to Oregon mail server',
                'description': 'Users in the Portland office are unable to access the Oregon mail server. Email synchronization is failing and users cannot send or receive emails. This appears to be a network connectivity issue affecting the entire Portland office.',
                'priority': '2',
                'state': '2',
                'category': 'network',
                'subcategory': 'email',
                'assigned_to': 'John Smith',
                'sys_created_on': (base_date - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_002',
                'number': 'INC0010035',
                'short_description': 'Router hardware malfunction - Cisco ISR 4351',
                'description': 'The main router (Cisco ISR 4351) in the data center is experiencing hardware failures. Multiple interface cards are showing errors and the device is randomly rebooting. This is affecting network connectivity for the entire building.',
                'priority': '1',
                'state': '1',
                'category': 'hardware',
                'subcategory': 'network_equipment',
                'assigned_to': 'Network Team',
                'sys_created_on': (base_date - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_003',
                'number': 'INC0010034',
                'short_description': 'Network performance degradation - Cisco ISR 1100-4G',
                'description': 'Users are reporting slow network performance and intermittent connectivity issues. The Cisco ISR 1100-4G router is showing high CPU utilization and memory usage. Network latency has increased significantly.',
                'priority': '3',
                'state': '2',
                'category': 'network',
                'subcategory': 'performance',
                'assigned_to': 'Sarah Johnson',
                'sys_created_on': (base_date - timedelta(days=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_004',
                'number': 'INC0010036',
                'short_description': 'VPN connection issues for remote workers',
                'description': 'Multiple remote workers are unable to establish VPN connections to the corporate network. The VPN server appears to be rejecting authentication requests and connection attempts are timing out.',
                'priority': '2',
                'state': '1',
                'category': 'network',
                'subcategory': 'vpn',
                'assigned_to': 'IT Security Team',
                'sys_created_on': (base_date - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_005',
                'number': 'INC0010037',
                'short_description': 'Firewall blocking legitimate traffic',
                'description': 'The corporate firewall is incorrectly blocking legitimate business traffic to external partners. Several important business applications are unable to communicate with external services, affecting daily operations.',
                'priority': '2',
                'state': '3',
                'category': 'network',
                'subcategory': 'firewall',
                'assigned_to': 'Network Security',
                'sys_created_on': (base_date - timedelta(days=1, hours=4)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_006',
                'number': 'INC0010038',
                'short_description': 'WiFi connectivity issues in conference rooms',
                'description': 'WiFi access points in conference rooms 201-205 are not providing stable connections. Users experience frequent disconnections during meetings and video conferences are being disrupted.',
                'priority': '3',
                'state': '2',
                'category': 'network',
                'subcategory': 'wireless',
                'assigned_to': 'Facilities IT',
                'sys_created_on': (base_date - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_007',
                'number': 'INC0010039',
                'short_description': 'Server hardware failure - Dell PowerEdge R740',
                'description': 'Critical database server (Dell PowerEdge R740) has experienced a hardware failure. Multiple hard drives in the RAID array have failed simultaneously, and the server is currently offline affecting multiple business applications.',
                'priority': '1',
                'state': '1',
                'category': 'hardware',
                'subcategory': 'server',
                'assigned_to': 'Server Team',
                'sys_created_on': (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_008',
                'number': 'INC0010040',
                'short_description': 'Software license expiration - Microsoft Office',
                'description': 'Microsoft Office licenses for 50 users are expiring next week. Users are receiving license expiration warnings and some features are already being disabled. Need to renew licenses urgently.',
                'priority': '3',
                'state': '1',
                'category': 'software',
                'subcategory': 'licensing',
                'assigned_to': 'Software Management',
                'sys_created_on': (base_date - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_009',
                'number': 'INC0010041',
                'short_description': 'Database performance issues - SQL Server',
                'description': 'The main SQL Server database is experiencing severe performance issues. Query response times have increased by 300% and several business applications are timing out when trying to access data.',
                'priority': '2',
                'state': '2',
                'category': 'software',
                'subcategory': 'database',
                'assigned_to': 'Database Team',
                'sys_created_on': (base_date - timedelta(hours=10)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_010',
                'number': 'INC0010042',
                'short_description': 'Printer network connectivity issues',
                'description': 'Network printers on floors 3 and 4 are not responding to print jobs. The printers appear to be connected to the network but are not accepting print requests from user workstations.',
                'priority': '4',
                'state': '1',
                'category': 'hardware',
                'subcategory': 'printer',
                'assigned_to': 'Desktop Support',
                'sys_created_on': (base_date - timedelta(days=1, hours=8)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_011',
                'number': 'INC0010043',
                'short_description': 'Email server storage capacity warning',
                'description': 'The email server storage is at 95% capacity. Users may start experiencing issues with sending and receiving emails if storage is not expanded soon. Need to add additional storage or archive old emails.',
                'priority': '3',
                'state': '1',
                'category': 'software',
                'subcategory': 'email',
                'assigned_to': 'Email Admin',
                'sys_created_on': (base_date - timedelta(days=2, hours=6)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(days=1, hours=2)).strftime('%Y-%m-%d %H:%M:%S')
            },
            {
                'sys_id': 'sample_012',
                'number': 'INC0010044',
                'short_description': 'Security software update required',
                'description': 'Critical security updates are available for the antivirus software deployed across all workstations. The current version has known vulnerabilities that need to be patched immediately.',
                'priority': '2',
                'state': '1',
                'category': 'software',
                'subcategory': 'security',
                'assigned_to': 'Security Team',
                'sys_created_on': (base_date - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S'),
                'sys_updated_on': (base_date - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            }
        ]
        
        return sample_tickets

    def _apply_filters(self, tickets: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to ticket list"""
        filtered_tickets = tickets
        
        if 'priority' in filters:
            priority = str(filters['priority'])
            filtered_tickets = [t for t in filtered_tickets if str(t.get('priority', '5')) == priority]
        
        if 'state' in filters:
            state = str(filters['state'])
            filtered_tickets = [t for t in filtered_tickets if str(t.get('state', '1')) == state]
        
        if 'category' in filters:
            category = filters['category'].lower()
            filtered_tickets = [t for t in filtered_tickets if t.get('category', '').lower() == category]
        
        return filtered_tickets

def create_servicenow_interface() -> gr.Blocks:
    """Create the ServiceNow management interface"""
    
    servicenow_ui = ServiceNowUI()
    
    with gr.Blocks(title="ServiceNow Ticket Management") as interface:
        gr.Markdown("# 🎫 ServiceNow Ticket Management")
        gr.Markdown("Browse, filter, and selectively ingest ServiceNow tickets into your RAG system")
        
        with gr.Tab("📋 Browse Tickets"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🔍 Filters")
                    priority_filter = gr.Dropdown(
                        choices=["All", "1", "2", "3", "4", "5"],
                        value="All",
                        label="Priority Filter"
                    )
                    state_filter = gr.Dropdown(
                        choices=["All", "1", "2", "3", "6", "7"],
                        value="All", 
                        label="State Filter"
                    )
                    category_filter = gr.Dropdown(
                        choices=["All", "network", "hardware", "software", "inquiry"],
                        value="All",
                        label="Category Filter"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Pagination")
                    current_page = gr.Number(value=1, label="Page", precision=0)
                    page_size = gr.Number(value=10, label="Items per page", precision=0)
                    
            fetch_btn = gr.Button("🔄 Fetch Tickets", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    tickets_table = gr.Textbox(
                        label="📋 ServiceNow Tickets",
                        lines=15,
                        max_lines=20,
                        interactive=False
                    )
                    
                with gr.Column(scale=1):
                    pagination_info = gr.Markdown("📄 Pagination info will appear here")
        
        with gr.Tab("✅ Select & Ingest"):
            gr.Markdown("### 🎯 Ticket Selection")
            ticket_checkboxes = gr.HTML(label="Select Tickets")
            
            with gr.Row():
                selected_ids = gr.Textbox(
                    label="Selected Ticket IDs (comma-separated)",
                    placeholder="Enter ticket IDs manually or use checkboxes above",
                    lines=2
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
        
        with gr.Tab("📊 Statistics"):
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
        
        # Auto-load stats on interface load
        interface.load(
            fn=servicenow_ui.get_servicenow_stats,
            inputs=[],
            outputs=[stats_display]
        )
    
    return interface

if __name__ == "__main__":
    # For testing the interface standalone
    interface = create_servicenow_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861) 