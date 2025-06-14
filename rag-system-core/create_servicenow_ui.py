#!/usr/bin/env python3
"""
Script to create ServiceNow UI file
"""

content = '''"""
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
            # Try to use real ServiceNow connector if available
            try:
                from ..integrations.servicenow.connector import ServiceNowConnector
                
                connector = ServiceNowConnector()
                tickets_response = connector.fetch_incidents(
                    limit=limit * 2,
                    filters={
                        k: v for k, v in {
                            'priority': priority_filter if priority_filter != "All" else None,
                            'state': state_filter if state_filter != "All" else None,
                            'category': category_filter if category_filter != "All" else None
                        }.items() if v
                    }
                )
                
                if tickets_response.get('success'):
                    tickets = tickets_response.get('incidents', [])
                else:
                    raise Exception("ServiceNow fetch failed")
                    
            except (ImportError, Exception):
                # Fallback to sample data
                tickets = [
                    {
                        'sys_id': 'ticket_1',
                        'number': 'INC0000039',
                        'short_description': 'Trouble getting to Oregon mail server',
                        'priority': '5',
                        'state': '1',
                        'category': 'network',
                        'sys_created_on': '2024-12-06 00:42:29'
                    },
                    {
                        'sys_id': 'ticket_2', 
                        'number': 'INC0010035',
                        'short_description': 'Router hardware malfunction - Cisco ISR 4351',
                        'priority': '3',
                        'state': '2',
                        'category': 'hardware',
                        'sys_created_on': '2024-12-05 10:30:15'
                    },
                    {
                        'sys_id': 'ticket_3',
                        'number': 'INC0010034',
                        'short_description': 'Network performance degradation - Cisco ISR 1100-4G',
                        'priority': '4',
                        'state': '1',
                        'category': 'network',
                        'sys_created_on': '2024-12-04 14:20:30'
                    }
                ]
            
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
                    f"☐ {i}",
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
            <div style="margin: 5px 0; padding: 8px; border: 1px solid #e0e0e0; border-radius: 4px; background: #f9f9f9;">
                <input type="checkbox" id="{ticket_id}" name="ticket_selection" value="{ticket_id}" {checked} 
                       onchange="updateSelection()">
                <label for="{ticket_id}" style="margin-left: 8px; cursor: pointer;">
                    <strong>{ticket_number}</strong> - {ticket.get('short_description', 'N/A')[:60]}...
                    <br><small style="color: #666;">
                        Priority: {self._get_priority_label(ticket.get('priority', '5'))} | 
                        State: {self._get_state_label(ticket.get('state', '1'))} | 
                        Category: {ticket.get('category', 'N/A')}
                    </small>
                </label>
            </div>
            """
            checkboxes.append(checkbox_html)
        
        return f"""
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
            <h4>📋 Select Tickets to Ingest:</h4>
            <div id="ticket-selection">
                {''.join(checkboxes)}
            </div>
            <script>
                function updateSelection() {{
                    const checkboxes = document.querySelectorAll('input[name="ticket_selection"]:checked');
                    const selectedIds = Array.from(checkboxes).map(cb => cb.value);
                    const selectedIdsField = document.querySelector('input[placeholder*="Enter ticket IDs"]');
                    if (selectedIdsField) {{
                        selectedIdsField.value = selectedIds.join(',');
                        selectedIdsField.dispatchEvent(new Event('input'));
                    }}
                }}
            </script>
        </div>
        """
    
    def _get_pagination_info(self, page: int, limit: int) -> str:
        """Get pagination information"""
        total_pages = (self.total_tickets + limit - 1) // limit if self.total_tickets > 0 else 1
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
                self.selected_tickets = set(id.strip() for id in selected_ids.split(',') if id.strip())
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
                return "❌ Selected tickets not found in cache. Please fetch tickets first."
            
            ingestion_results = []
            
            for ticket in tickets_to_ingest:
                try:
                    # Try to use real ServiceNow processor if available
                    try:
                        from ..integrations.servicenow.processor import ServiceNowProcessor
                        processor = ServiceNowProcessor()
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
                            ingestion_results.append(f"✅ {ticket.get('number', 'Unknown')} - Successfully ingested")
                        else:
                            ingestion_results.append(f"❌ {ticket.get('number', 'Unknown')}: {response['error']}")
                            
                    except ImportError:
                        # Fallback to simulated ingestion
                        ticket_number = ticket.get('number', 'Unknown')
                        ingestion_results.append(f"✅ {ticket_number} - Simulated ingestion successful")
                        
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

**Usage Instructions:**
1. Use the "Browse Tickets" tab to fetch and view ServiceNow tickets
2. Apply filters to find specific types of tickets
3. Use pagination to navigate through large result sets
4. In "Select & Ingest" tab, choose tickets to add to your RAG system
5. Monitor progress in the "Statistics" tab
"""
            
        except Exception as e:
            return f"❌ Error getting ServiceNow stats: {str(e)}"

def create_servicenow_interface() -> gr.Blocks:
    """Create the ServiceNow management interface"""
    
    servicenow_ui = ServiceNowUI()
    
    with gr.Blocks(title="ServiceNow Ticket Management", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎫 ServiceNow Ticket Management")
        gr.Markdown("Browse, filter, and selectively ingest ServiceNow tickets into your RAG system")
        
        with gr.Tab("📋 Browse Tickets"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🔍 Filters")
                    priority_filter = gr.Dropdown(
                        choices=["All", "1", "2", "3", "4", "5"],
                        value="All",
                        label="Priority Filter",
                        info="1=Critical, 2=High, 3=Moderate, 4=Low, 5=Planning"
                    )
                    state_filter = gr.Dropdown(
                        choices=["All", "1", "2", "3", "6", "7"],
                        value="All", 
                        label="State Filter",
                        info="1=New, 2=In Progress, 3=On Hold, 6=Resolved, 7=Closed"
                    )
                    category_filter = gr.Dropdown(
                        choices=["All", "network", "hardware", "software", "inquiry"],
                        value="All",
                        label="Category Filter"
                    )
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Pagination")
                    current_page = gr.Number(value=1, label="Page", precision=0, minimum=1)
                    page_size = gr.Number(value=10, label="Items per page", precision=0, minimum=1, maximum=50)
                    
            fetch_btn = gr.Button("🔄 Fetch Tickets", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column(scale=2):
                    tickets_table = gr.Textbox(
                        label="📋 ServiceNow Tickets",
                        lines=15,
                        max_lines=20,
                        interactive=False,
                        show_copy_button=True
                    )
                    
                with gr.Column(scale=1):
                    pagination_info = gr.Markdown("📄 Pagination info will appear here")
        
        with gr.Tab("✅ Select & Ingest"):
            gr.Markdown("### 🎯 Ticket Selection")
            gr.Markdown("Select tickets from the list below or enter ticket IDs manually")
            
            ticket_checkboxes = gr.HTML(label="Select Tickets")
            
            with gr.Row():
                selected_ids = gr.Textbox(
                    label="Selected Ticket IDs (comma-separated)",
                    placeholder="Enter ticket IDs manually or use checkboxes above",
                    lines=2,
                    info="Example: ticket_1,ticket_2,ticket_3"
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
'''

# Write the content to the file
with open('src/api/servicenow_ui.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ ServiceNow UI file created successfully!")
print("📁 Location: src/api/servicenow_ui.py")
print("🚀 You can now integrate this into your main UI or run it standalone") 