#!/usr/bin/env python3
"""
Comprehensive RAG System UI - Complete Document Lifecycle Management
===================================================================
Provides upload, update, delete document functionality with real-time query testing
"""

import sys
import os
import time
import requests
import gradio as gr
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List
import uuid

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class ComprehensiveRAGUI:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.document_registry = {}  # Track documents for lifecycle management
        print(f"DEBUG: ComprehensiveRAGUI initialized with API URL: {api_url}")
        print(f"DEBUG: Initial registry state: {self.document_registry}")
        
    def check_api_connection(self) -> str:
        """Check if the API is accessible"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                status = data.get('status', 'unknown')
                timestamp = data.get('timestamp', 'unknown')
                components = data.get('components', {})
                
                status_text = f"✅ **API Status: {status.upper()}**\n"
                status_text += f"🕐 Last Check: {timestamp}\n"
                status_text += f"🔧 Components: {len(components)} active\n"
                status_text += f"🌐 Backend URL: {self.api_url}"
                
                return status_text
            else:
                return f"❌ **API Error: HTTP {response.status_code}**"
        except Exception as e:
            return f"❌ **Connection Error:** {str(e)}"

    def upload_document(self, file, doc_path: str = "") -> Tuple[str, str]:
        """Upload a new document file to the system"""
        if file is None:
            return "❌ Please select a file to upload", ""
        
        try:
            # Use provided path or generate from filename
            if not doc_path.strip():
                doc_path = f"/docs/{Path(file.name).stem}"
            
            # Generate unique document ID
            doc_id = f"{doc_path}_{uuid.uuid4().hex[:8]}"
            
            # Read file content
            with open(file.name, 'rb') as f:
                file_content = f.read()
            
            # Prepare metadata
            metadata = {
                "doc_path": doc_path,
                "original_filename": Path(file.name).name,
                "doc_id": doc_id,
                "operation": "upload",
                "timestamp": datetime.now().isoformat(),
                "source": "comprehensive_ui",
                "file_size": len(file_content)
            }
            
            # Upload via API
            files = {'file': (Path(file.name).name, file_content, 'application/octet-stream')}
            data = {'metadata': json.dumps(metadata)}
            
            response = requests.post(
                f"{self.api_url}/upload",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Check if this was an update (old vectors deleted)
                is_update = result_data.get('is_update', False)
                old_vectors_deleted = result_data.get('old_vectors_deleted', 0)
                
                # Store in registry
                self.document_registry[doc_path] = {
                    "doc_id": doc_id,
                    "original_filename": Path(file.name).name,
                    "file_path": file.name,
                    "chunks_created": result_data.get('chunks_created', 0),
                    "last_updated": datetime.now().isoformat(),
                    "status": "updated" if is_update else "active",
                    "file_size": len(file_content),
                    "content_preview": str(file_content[:200], 'utf-8', errors='ignore') + "...",
                    "old_vectors_deleted": old_vectors_deleted
                }
                
                print(f"DEBUG: Added document to registry: {doc_path}")
                print(f"DEBUG: Registry now has {len(self.document_registry)} documents")
                
                result = f"✅ **Document {'Updated' if is_update else 'Uploaded'} Successfully!**\n\n"
                result += f"📄 **Document Path:** `{doc_path}`\n"
                result += f"📁 **Original File:** `{Path(file.name).name}`\n"
                result += f"🆔 **Document ID:** `{doc_id}`\n"
                result += f"📝 **Chunks Created:** {result_data.get('chunks_created', 0)}\n"
                result += f"🔢 **Embeddings Generated:** {result_data.get('embeddings_generated', 0)}\n"
                result += f"📊 **File Size:** {len(file_content)} bytes\n"
                result += f"📅 **{'Updated' if is_update else 'Uploaded'}:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                if is_update and old_vectors_deleted > 0:
                    result += f"🗑️ **Old Vectors Deleted:** {old_vectors_deleted}\n"
                    result += f"✨ **Vector Replacement:** Complete\n"
                elif is_update:
                    result += f"⚠️ **Note:** No old vectors found to replace\n"
                
                result += f"\n**Content Preview:** {str(file_content[:200], 'utf-8', errors='ignore')}..."
                
                # Update registry display
                registry_display = self._format_document_registry()
                
                return result, registry_display
            else:
                error_msg = f"❌ **Document Upload Failed**\n\n"
                error_msg += f"HTTP Status: {response.status_code}\n"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"Details: {error_detail}"
                except:
                    error_msg += f"Response: {response.text[:200]}"
                return error_msg, ""
                
        except Exception as e:
            return f"❌ **Error:** {str(e)}", ""

    def update_document(self, file, doc_path: str) -> Tuple[str, str]:
        """Update an existing document with a new file"""
        if file is None:
            return "❌ Please select a file to update with", ""
        
        if not doc_path or not doc_path.strip():
            return "❌ Please select a document from the dropdown to update", ""
        
        if doc_path == "No documents uploaded" or doc_path == "(No documents uploaded yet)":
            return "❌ No documents available to update. Please upload a document first.", ""
        
        if doc_path not in self.document_registry:
            available_docs = list(self.document_registry.keys())
            if available_docs:
                return f"❌ Document '{doc_path}' not found in registry.\n\nAvailable documents: {', '.join(available_docs)}", ""
            else:
                return f"❌ No documents in registry. Please upload a document first.", ""
        
        try:
            # Get existing document info
            existing_doc = self.document_registry[doc_path]
            
            # Create new document ID for the update
            new_doc_id = f"{doc_path}_{uuid.uuid4().hex[:8]}_updated"
            
            # Read new file content
            with open(file.name, 'rb') as f:
                file_content = f.read()
            
            # Prepare metadata
            metadata = {
                "doc_path": doc_path,
                "original_filename": Path(file.name).name,
                "doc_id": new_doc_id,
                "previous_doc_id": existing_doc["doc_id"],
                "operation": "update",
                "timestamp": datetime.now().isoformat(),
                "source": "comprehensive_ui",
                "file_size": len(file_content)
            }
            
            # Upload updated file via API
            files = {'file': (Path(file.name).name, file_content, 'application/octet-stream')}
            data = {'metadata': json.dumps(metadata)}
            
            response = requests.post(
                f"{self.api_url}/upload",
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Check if this was an update (old vectors deleted)
                is_update = result_data.get('is_update', False)
                old_vectors_deleted = result_data.get('old_vectors_deleted', 0)
                
                # Update registry
                self.document_registry[doc_path].update({
                    "doc_id": new_doc_id,
                    "original_filename": Path(file.name).name,
                    "file_path": file.name,
                    "chunks_created": result_data.get('chunks_created', 0),
                    "last_updated": datetime.now().isoformat(),
                    "status": "updated",
                    "file_size": len(file_content),
                    "content_preview": str(file_content[:200], 'utf-8', errors='ignore') + "...",
                    "old_vectors_deleted": old_vectors_deleted
                })
                
                result = f"✅ **Document Updated Successfully!**\n\n"
                result += f"📄 **Document Path:** `{doc_path}`\n"
                result += f"📁 **New File:** `{Path(file.name).name}`\n"
                result += f"🆔 **New Document ID:** `{new_doc_id}`\n"
                result += f"📝 **New Chunks Created:** {result_data.get('chunks_created', 0)}\n"
                result += f"📊 **New File Size:** {len(file_content)} bytes\n"
                result += f"🔄 **Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                if is_update and old_vectors_deleted > 0:
                    result += f"🗑️ **Old Vectors Deleted:** {old_vectors_deleted}\n"
                    result += f"✨ **Vector Replacement:** Complete\n"
                else:
                    result += f"⚠️ **Note:** No old vectors found to replace\n"
                
                result += f"\n**New Content Preview:** {str(file_content[:200], 'utf-8', errors='ignore')}..."
                
                # Update registry display
                registry_display = self._format_document_registry()
                
                return result, registry_display
            else:
                error_msg = f"❌ **Document Update Failed**\n\n"
                error_msg += f"HTTP Status: {response.status_code}\n"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"Details: {error_detail}"
                except:
                    error_msg += f"Response: {response.text[:200]}"
                return error_msg, ""
                
        except Exception as e:
            return f"❌ **Error:** {str(e)}", ""

    def delete_document(self, doc_path: str) -> Tuple[str, str]:
        """Delete a document from the system"""
        if not doc_path or not doc_path.strip():
            return "❌ Please select a document from the dropdown to delete", ""
        
        if doc_path == "No documents uploaded" or doc_path == "(No documents uploaded yet)":
            return "❌ No documents available to delete. Please upload a document first.", ""
        
        if doc_path not in self.document_registry:
            available_docs = list(self.document_registry.keys())
            if available_docs:
                return f"❌ Document '{doc_path}' not found in registry.\n\nAvailable documents: {', '.join(available_docs)}", ""
            else:
                return f"❌ No documents in registry. Please upload a document first.", ""
        
        try:
            doc_info = self.document_registry[doc_path]
            doc_id = doc_info["doc_id"]
            
            # Try to delete from backend via metadata filtering
            # Note: This is a simulated deletion since there's no direct delete endpoint
            # In a real system, you'd call a DELETE endpoint with the document ID
            
            # Mark as deleted in registry
            self.document_registry[doc_path]["status"] = "deleted"
            self.document_registry[doc_path]["deleted_at"] = datetime.now().isoformat()
            
            # Create a "deletion marker" document to override the original
            deletion_metadata = {
                "doc_path": doc_path,
                "doc_id": f"{doc_id}_DELETED",
                "operation": "delete",
                "timestamp": datetime.now().isoformat(),
                "source": "comprehensive_ui",
                "deletion_marker": True,
                "original_doc_id": doc_id
            }
            
            # Send deletion marker to system
            deletion_payload = {
                "text": f"[DELETED] Document {doc_path} was deleted on {datetime.now().isoformat()}",
                "metadata": deletion_metadata
            }
            
            try:
                response = requests.post(
                    f"{self.api_url}/ingest",
                    json=deletion_payload,
                    timeout=30
                )
                
                deletion_success = response.status_code == 200
            except:
                deletion_success = False
            
            result = f"✅ **Document Deletion Processed**\n\n"
            result += f"📄 **Document Path:** `{doc_path}`\n"
            result += f"📁 **Original File:** `{doc_info['original_filename']}`\n"
            result += f"🆔 **Document ID:** `{doc_id}`\n"
            result += f"🗑️ **Deleted:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            if deletion_success:
                result += f"✅ **Deletion marker added to system**\n"
                result += f"🔍 **Testing:** Query for this content should now show deletion marker\n\n"
            else:
                result += f"⚠️ **Registry deletion only** (backend deletion failed)\n"
                result += f"🔍 **Testing:** Document marked as deleted in registry but vectors may persist\n\n"
            
            result += f"**How to test deletion:**\n"
            result += f"1. Go to Query Testing tab\n"
            result += f"2. Search for content from this file\n"
            result += f"3. Check if document appears in results\n"
            result += f"4. Look for deletion status in Lifecycle Analysis\n\n"
            result += f"**Expected behavior:**\n"
            result += f"- Document should not appear in relevant search results\n"
            result += f"- If it appears, it should be marked as DELETED\n"
            result += f"- Registry shows 🗑️ deleted status"
            
            # Update registry display
            registry_display = self._format_document_registry()
            
            return result, registry_display
            
        except Exception as e:
            return f"❌ **Error:** {str(e)}", ""

    def get_document_paths(self) -> List[str]:
        """Get list of document paths for dropdown"""
        paths = list(self.document_registry.keys()) if self.document_registry else []
        print(f"DEBUG: Registry has {len(self.document_registry)} documents: {paths}")
        return paths

    def _format_document_registry(self) -> str:
        """Format the document registry for display"""
        if not self.document_registry:
            return "📋 **No documents in registry**"
        
        registry_text = f"📋 **Document Registry** ({len(self.document_registry)} documents)\n\n"
        
        for doc_path, info in self.document_registry.items():
            status_emoji = {
                "active": "✅",
                "updated": "🔄", 
                "deleted": "🗑️"
            }.get(info["status"], "❓")
            
            registry_text += f"{status_emoji} **{doc_path}**\n"
            registry_text += f"   📁 File: {info['original_filename']}\n"
            registry_text += f"   🆔 ID: `{info['doc_id']}`\n"
            registry_text += f"   📝 Chunks: {info['chunks_created']}\n"
            registry_text += f"   📊 Size: {info['file_size']} bytes\n"
            registry_text += f"   📅 Last Updated: {info['last_updated']}\n"
            registry_text += f"   📊 Status: {info['status'].upper()}\n"
            
            if info["status"] == "deleted" and "deleted_at" in info:
                registry_text += f"   🗑️ Deleted: {info['deleted_at']}\n"
            
            registry_text += f"   📖 Preview: {info['content_preview']}\n\n"
        
        return registry_text

    def test_query(self, query: str, max_results: int = 5) -> Tuple[str, str, str]:
        """Test query to see document lifecycle effects"""
        if not query.strip():
            return "Please enter a query to test.", "", ""
        
        try:
            payload = {
                "query": query,
                "max_results": max_results
            }
            
            response = requests.post(
                f"{self.api_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Format response
                answer = data.get('response', 'No response generated')
                
                # Format sources with lifecycle tracking
                sources = data.get('sources', [])
                sources_text = ""
                lifecycle_analysis = ""
                
                if sources:
                    sources_text = "📚 **Sources Found:**\n\n"
                    lifecycle_analysis = "🔍 **Document Lifecycle Analysis:**\n\n"
                    
                    for i, source in enumerate(sources, 1):
                        score = source.get('score', 0)
                        doc_id = source.get('doc_id', 'Unknown')
                        text_preview = source.get('text', '')[:150] + "..."
                        
                        # Check if this is a deletion marker
                        is_deletion_marker = "[DELETED]" in text_preview or source.get('metadata', {}).get('deletion_marker', False)
                        
                        # Check if this source matches any document in our registry
                        registry_match = None
                        for doc_path, info in self.document_registry.items():
                            if doc_id.startswith(doc_path) or info["doc_id"] == doc_id or doc_id.startswith(info["doc_id"]):
                                registry_match = (doc_path, info)
                                break
                        
                        sources_text += f"**Source {i}** (Score: {score:.3f})\n"
                        sources_text += f"Document ID: `{doc_id}`\n"
                        
                        if is_deletion_marker:
                            sources_text += f"🗑️ **DELETION MARKER** - This document was deleted\n"
                            sources_text += f"Preview: {text_preview}\n"
                        else:
                            sources_text += f"Preview: {text_preview}\n"
                        
                        if registry_match:
                            doc_path, info = registry_match
                            status_emoji = {
                                "active": "✅",
                                "updated": "🔄",
                                "deleted": "🗑️"
                            }.get(info["status"], "❓")
                            
                            sources_text += f"Registry Match: {status_emoji} `{doc_path}` ({info['status']})\n"
                            sources_text += f"Original File: `{info['original_filename']}`\n"
                            
                            lifecycle_analysis += f"**Source {i}:** {status_emoji} Document `{doc_path}`\n"
                            lifecycle_analysis += f"   File: {info['original_filename']}\n"
                            lifecycle_analysis += f"   Status: {info['status'].upper()}\n"
                            lifecycle_analysis += f"   Last Updated: {info['last_updated']}\n"
                            
                            if is_deletion_marker:
                                lifecycle_analysis += f"   🗑️ DELETION MARKER - This confirms the document was deleted\n"
                            elif info["status"] == "deleted":
                                lifecycle_analysis += f"   ⚠️ This document was marked as deleted but still appears in results\n"
                            elif info["status"] == "updated":
                                lifecycle_analysis += f"   ✅ This shows the updated file content\n"
                            else:
                                lifecycle_analysis += f"   ✅ This is the original uploaded file\n"
                            
                            lifecycle_analysis += "\n"
                        else:
                            if is_deletion_marker:
                                lifecycle_analysis += f"**Source {i}:** 🗑️ DELETION MARKER (document was deleted)\n\n"
                            else:
                                lifecycle_analysis += f"**Source {i}:** ❓ Not tracked in registry\n\n"
                        
                        sources_text += "\n"
                else:
                    sources_text = "❌ **No sources found for this query**"
                    lifecycle_analysis = "🔍 **No documents matched this query**"
                
                # Format metadata
                context_used = data.get('context_used', 0)
                metadata = f"**Query Results Metadata:**\n"
                metadata += f"- Query: `{query}`\n"
                metadata += f"- Context chunks used: {context_used}\n"
                metadata += f"- Max results requested: {max_results}\n"
                metadata += f"- Sources found: {len(sources)}\n"
                metadata += f"- Registry documents: {len(self.document_registry)}\n"
                metadata += f"- Query timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                
                return answer, sources_text, lifecycle_analysis
                
            else:
                error_msg = f"❌ **Query Failed:** HTTP {response.status_code}"
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    error_msg += f"\nDetails: {error_detail}"
                except:
                    error_msg += f"\nResponse: {response.text[:200]}"
                
                return error_msg, "", ""
                
        except Exception as e:
            return f"❌ **Query Error:** {str(e)}", "", ""

    def get_system_stats(self) -> str:
        """Get system statistics"""
        try:
            response = requests.get(f"{self.api_url}/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                stats_text = "## 📊 System Statistics\n\n"
                
                # FAISS Store Stats
                faiss_stats = data.get('faiss_store', {})
                stats_text += f"**🔍 FAISS Vector Store:**\n"
                stats_text += f"- Total Vectors: {faiss_stats.get('vector_count', 0):,}\n"
                stats_text += f"- Active Vectors: {faiss_stats.get('active_vectors', 0):,}\n"
                stats_text += f"- Deleted Vectors: {faiss_stats.get('deleted_vectors', 0):,}\n"
                stats_text += f"- Dimension: {faiss_stats.get('dimension', 0)}\n"
                stats_text += f"- Index Size: {faiss_stats.get('index_size_mb', 0):.2f} MB\n\n"
                
                # Metadata Store Stats
                metadata_stats = data.get('metadata_store', {})
                stats_text += f"**📋 Metadata Store:**\n"
                stats_text += f"- Total Files: {metadata_stats.get('total_files', 0):,}\n"
                stats_text += f"- Collections: {metadata_stats.get('collections', 0):,}\n\n"
                
                # Ingestion Engine Stats
                ingestion_stats = data.get('ingestion_engine', {})
                stats_text += f"**⚙️ Ingestion Engine:**\n"
                stats_text += f"- Files Processed: {ingestion_stats.get('files_processed', 0):,}\n"
                stats_text += f"- Chunks Created: {ingestion_stats.get('chunks_created', 0):,}\n"
                stats_text += f"- Embeddings Generated: {ingestion_stats.get('embeddings_generated', 0):,}\n\n"
                
                # Query Engine Stats
                query_stats = data.get('query_engine', {})
                stats_text += f"**🔍 Query Engine:**\n"
                stats_text += f"- Queries Processed: {query_stats.get('queries_processed', 0):,}\n"
                stats_text += f"- Average Response Time: {query_stats.get('avg_response_time', 0):.2f}s\n\n"
                
                stats_text += f"\n**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                return stats_text
            else:
                return f"❌ Error getting stats: HTTP {response.status_code}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    # Folder Monitoring Methods
    def get_folder_monitor_status(self) -> Tuple[str, str]:
        """Get folder monitoring status"""
        try:
            response = requests.get(f"{self.api_url}/folder-monitor/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    status_data = data.get('status', {})
                    
                    # Format status display
                    status_text = "## 📁 Folder Monitoring Status\n\n"
                    
                    is_running = status_data.get('is_running', False)
                    status_text += f"**🔄 Status:** {'🟢 Running' if is_running else '🔴 Stopped'}\n"
                    status_text += f"**📁 Monitored Folders:** {len(status_data.get('monitored_folders', []))}\n"
                    status_text += f"**📄 Files Tracked:** {status_data.get('total_files_tracked', 0)}\n"
                    status_text += f"**✅ Files Ingested:** {status_data.get('files_ingested', 0)}\n"
                    status_text += f"**❌ Files Failed:** {status_data.get('files_failed', 0)}\n"
                    status_text += f"**⏳ Files Pending:** {status_data.get('files_pending', 0)}\n"
                    status_text += f"**📊 Total Scans:** {status_data.get('scan_count', 0)}\n"
                    status_text += f"**⏱️ Check Interval:** {status_data.get('check_interval', 0)} seconds\n"
                    
                    last_scan = status_data.get('last_scan_time')
                    if last_scan:
                        status_text += f"**🕐 Last Scan:** {last_scan}\n"
                    else:
                        status_text += f"**🕐 Last Scan:** Never\n"
                    
                    status_text += f"**🔄 Auto-Ingest:** {'✅ Enabled' if status_data.get('auto_ingest', False) else '❌ Disabled'}\n"
                    
                    # Format folder list
                    folders = status_data.get('monitored_folders', [])
                    folder_list = ""
                    if folders:
                        folder_list = "## 📋 Monitored Folders\n\n"
                        for i, folder in enumerate(folders, 1):
                            folder_list += f"{i}. `{folder}`\n"
                    else:
                        folder_list = "## 📋 Monitored Folders\n\n❌ No folders are currently being monitored"
                    
                    return status_text, folder_list
                else:
                    error_msg = f"❌ Error: {data.get('error', 'Unknown error')}"
                    return error_msg, ""
            else:
                error_msg = f"❌ HTTP Error: {response.status_code}"
                return error_msg, ""
        except Exception as e:
            error_msg = f"❌ Connection Error: {str(e)}"
            return error_msg, ""

    def add_folder_to_monitoring(self, folder_path: str) -> str:
        """Add a folder to monitoring"""
        if not folder_path or not folder_path.strip():
            return "❌ Please enter a folder path"
        
        folder_path = folder_path.strip()
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            return f"❌ Folder does not exist: {folder_path}"
        
        if not os.path.isdir(folder_path):
            return f"❌ Path is not a directory: {folder_path}"
        
        try:
            response = requests.post(
                f"{self.api_url}/folder-monitor/add",
                json={"folder_path": folder_path},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = f"✅ **Folder Added Successfully!**\n\n"
                    result += f"📁 **Folder Path:** `{folder_path}`\n"
                    result += f"📄 **Files Found:** {data.get('files_found', 0)}\n"
                    result += f"📅 **Added At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    
                    # Check for immediate scan results
                    if data.get('immediate_scan'):
                        result += f"\n🔍 **Immediate Scan Results:**\n"
                        result += f"- Changes Detected: {data.get('changes_detected', 0)}\n"
                        result += f"- Files Tracked: {data.get('files_tracked', 0)}\n"
                    
                    result += f"\n💡 **Note:** Monitoring will automatically detect new files and changes in this folder."
                    
                    return result
                else:
                    return f"❌ Failed to add folder: {data.get('error', 'Unknown error')}"
            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    return f"❌ HTTP {response.status_code}: {error_detail}"
                except:
                    return f"❌ HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def start_folder_monitoring(self) -> str:
        """Start folder monitoring"""
        try:
            response = requests.post(f"{self.api_url}/folder-monitor/start", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = f"✅ **Folder Monitoring Started!**\n\n"
                    result += f"📁 **Folders Being Monitored:** {len(data.get('folders', []))}\n"
                    result += f"⏱️ **Check Interval:** {data.get('interval', 60)} seconds\n"
                    result += f"📅 **Started At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    
                    folders = data.get('folders', [])
                    if folders:
                        result += f"\n📋 **Monitored Folders:**\n"
                        for i, folder in enumerate(folders, 1):
                            result += f"{i}. `{folder}`\n"
                    
                    return result
                else:
                    return f"❌ Failed to start monitoring: {data.get('error', 'Unknown error')}"
            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    return f"❌ HTTP {response.status_code}: {error_detail}"
                except:
                    return f"❌ HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def stop_folder_monitoring(self) -> str:
        """Stop folder monitoring"""
        try:
            response = requests.post(f"{self.api_url}/folder-monitor/stop", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = f"🛑 **Folder Monitoring Stopped**\n\n"
                    result += f"📅 **Stopped At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    result += f"💡 **Note:** Files will no longer be automatically monitored for changes."
                    return result
                else:
                    return f"❌ Failed to stop monitoring: {data.get('error', 'Unknown error')}"
            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    return f"❌ HTTP {response.status_code}: {error_detail}"
                except:
                    return f"❌ HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def force_folder_scan(self) -> str:
        """Force an immediate scan of all monitored folders"""
        try:
            response = requests.post(f"{self.api_url}/folder-monitor/scan", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    result = f"🔍 **Folder Scan Completed!**\n\n"
                    result += f"📊 **Changes Detected:** {data.get('changes_detected', 0)}\n"
                    result += f"📄 **Files Tracked:** {data.get('files_tracked', 0)}\n"
                    result += f"📅 **Scan Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    
                    if data.get('changes_detected', 0) > 0:
                        result += f"\n✨ **New changes detected and will be processed automatically.**"
                    else:
                        result += f"\n💡 **No new changes detected in monitored folders.**"
                    
                    return result
                else:
                    return f"❌ Scan failed: {data.get('error', 'Unknown error')}"
            else:
                try:
                    error_detail = response.json().get('detail', 'Unknown error')
                    return f"❌ HTTP {response.status_code}: {error_detail}"
                except:
                    return f"❌ HTTP {response.status_code}: {response.text[:200]}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_monitored_files(self) -> str:
        """Get status of all monitored files"""
        try:
            response = requests.get(f"{self.api_url}/folder-monitor/files", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    files = data.get('files', {})
                    
                    if not files:
                        return "📄 No files are currently being tracked"
                    
                    result = f"## 📄 Monitored Files ({len(files)} total)\n\n"
                    
                    # Group files by status
                    status_groups = {}
                    for file_path, file_info in files.items():
                        status = file_info.get('ingestion_status', 'unknown')
                        if status not in status_groups:
                            status_groups[status] = []
                        status_groups[status].append((file_path, file_info))
                    
                    # Display by status
                    status_icons = {
                        'success': '✅',
                        'pending': '⏳',
                        'failed': '❌',
                        'unknown': '❓'
                    }
                    
                    for status, file_list in status_groups.items():
                        icon = status_icons.get(status, '❓')
                        result += f"### {icon} {status.title()} ({len(file_list)} files)\n\n"
                        
                        for file_path, file_info in file_list:
                            filename = os.path.basename(file_path)
                            result += f"- **{filename}**\n"
                            result += f"  - Path: `{file_path}`\n"
                            result += f"  - Size: {file_info.get('size', 0)} bytes\n"
                            
                            if file_info.get('error_message'):
                                result += f"  - Error: {file_info.get('error_message')}\n"
                            
                            result += f"\n"
                    
                    return result
                else:
                    return f"❌ Error: {data.get('error', 'Unknown error')}"
            else:
                return f"❌ HTTP Error: {response.status_code}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

def create_comprehensive_interface():
    """Create the comprehensive document lifecycle management interface"""
    
    print("DEBUG: Creating ComprehensiveRAGUI instance")
    ui = ComprehensiveRAGUI()
    print("DEBUG: ComprehensiveRAGUI instance created")
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .lifecycle-section {
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    """
    
    with gr.Blocks(css=css, title="RAG System - Document Lifecycle Management") as interface:
        
        gr.Markdown("""
        # 📁 RAG System - Complete Document Lifecycle Management
        
        **Upload Files → Update Files → Delete Documents → Test Queries**
        
        This interface provides complete document lifecycle management with file uploads 
        and real-time query testing to demonstrate how document changes affect search results.
        """)
        
        # Connection Status
        with gr.Row():
            connection_status = gr.Markdown(
                value="🔍 Checking API connection...",
                label="API Connection Status"
            )
            refresh_connection_btn = gr.Button("🔄 Refresh Connection", size="sm")
        
        with gr.Tabs():
            
            # Document Lifecycle Management Tab
            with gr.Tab("📁 Document Lifecycle"):
                gr.Markdown("### Complete Document Management: Upload → Update → Delete")
                
                with gr.Row():
                    # Left Column: Document Operations
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Document Operations")
                        
                        # Upload Section
                        gr.Markdown("##### ➕ Upload New Document")
                        upload_file_input = gr.File(
                            label="📁 Select File to Upload",
                            file_types=[".txt", ".pdf", ".docx", ".md", ".json", ".csv"],
                            type="filepath"
                        )
                        
                        upload_doc_path_input = gr.Textbox(
                            label="📄 Document Path (Optional)",
                            placeholder="e.g., /docs/my-document (auto-generated if empty)",
                            info="Unique path identifier for your document"
                        )
                        
                        upload_doc_btn = gr.Button("➕ Upload Document", variant="primary")
                        
                        gr.Markdown("---")
                        
                        # Update Section
                        gr.Markdown("##### 🔄 Update Existing Document")
                        update_file_input = gr.File(
                            label="📁 Select New File for Update",
                            file_types=[".txt", ".pdf", ".docx", ".md", ".json", ".csv"],
                            type="filepath"
                        )
                        
                        update_doc_path_input = gr.Dropdown(
                            label="📄 Select Document to Update",
                            choices=["(No documents uploaded yet)"],
                            allow_custom_value=True,
                            info="Choose from uploaded documents"
                        )
                        
                        update_doc_btn = gr.Button("🔄 Update Document", variant="secondary")
                        
                        gr.Markdown("---")
                        
                        # Delete Section
                        gr.Markdown("##### 🗑️ Delete Document")
                        delete_doc_path_input = gr.Dropdown(
                            label="📄 Select Document to Delete",
                            choices=["(No documents uploaded yet)"],
                            allow_custom_value=True,
                            info="Choose from uploaded documents"
                        )
                        
                        delete_doc_btn = gr.Button("🗑️ Delete Document", variant="stop")
                        
                        gr.Markdown("---")
                        
                        # Manual refresh button for dropdowns
                        refresh_dropdowns_btn = gr.Button("🔄 Refresh Document Lists", variant="secondary", size="sm")
                        
                        operation_result = gr.Markdown(
                            label="Operation Result",
                            value="Ready for document operations..."
                        )
                    
                    # Right Column: Document Registry
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📋 Document Registry")
                        
                        document_registry_display = gr.Markdown(
                            label="Active Documents",
                            value="📋 No documents in registry",
                            height=600
                        )
                        
                        refresh_registry_btn = gr.Button("🔄 Refresh Registry")
                        
                        # Helper function to update dropdowns
                        def update_dropdowns():
                            print("DEBUG: update_dropdowns() called")
                            paths = ui.get_document_paths()
                            print(f"DEBUG: update_dropdowns got paths: {paths}")
                            if not paths:
                                # If no documents, show empty choices with helpful label
                                print("DEBUG: No paths found, returning placeholder")
                                return (
                                    gr.update(choices=["(No documents uploaded yet)"], value=None),
                                    gr.update(choices=["(No documents uploaded yet)"], value=None)
                                )
                            else:
                                # If documents exist, show them
                                print(f"DEBUG: Paths found, updating dropdowns with: {paths}")
                                return (
                                    gr.update(choices=paths, value=None),
                                    gr.update(choices=paths, value=None)
                                )
            
            # Query Testing Tab
            with gr.Tab("🔍 Query Testing"):
                gr.Markdown("### Test Queries to See Document Lifecycle Effects")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        test_query_input = gr.Textbox(
                            label="🔍 Test Query",
                            placeholder="Enter a query to test document lifecycle effects...",
                            lines=2
                        )
                        
                        max_results_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Maximum Results"
                        )
                        
                        test_query_btn = gr.Button("🔍 Test Query", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("#### 💡 Query Testing Tips")
                        gr.Markdown("""
                        **Test the complete lifecycle:**
                        1. Upload a document file
                        2. Query for content → should appear
                        3. Update with a different file  
                        4. Query again → should show updated content
                        5. Delete the document
                        6. Query again → should show deletion marker or not appear
                        
                        **Testing deletion specifically:**
                        - After deleting, search for specific content from that file
                        - Look for 🗑️ DELETION MARKER in results
                        - Check Lifecycle Analysis for deletion status
                        - Deleted documents should not contribute to answers
                        
                        **Supported file types:**
                        - 📄 Text files (.txt, .md)
                        - 📄 PDF documents (.pdf)
                        - 📄 Word documents (.docx)
                        - 📄 JSON files (.json)
                        - 📄 CSV files (.csv)
                        
                        **Example workflow:**
                        - Upload: document about "AI basics"
                        - Query: "What is artificial intelligence?"
                        - Update: upload new file about "Advanced AI"
                        - Query again → see updated response
                        - Delete document
                        - Query again → see effect
                        """)
                
                # Query Results
                with gr.Row():
                    with gr.Column():
                        query_answer = gr.Textbox(
                            label="🤖 AI Response",
                            lines=6,
                            interactive=False
                        )
                    
                    with gr.Column():
                        query_sources = gr.Markdown(
                            label="📚 Sources & Citations"
                        )
                
                query_lifecycle_analysis = gr.Markdown(
                    label="🔍 Document Lifecycle Analysis"
                )
            
            # System Overview Tab
            with gr.Tab("📊 System Overview"):
                gr.Markdown("### System Statistics and Health")
                
                with gr.Row():
                    with gr.Column():
                        system_stats_display = gr.Markdown(
                            label="System Statistics",
                            value="Click 'Get Statistics' to load system info..."
                        )
                        
                        get_stats_btn = gr.Button("📊 Get Statistics", variant="secondary")
                    
                    with gr.Column():
                        gr.Markdown("#### 🎯 System Health")
                        gr.Markdown("""
                        **Backend Integration:**
                        - ✅ Full API synchronization
                        - ✅ File upload support
                        - ✅ Real-time document tracking
                        - ✅ Lifecycle state management
                        - ✅ Query result analysis
                        
                        **Features:**
                        - 📁 File upload & ingestion
                        - 🔄 Document updates
                        - 🗑️ Document deletion tracking
                        - 🔍 Real-time query testing
                        - 📊 Comprehensive statistics
                        
                        **API Endpoints Used:**
                        - `POST /upload` - File operations
                        - `POST /query` - Query testing
                        - `GET /stats` - System statistics
                        - `GET /health` - Health monitoring
                        
                        **Supported File Types:**
                        - Text: .txt, .md
                        - Documents: .pdf, .docx
                        - Data: .json, .csv
                        """)
            
            # Help & Guide Tab
            with gr.Tab("❓ Help & Guide"):
                gr.Markdown("""
                # 📖 Complete Document Lifecycle Guide
                
                ## 🎯 Overview
                This interface demonstrates the complete document lifecycle in your RAG system:
                **Upload Files → Update Files → Delete Documents → Query Testing**
                
                ## 📁 Document Lifecycle Workflow
                
                ### 1. ➕ **Upload Document**
                - Click **"Select File to Upload"** and choose your document
                - Optionally provide a **Document Path** (auto-generated if empty)
                - Click **"Upload Document"**
                - ✅ Document is processed and becomes searchable
                
                ### 2. 🔄 **Update Document**
                - Select a **new file** to replace the existing content
                - Choose the **document to update** from the dropdown
                - Click **"Update Document"**
                - ✅ New version is created, old content is superseded
                
                ### 3. 🗑️ **Delete Document**
                - Select the **document to delete** from the dropdown
                - Click **"Delete Document"**
                - ✅ Document is marked as deleted in the registry
                
                ### 4. 🔍 **Test Queries**
                - Go to the **"Query Testing"** tab
                - Enter queries related to your document content
                - See how **Upload/Update/Delete** operations affect search results
                - View **Document Lifecycle Analysis** to understand what's happening
                
                ## 📋 Document Registry
                The registry tracks all your documents and their states:
                - ✅ **Active**: Original uploaded documents
                - 🔄 **Updated**: Documents that have been replaced with new files
                - 🗑️ **Deleted**: Documents marked for deletion
                
                ## 📁 Supported File Types
                
                ### Text Files
                - **.txt** - Plain text files
                - **.md** - Markdown files
                
                ### Document Files
                - **.pdf** - PDF documents
                - **.docx** - Microsoft Word documents
                
                ### Data Files
                - **.json** - JSON data files
                - **.csv** - Comma-separated values
                
                ## 🔍 Query Testing Features
                - **Real-time Results**: See immediate effects of document changes
                - **Source Tracking**: Identify which files contributed to answers
                - **Lifecycle Analysis**: Understand document states in search results
                - **File Information**: See original filenames and file sizes
                - **Score Analysis**: See relevance scores for each source
                
                ## 💡 Example Workflow
                
                ### Step 1: Upload Initial Document
                ```
                File: ai-basics.txt (containing "AI is machine intelligence")
                Path: /docs/ai-guide (auto-generated if empty)
                ```
                
                ### Step 2: Test Query
                ```
                Query: "What is artificial intelligence?"
                Expected: Should return information from ai-basics.txt
                ```
                
                ### Step 3: Update Document
                ```
                New File: advanced-ai.txt (containing "AI includes neural networks and deep learning")
                Document: /docs/ai-guide (same path)
                ```
                
                ### Step 4: Test Same Query
                ```
                Query: "What is artificial intelligence?"
                Expected: Should return updated information about neural networks
                ```
                
                ### Step 5: Delete Document
                ```
                Document: /docs/ai-guide
                Action: Delete
                ```
                
                ### Step 6: Test Query Again
                ```
                Query: "What is artificial intelligence?"
                Expected: Should not return the deleted document content
                ```
                
                ## 🔧 Technical Details
                
                ### File Processing
                - **Upload Endpoint**: `POST /upload` for all file operations
                - **Automatic Processing**: Files are automatically parsed and chunked
                - **Metadata Tracking**: Each operation includes lifecycle metadata
                - **Unique IDs**: Each document version gets a unique identifier
                - **Real-time Sync**: Changes are immediately reflected in queries
                
                ### Document Versioning
                - **Upload**: Creates new document with unique ID
                - **Update**: Creates new version, supersedes previous
                - **Delete**: Marks document as deleted in registry
                
                ### Query Analysis
                - **Source Matching**: Links query results to registry documents
                - **File Tracking**: Shows original filenames and sizes
                - **Status Tracking**: Shows document lifecycle state
                - **Relevance Scoring**: Displays similarity scores
                
                ## 🚀 Best Practices
                
                1. **Use Descriptive Paths**: Choose meaningful document paths
                2. **Test Immediately**: Query after each operation to see effects
                3. **Monitor Registry**: Keep track of document states and file sizes
                4. **Use Different Content**: Make files easily distinguishable for testing
                5. **Check Lifecycle Analysis**: Understand what's happening behind the scenes
                6. **Organize Files**: Use consistent naming and path conventions
                
                ## 🔍 Troubleshooting
                
                ### File Upload Issues
                - Check file type is supported (.txt, .pdf, .docx, .md, .json, .csv)
                - Ensure file is not corrupted or empty
                - Try smaller files if upload times out
                
                ### Document Not Found in Query
                - Check if document was successfully uploaded (see Operation Result)
                - Verify file content is relevant to your query
                - Try broader search terms
                - Check file was properly processed (see chunks created)
                
                ### Update Not Reflected
                - Ensure you selected the correct document path from dropdown
                - Check that new file content is significantly different
                - Wait a moment and try querying again
                - Verify new file was uploaded successfully
                
                ### Delete Not Working
                - Deletion marks documents in registry but vectors may persist
                - Check Document Registry to confirm deletion status
                - Deleted documents may still appear in results but marked as deleted
                
                ## 📊 System Requirements
                - ✅ Backend server running on `http://localhost:8000`
                - ✅ File upload endpoint functional (`POST /upload`)
                - ✅ Query endpoint operational (`POST /query`)
                - ✅ Vector database operational
                - ✅ LLM service available
                
                ## 🎯 Advanced Features
                
                ### Automatic Path Generation
                - If no document path is provided, one is auto-generated from filename
                - Format: `/docs/{filename_without_extension}`
                
                ### File Size Tracking
                - Registry tracks file sizes for monitoring
                - System statistics show total storage usage
                
                ### Content Preview
                - Registry shows preview of file content
                - Helps identify documents quickly
                
                ### Dropdown Updates
                - Document dropdowns automatically update when new documents are added
                - Makes it easy to select documents for update/delete operations
                
                ---
                
                **🎯 This interface provides complete visibility into your RAG system's document lifecycle with file upload support!**
                """)
            
            # Folder Monitoring Tab
            with gr.Tab("📁 Folder Monitoring"):
                gr.Markdown("### Automatic Folder Monitoring & File Ingestion")
                
                with gr.Row():
                    # Left Column: Folder Operations
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📁 Folder Management")
                        
                        # Add Folder Section
                        gr.Markdown("##### ➕ Add Folder to Monitor")
                        folder_path_input = gr.Textbox(
                            label="📁 Folder Path",
                            placeholder="e.g., C:\\Documents\\MyFiles or /home/user/documents",
                            info="Enter the full path to the folder you want to monitor"
                        )
                        
                        add_folder_btn = gr.Button("➕ Add Folder", variant="primary")
                        
                        gr.Markdown("---")
                        
                        # Control Section
                        gr.Markdown("##### 🔄 Monitoring Controls")
                        
                        with gr.Row():
                            start_monitoring_btn = gr.Button("▶️ Start Monitoring", variant="secondary")
                            stop_monitoring_btn = gr.Button("⏸️ Stop Monitoring", variant="secondary")
                        
                        force_scan_btn = gr.Button("🔍 Force Scan Now", variant="secondary")
                        
                        gr.Markdown("---")
                        
                        # Status Refresh
                        refresh_status_btn = gr.Button("🔄 Refresh Status", variant="secondary", size="sm")
                        
                        folder_operation_result = gr.Markdown(
                            label="Operation Result",
                            value="Ready for folder monitoring operations..."
                        )
                    
                    # Right Column: Status & Information
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📊 Monitoring Status")
                        
                        folder_status_display = gr.Markdown(
                            label="Folder Monitor Status",
                            value="🔍 Loading status...",
                            height=300
                        )
                        
                        folder_list_display = gr.Markdown(
                            label="Monitored Folders",
                            value="📋 Loading folder list...",
                            height=200
                        )
                
                with gr.Row():
                    # File Status Section
                    with gr.Column():
                        gr.Markdown("#### 📄 Monitored Files Status")
                        
                        refresh_files_btn = gr.Button("🔄 Refresh File Status", variant="secondary", size="sm")
                        
                        monitored_files_display = gr.Markdown(
                            label="File Status",
                            value="📄 Loading file status...",
                            height=400
                        )
                
                # Information Section
                gr.Markdown("""
                ### 💡 How Folder Monitoring Works
                
                **Automatic File Detection:**
                - Monitors specified folders for new files and changes
                - Automatically ingests supported file types
                - Tracks file status and ingestion results
                
                **Supported File Types:**
                - 📄 Text files: .txt, .md
                - 📄 Documents: .pdf, .docx  
                - 📄 Data files: .json, .csv
                
                **Monitoring Features:**
                - ⏱️ Automatic scanning every 15 seconds
                - 🔍 Manual scan on demand
                - 📊 File status tracking
                - ✅ Ingestion success/failure monitoring
                - 🔄 Real-time status updates
                
                **Usage Tips:**
                1. Add folders containing documents you want to monitor
                2. Start monitoring to enable automatic scanning
                3. Files will be automatically detected and ingested
                4. Use "Force Scan" to immediately check for changes
                5. Monitor file status to see ingestion results
                """)
        
        # Event Handlers
        
        # Connection status
        refresh_connection_btn.click(
            fn=ui.check_api_connection,
            outputs=[connection_status]
        )
        
        # Document operations
        def upload_and_update(file, doc_path):
            print(f"DEBUG: upload_and_update called with file: {file}, doc_path: {doc_path}")
            result, registry = ui.upload_document(file, doc_path)
            print(f"DEBUG: Upload result: {result[:100]}...")
            print(f"DEBUG: Registry after upload: {registry[:100]}...")
            return result, registry
        
        upload_doc_btn.click(
            fn=upload_and_update,
            inputs=[upload_file_input, upload_doc_path_input],
            outputs=[operation_result, document_registry_display]
        ).then(
            fn=update_dropdowns,
            outputs=[update_doc_path_input, delete_doc_path_input]
        )
        
        update_doc_btn.click(
            fn=ui.update_document,
            inputs=[update_file_input, update_doc_path_input],
            outputs=[operation_result, document_registry_display]
        ).then(
            fn=update_dropdowns,
            outputs=[update_doc_path_input, delete_doc_path_input]
        )
        
        delete_doc_btn.click(
            fn=ui.delete_document,
            inputs=[delete_doc_path_input],
            outputs=[operation_result, document_registry_display]
        ).then(
            fn=update_dropdowns,
            outputs=[update_doc_path_input, delete_doc_path_input]
        )
        
        refresh_registry_btn.click(
            fn=ui._format_document_registry,
            outputs=[document_registry_display]
        ).then(
            fn=update_dropdowns,
            outputs=[update_doc_path_input, delete_doc_path_input]
        )
        
        # Manual dropdown refresh
        refresh_dropdowns_btn.click(
            fn=update_dropdowns,
            outputs=[update_doc_path_input, delete_doc_path_input]
        )
        
        # Query testing
        test_query_btn.click(
            fn=ui.test_query,
            inputs=[test_query_input, max_results_slider],
            outputs=[query_answer, query_sources, query_lifecycle_analysis]
        )
        
        # System stats
        get_stats_btn.click(
            fn=ui.get_system_stats,
            outputs=[system_stats_display]
        )
        
        # Folder monitoring event handlers
        add_folder_btn.click(
            fn=ui.add_folder_to_monitoring,
            inputs=[folder_path_input],
            outputs=[folder_operation_result]
        ).then(
            fn=ui.get_folder_monitor_status,
            outputs=[folder_status_display, folder_list_display]
        )
        
        start_monitoring_btn.click(
            fn=ui.start_folder_monitoring,
            outputs=[folder_operation_result]
        ).then(
            fn=ui.get_folder_monitor_status,
            outputs=[folder_status_display, folder_list_display]
        )
        
        stop_monitoring_btn.click(
            fn=ui.stop_folder_monitoring,
            outputs=[folder_operation_result]
        ).then(
            fn=ui.get_folder_monitor_status,
            outputs=[folder_status_display, folder_list_display]
        )
        
        force_scan_btn.click(
            fn=ui.force_folder_scan,
            outputs=[folder_operation_result]
        ).then(
            fn=ui.get_folder_monitor_status,
            outputs=[folder_status_display, folder_list_display]
        )
        
        refresh_status_btn.click(
            fn=ui.get_folder_monitor_status,
            outputs=[folder_status_display, folder_list_display]
        )
        
        refresh_files_btn.click(
            fn=ui.get_monitored_files,
            outputs=[monitored_files_display]
        )
        
        # Initialize connection status and folder monitoring status on load
        interface.load(
            fn=ui.check_api_connection,
            outputs=[connection_status]
        ).then(
            fn=ui.get_folder_monitor_status,
            outputs=[folder_status_display, folder_list_display]
        ).then(
            fn=ui.get_monitored_files,
            outputs=[monitored_files_display]
        )
    
    return interface

def check_server_status(api_url: str = "http://localhost:8000", max_retries: int = 3) -> bool:
    """Check if the RAG system server is running"""
    print(f"🔍 Checking server status at {api_url}...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Server is running and healthy!")
                return True
            else:
                print(f"⚠️ Server responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries}: Server not responding - {e}")
            if attempt < max_retries - 1:
                print("⏳ Waiting 2 seconds before retry...")
                time.sleep(2)
    
    return False

def main():
    """Main function to launch the comprehensive UI"""
    print("📁 RAG System - Document Lifecycle Management UI")
    print("=" * 60)
    print("DEBUG: Starting main() function")
    
    # Check if server is running
    print("DEBUG: Checking server status...")
    if not check_server_status():
        print("\n❌ RAG System server is not running!")
        print("📋 To start the server, run:")
        print("   python main.py")
        print("\n🔄 Or if you're in the rag-system directory:")
        print("   cd rag-system && python main.py")
        return False
    
    print("DEBUG: Server is running, proceeding with UI creation...")
    
    try:
        print("\n🎛️ Creating comprehensive interface...")
        print("DEBUG: About to call create_comprehensive_interface()")
        interface = create_comprehensive_interface()
        print("DEBUG: Interface created successfully")
        
        print("\n🌟 DOCUMENT LIFECYCLE MANAGEMENT UI")
        print("=" * 50)
        print("🌐 API Server: http://localhost:8000")
        print("🎛️ Document UI: http://localhost:7866")
        print("\n📁 Complete File-Based Lifecycle Features:")
        print("  ➕ Upload Document Files")
        print("  🔄 Update with New Files") 
        print("  🗑️ Delete Documents")
        print("  🔍 Real-time Query Testing")
        print("  📋 Document Registry Tracking")
        print("  📊 File-based Lifecycle Analysis")
        print("\n📄 Supported File Types:")
        print("  • Text: .txt, .md")
        print("  • Documents: .pdf, .docx")
        print("  • Data: .json, .csv")
        print("\n🎯 Test the complete Upload → Update → Delete → Query workflow!")
        print("   Upload actual files and see real-time effects in search results.")
        print("\nReady to launch! Press Ctrl+C to stop the UI")
        print("=" * 50)
        
        # Launch the interface
        print("DEBUG: About to launch interface on port 7866")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7866,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Error launching comprehensive UI: {e}")
        return False

if __name__ == "__main__":
    main() 