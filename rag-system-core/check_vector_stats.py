#!/usr/bin/env python3
"""
Check Vector Statistics
Check the vector store statistics to understand active vs deleted vectors
"""

import requests
import json

def check_vector_stats():
    """Check vector statistics"""
    
    print("📊 Checking Vector Statistics")
    print("=" * 40)
    
    api_url = "http://localhost:8000"
    
    try:
        # Get general stats
        print("\n1️⃣ General Stats:")
        response = requests.get(f"{api_url}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   📊 Total vectors: {stats.get('total_vectors', 0)}")
            print(f"   📄 Total documents: {stats.get('total_documents', 0)}")
            print(f"   📝 Total chunks: {stats.get('total_chunks', 0)}")
        else:
            print(f"   ❌ Error getting stats: {response.status_code}")
        
        # Get detailed FAISS stats
        print("\n2️⃣ FAISS Store Details:")
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            components = health.get('components', {})
            faiss_store = components.get('faiss_store', {})
            
            if faiss_store.get('status') == 'healthy':
                details = faiss_store.get('details', {})
                print(f"   📊 Vector count (physical): {details.get('vector_count', 0)}")
                print(f"   ✅ Active vectors: {details.get('active_vectors', 0)}")
                print(f"   🗑️ Deleted vectors: {details.get('deleted_vectors', 0)}")
                print(f"   📏 Dimension: {details.get('dimension', 0)}")
                print(f"   💾 Index size: {details.get('index_size_mb', 0):.2f} MB")
                print(f"   🔢 Next ID: {details.get('next_id', 0)}")
                
                # Calculate deletion percentage
                total_metadata = details.get('metadata_count', 0)
                deleted = details.get('deleted_vectors', 0)
                if total_metadata > 0:
                    deletion_rate = (deleted / total_metadata) * 100
                    print(f"   📈 Deletion rate: {deletion_rate:.1f}%")
            else:
                print(f"   ❌ FAISS store status: {faiss_store.get('status', 'unknown')}")
        else:
            print(f"   ❌ Error getting health: {response.status_code}")
        
        # Test search to verify deleted content is excluded
        print("\n3️⃣ Testing Search Exclusion:")
        search_payload = {
            "query": "test deletion document",
            "top_k": 10
        }
        response = requests.post(f"{api_url}/query", json=search_payload, timeout=10)
        if response.status_code == 200:
            results = response.json()
            result_count = len(results.get('results', []))
            print(f"   🔍 Search results: {result_count}")
            
            # Check if any results contain deleted content
            found_deleted_content = False
            for result in results.get('results', []):
                content = result.get('content', '').lower()
                if 'test deletion' in content:
                    found_deleted_content = True
                    print(f"   ⚠️ Found deleted content: {content[:100]}...")
            
            if not found_deleted_content:
                print(f"   ✅ No deleted content found in search results")
        else:
            print(f"   ❌ Search test failed: {response.status_code}")
        
        # Check folder monitoring status
        print("\n4️⃣ Folder Monitoring Status:")
        response = requests.get(f"{api_url}/folder-monitor/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                status = data.get('status', {})
                print(f"   📁 Files tracked: {status.get('total_files_tracked', 0)}")
                print(f"   ✅ Files ingested: {status.get('files_ingested', 0)}")
                print(f"   ❌ Files failed: {status.get('files_failed', 0)}")
                print(f"   ⏳ Files pending: {status.get('files_pending', 0)}")
            else:
                print(f"   ❌ Error: {data.get('error')}")
        else:
            print(f"   ❌ Error getting folder monitor status: {response.status_code}")
        
        print("\n📋 Summary:")
        print("   The system uses 'soft deletion' for vectors:")
        print("   • Deleted vectors are marked as deleted in metadata")
        print("   • Physical vectors remain in the FAISS index")
        print("   • Search automatically excludes deleted vectors")
        print("   • This is efficient and functionally correct")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_vector_stats() 