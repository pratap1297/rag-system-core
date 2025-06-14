"""
ServiceNow API Routes
REST API endpoints for ServiceNow integration management
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from ...core.dependency_container import get_dependency_container
from ...integrations.servicenow import ServiceNowIntegration
from ...core.error_handling import IntegrationError

# Create router
router = APIRouter(prefix="/servicenow", tags=["ServiceNow Integration"])
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ServiceNowConfig(BaseModel):
    """ServiceNow configuration model"""
    enabled: bool = False
    fetch_interval_minutes: int = Field(default=15, ge=1, le=1440)
    batch_size: int = Field(default=100, ge=1, le=1000)
    max_incidents_per_fetch: int = Field(default=1000, ge=1, le=10000)
    priority_filter: List[str] = Field(default=["1", "2", "3"])
    state_filter: List[str] = Field(default=["1", "2", "3"])
    days_back: int = Field(default=7, ge=1, le=365)
    network_only: bool = False
    auto_ingest: bool = True
    cache_enabled: bool = True
    cache_ttl_hours: int = Field(default=1, ge=1, le=168)

class SyncFilters(BaseModel):
    """Filters for manual sync"""
    priority_filter: Optional[List[str]] = None
    state_filter: Optional[List[str]] = None
    days_back: Optional[int] = None
    network_only: Optional[bool] = None
    limit: Optional[int] = Field(default=100, ge=1, le=1000)

class ServiceNowStatus(BaseModel):
    """ServiceNow integration status"""
    initialized: bool
    connection_healthy: bool
    scheduler_running: bool
    last_sync_time: Optional[str]
    statistics: Dict[str, Any]
    connection_info: Optional[Dict[str, Any]]

class SyncResult(BaseModel):
    """Sync operation result"""
    incidents_fetched: int
    incidents_processed: int
    incidents_ingested: int
    new_incidents: int
    updated_incidents: int
    duration: float

# Dependency to get ServiceNow integration
def get_servicenow_integration():
    """Get ServiceNow integration instance"""
    try:
        container = get_dependency_container()
        config_manager = container.get('config_manager')
        ingestion_engine = container.get('ingestion_engine')
        
        # Create or get cached integration instance
        if not hasattr(get_servicenow_integration, '_integration'):
            get_servicenow_integration._integration = ServiceNowIntegration(
                config_manager=config_manager,
                ingestion_engine=ingestion_engine
            )
        
        return get_servicenow_integration._integration
    except Exception as e:
        logger.error(f"Error getting ServiceNow integration: {e}")
        raise HTTPException(status_code=500, detail="ServiceNow integration not available")

@router.get("/status", response_model=ServiceNowStatus)
async def get_servicenow_status(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Get ServiceNow integration status"""
    try:
        status = integration.get_integration_status()
        return ServiceNowStatus(**status)
    except Exception as e:
        logger.error(f"Error getting ServiceNow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def initialize_servicenow(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Initialize ServiceNow integration"""
    try:
        success = integration.initialize()
        return {
            "status": "success" if success else "failed",
            "message": "ServiceNow integration initialized successfully" if success else "Initialization failed"
        }
    except IntegrationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error initializing ServiceNow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_automated_sync(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Start automated ServiceNow synchronization"""
    try:
        success = integration.start_automated_sync()
        return {
            "status": "success" if success else "failed",
            "message": "Automated sync started successfully" if success else "Failed to start automated sync"
        }
    except IntegrationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting ServiceNow sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_automated_sync(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Stop automated ServiceNow synchronization"""
    try:
        integration.stop_automated_sync()
        return {
            "status": "success",
            "message": "Automated sync stopped successfully"
        }
    except Exception as e:
        logger.error(f"Error stopping ServiceNow sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync", response_model=SyncResult)
async def manual_sync(
    filters: Optional[SyncFilters] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    integration: ServiceNowIntegration = Depends(get_servicenow_integration)
):
    """Perform manual ServiceNow synchronization"""
    try:
        filter_dict = filters.dict(exclude_none=True) if filters else None
        result = integration.manual_sync(filters=filter_dict)
        return SyncResult(**result)
    except IntegrationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in manual sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync/incident/{incident_number}")
async def sync_specific_incident(
    incident_number: str,
    integration: ServiceNowIntegration = Depends(get_servicenow_integration)
):
    """Sync a specific incident by number"""
    try:
        result = integration.sync_specific_incident(incident_number)
        
        if result['status'] == 'not_found':
            raise HTTPException(status_code=404, detail=result['message'])
        elif result['status'] == 'error':
            raise HTTPException(status_code=500, detail=result['message'])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing specific incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tickets/recent")
async def get_recent_tickets(
    limit: int = 10,
    integration: ServiceNowIntegration = Depends(get_servicenow_integration)
):
    """Get recently processed ServiceNow tickets"""
    try:
        tickets = integration.get_recent_tickets(limit=limit)
        return {
            "tickets": tickets,
            "count": len(tickets)
        }
    except Exception as e:
        logger.error(f"Error getting recent tickets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_sync_history(
    limit: int = 10,
    integration: ServiceNowIntegration = Depends(get_servicenow_integration)
):
    """Get ServiceNow synchronization history"""
    try:
        history = integration.get_sync_history(limit=limit)
        return {
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting sync history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config")
async def update_servicenow_config(
    config: ServiceNowConfig,
    integration: ServiceNowIntegration = Depends(get_servicenow_integration)
):
    """Update ServiceNow integration configuration"""
    try:
        config_dict = config.dict()
        success = integration.update_configuration(config_dict)
        
        return {
            "status": "success" if success else "failed",
            "message": "Configuration updated successfully" if success else "Failed to update configuration",
            "config": config_dict
        }
    except Exception as e:
        logger.error(f"Error updating ServiceNow config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_servicenow_config(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Get current ServiceNow integration configuration"""
    try:
        status = integration.get_integration_status()
        config = status.get('statistics', {}).get('config', {})
        return {"config": config}
    except Exception as e:
        logger.error(f"Error getting ServiceNow config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test")
async def test_servicenow_integration(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Test ServiceNow integration functionality"""
    try:
        test_results = integration.test_integration()
        
        # Determine HTTP status based on test results
        if test_results['overall_status'] == 'failed':
            status_code = 500 if not test_results['connection_test'] else 400
        else:
            status_code = 200
        
        return test_results
    except Exception as e:
        logger.error(f"Error testing ServiceNow integration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/connection/info")
async def get_connection_info(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Get ServiceNow connection information"""
    try:
        if not integration.is_initialized:
            raise HTTPException(status_code=400, detail="ServiceNow integration not initialized")
        
        connection_info = integration.connector.get_connection_info()
        return connection_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting connection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connection/test")
async def test_connection(integration: ServiceNowIntegration = Depends(get_servicenow_integration)):
    """Test ServiceNow connection"""
    try:
        if not integration.is_initialized:
            integration.initialize()
        
        connected = integration.connector.test_connection()
        return {
            "connected": connected,
            "message": "Connection successful" if connected else "Connection failed"
        }
    except Exception as e:
        logger.error(f"Error testing connection: {e}")
        return {
            "connected": False,
            "message": f"Connection test failed: {str(e)}"
        } 