from fastapi import APIRouter, Depends, HTTPException
from typing import Dict
from app.agents.multi_agent_orchestrator import MultiAgentOrchestrator
from app.models.manager import ModelManager
from app.cache.redis_client import CacheManager
from app.schemas.requests import ResearchRequest
from app.schemas.responses import ResearchResponse, ResponseMetadata
from app.api.security import get_current_user, User
from app.dependencies import get_model_manager, get_cache_manager
import structlog
import uuid

router = APIRouter()
logger = structlog.get_logger("research_api")

@router.post("/deep-dive")
async def research_deep_dive(
    request: ResearchRequest,
    current_user: User = Depends(get_current_user),
    model_manager: ModelManager = Depends(get_model_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
) -> ResearchResponse:
    """
    Execute comprehensive research workflow using multi-agent system.
    Provides in-depth analysis with fact verification and citations.
    """
    try:
        # Initialize multi-agent orchestrator
        orchestrator = MultiAgentOrchestrator(model_manager, cache_manager)
        # Build constraints dict from request fields
        constraints = {
            "cost_budget": request.cost_budget,
            "time_budget": request.time_budget,
            "depth_level": request.depth_level,
            "sources": request.sources
        }
        # Execute research workflow
        research_results = await orchestrator.execute_research_workflow(
            research_question=request.research_question,
            methodology=request.methodology,
            constraints=constraints
        )
        if research_results["success"]:
            return ResearchResponse(
                status="success",
                data=research_results,
                metadata=ResponseMetadata(
                    query_id=research_results["workflow_metadata"]["workflow_id"],
                    execution_time=research_results["workflow_metadata"]["total_execution_time"],
                    cost=_calculate_research_cost(research_results),
                    models_used=research_results["workflow_metadata"]["agents_used"],
                    confidence=research_results["confidence_score"],
                    cached=False
                )
            )
        else:
            # Provide all required fields for ResponseMetadata in error case
            return ResearchResponse(
                status="error",
                data={"error": research_results.get("error", "Unknown research error")},
                metadata=ResponseMetadata(
                    query_id=research_results.get("workflow_id", str(uuid.uuid4())),
                    correlation_id=str(uuid.uuid4()),
                    execution_time=0.0,
                    cost=0.0,
                    models_used=[],
                    confidence=0.0,
                    cached=False
                )
            )
    except Exception as e:
        logger.error(f"Research API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Research execution failed: {str(e)}")

def _calculate_research_cost(research_results: Dict[str, any]) -> float:
    """Calculate total cost for research workflow execution."""
    base_cost = 0.10  # Base research cost
    agent_count = len(research_results.get("workflow_metadata", {}).get("agents_used", []))
    execution_time = research_results.get("workflow_metadata", {}).get("total_execution_time", 0)
    # Cost calculation: base + (agents * 0.02) + (time * 0.001)
    total_cost = base_cost + (agent_count * 0.02) + (execution_time * 0.001)
    return round(total_cost, 4)
