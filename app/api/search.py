"""
Real search API implementation with proper coroutine safety.
Provides search endpoints with proper security, validation and no coroutine leaks.
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Body
from pydantic import BaseModel, Field
import time
import uuid
import logging
import asyncio

from app.api.security import get_current_user, require_permission
from app.schemas.responses import SearchResponse, SearchData, create_success_response, create_error_response
from app.core.logging import get_logger, get_correlation_id, log_performance
from app.core.async_utils import ensure_awaited, safe_execute, coroutine_safe, AsyncSafetyValidator
from app.schemas.requests import SearchRequest
import json

router = APIRouter()
logger = get_logger("api.search")

@router.get("/health")
async def search_health(request: Request):
    search_system = getattr(request.app.state, 'search_system', None)
    search_available = search_system is not None
    return {
        "status": "healthy" if search_available else "degraded",
        "service": "search",
        "search_system": "available" if search_available else "initializing",
        "providers": ["brave_search", "duckduckgo", "scrapingbee"] if search_available else [],
        "timestamp": time.time(),
        "correlation_id": get_correlation_id()
    }

@router.post("/basic", response_model=SearchResponse)
@log_performance("basic_search")
@coroutine_safe(timeout=60.0)
async def basic_search(
    req: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    # Parse request body manually to avoid FastAPI wrapper
    try:
        data = await req.json()
        search_request = SearchRequest(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request body: {str(e)}")
    query_id = str(uuid.uuid4())
    correlation_id = get_correlation_id()
    start_time = time.time()
    logger.info(
        "Search request started",
        query=search_request.query,
        query_id=query_id,
        user_id=current_user["user_id"],
        correlation_id=correlation_id
    )
    try:
        search_system = getattr(req.app.state, 'search_system', None)
        if search_system and hasattr(search_system, 'execute_optimized_search'):
            search_result = await safe_execute(
                search_system.execute_optimized_search,
                query=search_request.query,
                budget=search_request.budget,
                quality=search_request.quality,
                max_results=search_request.max_results,
                timeout=30.0
            )
            search_result = await ensure_awaited(search_result)
            logger.debug(f"Search result type after safety: {type(search_result)}")
            if isinstance(search_result, dict):
                response_text = search_result.get("response", "Search completed")
                citations = search_result.get("citations", [])
                metadata = search_result.get("metadata", {})
            else:
                logger.warning(f"Unexpected search result format: {type(search_result)}")
                response_text = "Search completed"
                citations = []
                metadata = {}
            search_data_obj = SearchData(
                query=search_request.query,
                results=[],
                summary=response_text,
                total_results=0,
                search_time=time.time() - start_time,
                sources_consulted=citations
            )
            response_metadata = {
                "query_id": query_id,
                "correlation_id": correlation_id,
                "execution_time": time.time() - start_time,
                "cost": metadata.get("total_cost", 0.0),
                "models_used": metadata.get("models_used", []),
                "confidence": metadata.get("confidence", 1.0),
            }
            return create_success_response(
                data=search_data_obj,
                metadata=response_metadata
            )
        else:
            logger.warning("Search system not available, using fallback.")
            return create_safe_search_fallback(query_id, correlation_id, search_request.query)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return create_error_response(
            message="Search failed",
            error_code="SEARCH_ERROR",
            query_id=query_id,
            correlation_id=correlation_id,
            suggestions=["Try again later", "Check your query"]
        )

@router.post("/advanced", response_model=SearchResponse)
@require_permission("advanced_search")
@log_performance("advanced_search")
@coroutine_safe(timeout=120.0)
async def advanced_search(
    req: Request,
    body: SearchRequest = Body(..., embed=False),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    query_id = str(uuid.uuid4())
    correlation_id = get_correlation_id()
    start_time = time.time()
    logger.info(
        "Advanced search initiated",
        query=body.query,
        user_id=current_user["user_id"],
        filters=body.filters,
        budget=body.budget,
        quality=body.quality,
        query_id=query_id,
        correlation_id=correlation_id
    )
    try:
        search_system = getattr(req.app.state, 'search_system', None)
        if search_system:
            enhanced_query = _enhance_query_with_filters(
                body.query, 
                body.filters, 
                body.domains
            )
            search_result = await safe_execute(
                search_system.execute_optimized_search,
                query=enhanced_query,
                budget=body.budget,
                quality=body.quality,
                max_results=20,
                timeout=60.0
            )
            search_result = await ensure_awaited(search_result)
            advanced_results = _filter_advanced_results(
                search_result.get("citations", []), 
                body
            )
            search_data_obj = SearchData(
                query=body.query,
                results=advanced_results,
                summary=f"Advanced search found {len(advanced_results)} filtered results. {search_result['response']}" if len(advanced_results) > 0 else "No results found matching your advanced criteria.",
                total_results=len(advanced_results),
                search_time=search_result["metadata"]["execution_time"],
                sources_consulted=_extract_real_sources(search_result)
            )
            response = SearchResponse(
                status="success",
                data=search_data_obj,
                metadata={
                    "query_id": query_id,
                    "correlation_id": correlation_id,
                    "execution_time": search_result["metadata"]["execution_time"],
                    "cost": search_result["metadata"]["total_cost"],
                    "models_used": ["advanced_search_graph", "smart_router"],
                    "confidence": search_result["metadata"].get("confidence_score", 0.9),
                    "cached": False,
                    "search_provider": search_result["metadata"].get("provider_used", "multi"),
                    "enhanced": search_result["metadata"].get("enhanced", False),
                    "advanced_filters": body.filters,
                    "search_enabled": True
                }
            )
        else:
            search_data_obj = SearchData(
                query=body.query,
                results=[],
                summary="Advanced search system is initializing. Please try again shortly.",
                total_results=0,
                search_time=time.time() - start_time,
                sources_consulted=[]
            )
            response = SearchResponse(
                status="success",
                data=search_data_obj,
                metadata={
                    "query_id": query_id,
                    "correlation_id": correlation_id,
                    "execution_time": time.time() - start_time,
                    "cost": 0.0,
                    "models_used": [],
                    "confidence": 0.0,
                    "cached": False,
                    "search_system": "initializing"
                }
            )
        AsyncSafetyValidator.assert_no_coroutines(response, "Advanced search response contains coroutines")
        return response
    except Exception as e:
        logger.error(f"Advanced search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                message="Advanced search failed",
                error_code="ADVANCED_SEARCH_ERROR",
                query_id=query_id,
                correlation_id=correlation_id,
                technical_details=str(e)
            ).dict()
        )

@router.post("/test")
async def search_test(*, request: SearchRequest):
    return {
        "status": "success",
        "query": request.query,
        "mock_results": [
            {"title": "Test Result 1", "url": "https://example.com/1"},
            {"title": "Test Result 2", "url": "https://example.com/2"}
        ],
        "timestamp": time.time()
    }

def _enhance_query_with_filters(
    query: str, 
    filters: Dict[str, Any], 
    domains: Optional[List[str]]
) -> str:
    enhanced = query
    if filters:
        for key, value in filters.items():
            if isinstance(value, str) and value:
                enhanced += f" {key}:{value}"
    if domains:
        domain_filter = " OR ".join([f"site:{domain}" for domain in domains])
        enhanced += f" ({domain_filter})"
    return enhanced

def _filter_advanced_results(citations: List[Dict], advanced_data: SearchRequest) -> List[Dict]:
    filtered_results = []
    for citation in citations:
        if advanced_data.date_range:
            pass
        if advanced_data.domains:
            citation_url = citation.get("url", "")
            if not any(domain in citation_url for domain in advanced_data.domains):
                continue
        if advanced_data.language and advanced_data.language != "en":
            pass
        if advanced_data.safe_search:
            pass
        filtered_results.append(citation)
    return filtered_results

def _extract_real_sources(search_result: Dict) -> List[str]:
    sources = []
    if "citations" in search_result:
        for citation in search_result["citations"]:
            if isinstance(citation, dict) and "url" in citation:
                sources.append(citation["url"])
            elif isinstance(citation, str):
                sources.append(citation)
    if "metadata" in search_result:
        provider = search_result["metadata"].get("provider_used")
        if provider:
            sources.append(f"search_provider:{provider}")
    return sources

def create_safe_search_fallback(query_id: str, correlation_id: str, query: str) -> SearchResponse:
    search_data = SearchData(
        query=query,
        results=[],
        summary="Search system encountered a technical issue. Please try again.",
        total_results=0,
        search_time=0.0,
        sources_consulted=[]
    )
    return SearchResponse(
        status="error",
        data=search_data,
        metadata={
            "query_id": query_id,
            "correlation_id": correlation_id,
            "execution_time": 0.0,
            "cost": 0.0,
            "models_used": ["fallback"],
            "confidence": 0.0,
            "cached": False,
            "error": "coroutine_safety_failure"
        }
    )
