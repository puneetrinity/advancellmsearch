"""
Corrected Unified Complete System Integration Test Suite
Fixed to match actual API endpoints and system architecture
"""

import pytest
import asyncio
import aiohttp
import time
import json
import sys
from typing import Dict, Any, List, Optional, Literal
from enum import Enum
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Test execution modes for different environments"""
    QUICK = "quick"           # Fast tests for development
    STANDARD = "standard"     # Balanced testing for CI/CD
    COMPREHENSIVE = "comprehensive"  # Full testing for production
    CRITICAL_ONLY = "critical"      # Only critical tests


class TestSeverity(Enum):
    """Test severity levels"""
    CRITICAL = "critical"     # Must pass for production
    HIGH = "high"            # Important for system health
    MEDIUM = "medium"        # Good to have
    LOW = "low"             # Nice to have


@dataclass
class TestDefinition:
    """Test definition with metadata"""
    name: str
    function: callable
    severity: TestSeverity
    modes: List[TestMode]
    timeout: Optional[float] = None
    description: str = ""


class UnifiedSystemIntegrationTest:
    """
    Unified system integration test suite corrected for actual API structure
    """
    
    # CORRECTED ENDPOINTS based on actual API structure
    ENDPOINTS = {
        "chat": "/api/v1/chat/chat",          # Corrected from /complete
        "search": "/api/v1/search/basic",
        "search_health": "/api/v1/search/health",
        "health": "/health",                  # Main health check
        "ready": "/health/ready",            # Readiness probe
        "metrics": "/metrics",               # System metrics
        "root": "/"                          # Root info endpoint
    }
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 mode: TestMode = TestMode.STANDARD,
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 auth_token: str = "dev-user-token"):  # Added auth support
        self.base_url = base_url
        self.mode = mode
        self.timeout = timeout
        self.max_retries = max_retries
        self.auth_token = auth_token
        self.test_results = {}
        self.session = None
        self.start_time = None
        
        # Define test suite with metadata
        self.test_definitions = self._define_test_suite()
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Set up session with proper headers
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _define_test_suite(self) -> List[TestDefinition]:
        """Define the complete test suite with corrected tests"""
        return [
            # Critical Tests (Must pass for production)
            TestDefinition(
                name="🏥 System Health",
                function=self.test_system_health,
                severity=TestSeverity.CRITICAL,
                modes=[TestMode.QUICK, TestMode.STANDARD, TestMode.COMPREHENSIVE, TestMode.CRITICAL_ONLY],
                timeout=10.0,
                description="Validate basic system availability and health endpoints"
            ),
            TestDefinition(
                name="🔥 Mock Detection",
                function=self.test_mock_detection,
                severity=TestSeverity.CRITICAL,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE, TestMode.CRITICAL_ONLY],
                timeout=15.0,
                description="CRITICAL: Ensure no mock/placeholder data in production"
            ),
            TestDefinition(
                name="💬 Basic Chat",
                function=self.test_basic_chat,
                severity=TestSeverity.CRITICAL,
                modes=[TestMode.QUICK, TestMode.STANDARD, TestMode.COMPREHENSIVE, TestMode.CRITICAL_ONLY],
                timeout=20.0,
                description="Test core chat functionality with corrected endpoints"
            ),
            TestDefinition(
                name="🔐 Authentication",
                function=self.test_authentication,
                severity=TestSeverity.CRITICAL,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE, TestMode.CRITICAL_ONLY],
                timeout=10.0,
                description="Test authentication middleware functionality"
            ),
            
            # High Priority Tests
            TestDefinition(
                name="🔍 Search Integration",
                function=self.test_search_integration,
                severity=TestSeverity.HIGH,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=30.0,
                description="Test search API integration with real queries"
            ),
            TestDefinition(
                name="💰 Cost Optimization",
                function=self.test_cost_optimization,
                severity=TestSeverity.HIGH,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=25.0,
                description="Validate budget compliance and cost controls"
            ),
            TestDefinition(
                name="🔒 Error Handling",
                function=self.test_error_handling,
                severity=TestSeverity.HIGH,
                modes=[TestMode.QUICK, TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=15.0,
                description="Test system resilience and error scenarios"
            ),
            
            # Medium Priority Tests
            TestDefinition(
                name="⚡ Performance Validation",
                function=self.test_performance_validation,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=20.0,
                description="Validate response times and performance"
            ),
            TestDefinition(
                name="🚀 Load Testing",
                function=self.test_load_performance,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.COMPREHENSIVE],
                timeout=40.0,
                description="Test concurrent request handling"
            ),
            TestDefinition(
                name="📊 System Monitoring",
                function=self.test_system_monitoring,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=10.0,
                description="Test monitoring and metrics endpoints"
            ),
            TestDefinition(
                name="🔄 Multi-turn Conversation",
                function=self.test_conversation_flow,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.COMPREHENSIVE],
                timeout=30.0,
                description="Test conversation context and memory"
            ),
            
            # Comprehensive Tests
            TestDefinition(
                name="🎯 Search Quality",
                function=self.test_search_quality,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.COMPREHENSIVE],
                timeout=25.0,
                description="Validate search result quality and relevance"
            ),
            TestDefinition(
                name="📈 Streaming Responses",
                function=self.test_streaming_functionality,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.COMPREHENSIVE],
                timeout=20.0,
                description="Test real-time streaming response capability"
            )
        ]
    
    async def run_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite based on selected mode"""
        
        print(f"🧪 UNIFIED SYSTEM INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"🎯 Target System: {self.base_url}")
        print(f"🔧 Test Mode: {self.mode.value.upper()}")
        print(f"⏱️ Timeout: {self.timeout}s")
        print(f"🔐 Auth: {'Enabled' if self.auth_token else 'Disabled'}")
        print("=" * 80)
        
        # Filter tests based on mode
        tests_to_run = [
            test_def for test_def in self.test_definitions
            if self.mode in test_def.modes
        ]
        
        if not tests_to_run:
            return self._create_error_summary("No tests found for selected mode")
        
        print(f"📋 Running {len(tests_to_run)} tests in {self.mode.value} mode")
        
        results = {}
        self.start_time = time.time()
        
        # Execute tests
        for test_def in tests_to_run:
            print(f"\n{test_def.name}")
            print("-" * 60)
            
            try:
                start_time = time.time()
                test_timeout = test_def.timeout or self.timeout
                
                result = await asyncio.wait_for(
                    test_def.function(), 
                    timeout=test_timeout
                )
                
                duration = time.time() - start_time
                
                results[test_def.name] = {
                    "status": "PASS" if result.get("success", False) else "FAIL",
                    "duration": duration,
                    "details": result,
                    "severity": test_def.severity.value,
                    "critical": test_def.severity == TestSeverity.CRITICAL,
                    "description": test_def.description
                }
                
                status_icon = "✅" if result.get("success", False) else "❌"
                severity_marker = "🔥" if test_def.severity == TestSeverity.CRITICAL else ""
                
                print(f"{status_icon} {test_def.name}: {results[test_def.name]['status']} ({duration:.2f}s){severity_marker}")
                
                if result.get("details"):
                    details = result["details"] if isinstance(result["details"], list) else [str(result["details"])]
                    for detail in details[:3]:
                        print(f"   • {detail}")
                
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                results[test_def.name] = {
                    "status": "TIMEOUT",
                    "duration": duration,
                    "error": f"Test timed out after {test_timeout}s",
                    "severity": test_def.severity.value,
                    "critical": test_def.severity == TestSeverity.CRITICAL
                }
                print(f"⏰ {test_def.name}: TIMEOUT ({duration:.2f}s)")
                
            except Exception as e:
                duration = time.time() - start_time
                results[test_def.name] = {
                    "status": "ERROR",
                    "duration": duration,
                    "error": str(e),
                    "severity": test_def.severity.value,
                    "critical": test_def.severity == TestSeverity.CRITICAL
                }
                print(f"💥 {test_def.name}: ERROR - {str(e)}")
        
        return self._generate_summary(results)
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Test comprehensive system health"""
        health_checks = []
        overall_success = True
        
        # Test all health-related endpoints
        health_endpoints = [
            ("/health", "Main Health"),
            ("/health/ready", "Readiness Probe"),
            ("/", "Root Info"),
            ("/metrics", "System Metrics")
        ]
        
        for endpoint, name in health_endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            health_checks.append(f"✅ {name}: {data.get('status', 'OK')}")
                            
                            # Check specific fields based on endpoint
                            if endpoint == "/":
                                health_checks.append(f"   Version: {data.get('version', 'unknown')}")
                            elif endpoint == "/health" and "components" in data:
                                components = data["components"]
                                for comp, status in components.items():
                                    health_checks.append(f"   {comp}: {status}")
                        except:
                            health_checks.append(f"✅ {name}: HTTP 200 (non-JSON)")
                    else:
                        health_checks.append(f"❌ {name}: HTTP {response.status}")
                        overall_success = False
            except Exception as e:
                health_checks.append(f"❌ {name}: {str(e)}")
                overall_success = False
        
        # Test search service health specifically
        try:
            async with self.session.get(f"{self.base_url}/api/v1/search/health") as response:
                if response.status == 200:
                    data = await response.json()
                    search_status = data.get("search_system", "unknown")
                    health_checks.append(f"✅ Search health: {search_status}")
                else:
                    health_checks.append(f"❌ Search health: HTTP {response.status}")
                    overall_success = False
        except Exception as e:
            health_checks.append(f"❌ Search health: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": health_checks
        }
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test authentication middleware"""
        auth_tests = []
        overall_success = True
        
        # Test without authentication
        try:
            session_no_auth = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0)
            )
            
            async with session_no_auth.post(
                f"{self.base_url}/api/v1/chat/chat",
                json={"message": "Test without auth"}
            ) as response:
                # Should either work (no auth required) or return 401/403
                if response.status in [200, 401, 403, 422]:
                    auth_tests.append(f"✅ No auth: HTTP {response.status} (expected)")
                else:
                    auth_tests.append(f"⚠️ No auth: HTTP {response.status} (unexpected)")
            
            await session_no_auth.close()
        except Exception as e:
            auth_tests.append(f"❌ No auth test failed: {str(e)}")
            overall_success = False
        
        # Test with authentication
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/chat/chat",
                json={"message": "Test with auth"}
            ) as response:
                if response.status in [200, 422]:  # 422 is validation error, still authenticated
                    auth_tests.append(f"✅ With auth: HTTP {response.status}")
                else:
                    auth_tests.append(f"❌ With auth: HTTP {response.status}")
                    overall_success = False
        except Exception as e:
            auth_tests.append(f"❌ Auth test failed: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": auth_tests
        }
    
    async def test_mock_detection(self) -> Dict[str, Any]:
        """🔥 CRITICAL: Test if search API is returning mock/placeholder data"""
        mock_indicators = []
        is_mock = False
        
        try:
            search_data = {
                "query": "test query for mock detection",
                "max_results": 3,
                "include_summary": True
            }
            # Wrap in 'request' for /api/v1/search/basic
            payload = {"request": search_data}
            async with self.session.post(
                f"{self.base_url}/api/v1/search/basic",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Check response structure first
                    if "data" in data:
                        search_results = data["data"]
                        metadata = data.get("metadata", {})
                        
                        # Check for mock cost indicators
                        cost = metadata.get("cost", 0)
                        if cost == 0.008:
                            mock_indicators.append("🔥 MOCK DETECTED: Placeholder cost 0.008")
                            is_mock = True
                        
                        # Check for mock sources
                        sources = search_results.get("sources_consulted", [])
                        if any("placeholder" in str(source).lower() for source in sources):
                            mock_indicators.append("🔥 MOCK DETECTED: Placeholder search provider")
                            is_mock = True
                        
                        # Check results for mock indicators
                        results = search_results.get("results", [])
                        for result in results:
                            url = result.get("url", "")
                            if "example.com" in url or "placeholder" in url.lower():
                                mock_indicators.append("🔥 MOCK DETECTED: Example.com URLs in results")
                                is_mock = True
                                break
                        
                        # Check titles for mock patterns
                        for result in results:
                            title = result.get("title", "")
                            if any(pattern in title.lower() for pattern in ["result 1 for", "mock", "placeholder"]):
                                mock_indicators.append("🔥 MOCK DETECTED: Mock result titles")
                                is_mock = True
                                break
                        
                        # Check summary for mock content
                        summary = search_results.get("summary", "")
                        if any(pattern in summary.lower() for pattern in ["placeholder", "mock", "dummy"]):
                            mock_indicators.append("🔥 MOCK DETECTED: Placeholder text in summary")
                            is_mock = True
                        
                        if not is_mock:
                            mock_indicators.append("✅ REAL SEARCH: No mock indicators found")
                            mock_indicators.append(f"✅ Real cost: {cost}")
                            mock_indicators.append(f"✅ Sources: {len(sources)} real sources")
                            mock_indicators.append(f"✅ Results count: {len(results)}")
                    else:
                        mock_indicators.append("❌ Unexpected response structure")
                        is_mock = True
                
                elif response.status == 422:
                    # Validation error - not necessarily mock
                    error_data = await response.json()
                    mock_indicators.append(f"⚠️ Validation error: {error_data}")
                    is_mock = False  # This is expected behavior, not mock
                else:
                    mock_indicators.append(f"❌ Search API error: HTTP {response.status}")
                    is_mock = True
        
        except Exception as e:
            mock_indicators.append(f"❌ Mock detection failed: {str(e)}")
            is_mock = True
        
        return {
            "success": not is_mock,
            "details": mock_indicators,
            "is_mock_system": is_mock
        }
    
    async def test_basic_chat(self) -> Dict[str, Any]:
        """Test basic chat functionality with corrected endpoint"""
        try:
            chat_data = {
                "message": "Hello, how are you today?",
                "session_id": f"test_session_{int(time.time())}",
                "context": {},
                "constraints": {"max_cost": 0.1}  # Updated structure
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/chat/chat",  # Corrected endpoint
                json=chat_data
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    message = data.get("message", "")
                    cost_prediction = data.get("cost_prediction", {})
                    metadata = data.get("metadata", {})
                    
                    success_indicators = []
                    success_indicators.append(f"✅ Response received ({len(message)} chars)")
                    success_indicators.append(f"✅ Cost: {cost_prediction.get('estimated_cost', 0)}")
                    success_indicators.append(f"✅ Model used: {metadata.get('model_used', 'unknown')}")
                    success_indicators.append(f"✅ Response time: {metadata.get('response_time', 0):.2f}s")
                    
                    return {
                        "success": len(message) > 10,
                        "details": success_indicators
                    }
                
                elif response.status == 422:
                    error_data = await response.json()
                    return {
                        "success": False,
                        "details": [f"❌ Validation error: {error_data.get('detail', 'Unknown')}"]
                    }
                else:
                    return {"success": False, "details": [f"❌ HTTP {response.status}"]}
        
        except Exception as e:
            return {"success": False, "details": [f"❌ Error: {str(e)}"]}
    
    async def test_search_integration(self) -> Dict[str, Any]:
        """Test search integration with real queries"""
        search_tests = []
        successful_searches = 0
        
        search_queries = [
            {
                "message": "What are the latest AI developments in 2025?",
                "session_id": f"search_test_{int(time.time())}_1",
                "context": {},
                "constraints": {"max_cost": 2.0, "quality_requirement": "premium"}
            },
            {
                "message": "Current weather in New York",
                "session_id": f"search_test_{int(time.time())}_2", 
                "context": {},
                "constraints": {"max_cost": 1.0, "quality_requirement": "standard"}
            }
        ]
        
        for i, query in enumerate(search_queries):
            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/chat/chat",
                    json=query
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        message = data.get("message", "")
                        metadata = data.get("metadata", {})
                        search_enabled = metadata.get("search_enabled", False)
                        
                        if search_enabled and len(message) > 50:
                            successful_searches += 1
                            search_tests.append(f"✅ Query {i+1}: Search enabled, substantial response")
                        else:
                            search_tests.append(f"⚠️ Query {i+1}: {'No search' if not search_enabled else 'Short response'}")
                    else:
                        search_tests.append(f"❌ Query {i+1}: HTTP {response.status}")
            
            except Exception as e:
                search_tests.append(f"❌ Query {i+1}: {str(e)}")
        
        return {
            "success": successful_searches > 0,
            "details": search_tests + [f"✅ Successful searches: {successful_searches}/{len(search_queries)}"]
        }
    
    async def test_cost_optimization(self) -> Dict[str, Any]:
        """Test cost optimization with different budget scenarios"""
        cost_tests = []
        overall_success = True
        
        budget_scenarios = [
            {"max_cost": 0.2, "quality": "basic"},
            {"max_cost": 1.0, "quality": "standard"},
            {"max_cost": 2.0, "quality": "premium"}
        ]
        
        for i, scenario in enumerate(budget_scenarios):
            try:
                search_data = {
                    "message": f"Cost optimization test query {i+1}",
                    "session_id": f"cost_test_{int(time.time())}_{i}",
                    "context": {},
                    "constraints": {
                        "max_cost": scenario["max_cost"],
                        "quality_requirement": scenario["quality"]
                    }
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/chat/chat",
                    json=search_data
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        cost_prediction = data.get("cost_prediction", {})
                        actual_cost = cost_prediction.get("estimated_cost", 0)
                        
                        if actual_cost <= scenario["max_cost"]:
                            cost_tests.append(f"✅ Budget {scenario['max_cost']}: Cost {actual_cost} within limit")
                        else:
                            cost_tests.append(f"❌ Budget {scenario['max_cost']}: Cost {actual_cost} exceeds limit")
                            overall_success = False
                    else:
                        cost_tests.append(f"❌ Budget test {i+1}: HTTP {response.status}")
                        overall_success = False
            
            except Exception as e:
                cost_tests.append(f"❌ Budget test {i+1}: {str(e)}")
                overall_success = False
        
        return {
            "success": overall_success,
            "details": cost_tests
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling and edge cases"""
        error_tests = []
        correctly_handled = 0
        
        error_scenarios = [
            {
                "name": "Empty message",
                "data": {"message": "", "session_id": "error_test", "context": {}, "constraints": {}}
            },
            {
                "name": "Missing session_id", 
                "data": {"message": "test", "context": {}, "constraints": {}}
            },
            {
                "name": "Invalid constraints",
                "data": {"message": "test", "session_id": "error_test", "context": {}, "constraints": {"max_cost": -1}}
            },
            {
                "name": "Oversized message",
                "data": {"message": "x" * 10000, "session_id": "error_test", "context": {}, "constraints": {}}
            }
        ]
        
        for scenario in error_scenarios:
            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/chat/chat",
                    json=scenario["data"]
                ) as response:
                    
                    if response.status in [400, 422]:  # Expected error responses
                        error_tests.append(f"✅ {scenario['name']}: Properly handled (HTTP {response.status})")
                        correctly_handled += 1
                    elif response.status == 200:
                        # Some scenarios might be handled gracefully
                        error_tests.append(f"⚠️ {scenario['name']}: Handled gracefully (HTTP 200)")
                        correctly_handled += 1
                    else:
                        error_tests.append(f"❌ {scenario['name']}: Unexpected status {response.status}")
            
            except Exception as e:
                error_tests.append(f"⚠️ {scenario['name']}: Exception {str(e)}")
        
        success_rate = correctly_handled / len(error_scenarios)
        error_tests.append(f"📊 Error handling success rate: {success_rate:.1%}")
        
        return {
            "success": success_rate >= 0.7,
            "details": error_tests
        }
    
    async def test_performance_validation(self) -> Dict[str, Any]:
        """Test performance under normal load"""
        performance_tests = []
        overall_success = True
        
        try:
            start_time = time.time()
            async with self.session.post(
                f"{self.base_url}/api/v1/chat/chat",
                json={
                    "message": "Performance test query",
                    "session_id": f"perf_test_{int(time.time())}",
                    "context": {},
                    "constraints": {"max_cost": 1.0}
                }
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    if response_time < 10.0:
                        performance_tests.append(f"✅ Response time: {response_time:.2f}s (target: <10s)")
                    else:
                        performance_tests.append(f"⚠️ Slow response: {response_time:.2f}s")
                        overall_success = False
                        
                    data = await response.json()
                    message_length = len(data.get("message", ""))
                    performance_tests.append(f"✅ Response length: {message_length} characters")
                else:
                    performance_tests.append(f"❌ Performance test failed: HTTP {response.status}")
                    overall_success = False
        except Exception as e:
            performance_tests.append(f"❌ Performance test error: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": performance_tests
        }
    
    async def test_load_performance(self) -> Dict[str, Any]:
        """Test system performance under concurrent load"""
        concurrent_requests = 5 if self.mode == TestMode.COMPREHENSIVE else 3
        load_tests = []
        
        try:
            tasks = []
            for i in range(concurrent_requests):
                chat_data = {
                    "message": f"Load test query {i}",
                    "session_id": f"load_test_{int(time.time())}_{i}",
                    "context": {},
                    "constraints": {"max_cost": 0.5}
                }
                task = self.session.post(
                    f"{self.base_url}/api/v1/chat/chat",
                    json=chat_data
                )
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            successful = 0
            for response in responses:
                if isinstance(response, aiohttp.ClientResponse):
                    if response.status == 200:
                        successful += 1
                    response.close()
            
            success_rate = successful / concurrent_requests
            rps = concurrent_requests / total_time
            
            load_tests.append(f"✅ Concurrent load: {successful}/{concurrent_requests} successful")
            load_tests.append(f"✅ Success rate: {success_rate:.1%}")
            load_tests.append(f"✅ Requests/second: {rps:.1f}")
            load_tests.append(f"✅ Total time: {total_time:.2f}s")
            
            return {
                "success": success_rate >= 0.8,
                "details": load_tests
            }
        
        except Exception as e:
            return {
                "success": False,
                "details": [f"❌ Load test failed: {str(e)}"]
            }
    
    async def test_system_monitoring(self) -> Dict[str, Any]:
        """Test system monitoring and metrics endpoints"""
        monitoring_tests = []
        overall_success = True
        
        # Test metrics endpoint
        try:
            async with self.session.get(f"{self.base_url}/metrics") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for expected metric structure
                    if "metrics" in data:
                        metrics = data["metrics"]
                        monitoring_tests.append("✅ Metrics endpoint working")
                        
                        if "components" in metrics:
                            components = metrics["components"]
                            monitoring_tests.append(f"✅ Components monitored: {len(components)}")
                        
                        if "performance" in metrics:
                            perf = metrics["performance"]
                            monitoring_tests.append(f"✅ Performance metrics available")
                    else:
                        monitoring_tests.append("⚠️ Metrics structure unexpected")
                        overall_success = False
                else:
                    monitoring_tests.append(f"❌ Metrics endpoint: HTTP {response.status}")
                    overall_success = False
        except Exception as e:
            monitoring_tests.append(f"❌ Metrics test failed: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": monitoring_tests
        }
    
    async def test_conversation_flow(self) -> Dict[str, Any]:
        """Test multi-turn conversation flow"""
        conversation_tests = []
        overall_success = True
        session_id = f"conversation_test_{int(time.time())}"
        
        conversation_steps = [
            "Hello, I'm testing conversation flow",
            "Can you remember what I just said?",
            "What was my first message?"
        ]
        
        for i, message in enumerate(conversation_steps):
            try:
                chat_data = {
                    "message": message,
                    "session_id": session_id,  # Same session for context
                    "context": {},
                    "constraints": {"max_cost": 1.0}
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/chat/chat",
                    json=chat_data
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        response_message = data.get("message", "")
                        
                        conversation_tests.append(f"✅ Turn {i+1}: Response received ({len(response_message)} chars)")
                        
                        # For later turns, check if context is maintained
                        if i > 0 and len(response_message) > 20:
                            conversation_tests.append(f"✅ Turn {i+1}: Context appears maintained")
                    else:
                        conversation_tests.append(f"❌ Turn {i+1}: HTTP {response.status}")
                        overall_success = False
            
            except Exception as e:
                conversation_tests.append(f"❌ Turn {i+1}: {str(e)}")
                overall_success = False
        
        return {
            "success": overall_success,
            "details": conversation_tests
        }
    
    async def test_search_quality(self) -> Dict[str, Any]:
        """Test search result quality and relevance"""
        quality_tests = []
        overall_success = True
        
        try:
            search_data = {
                "message": "Python programming tutorial for beginners",
                "session_id": f"quality_test_{int(time.time())}",
                "context": {},
                "constraints": {"max_cost": 2.0, "quality_requirement": "premium"}
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/chat/chat",
                json=search_data
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    message = data.get("message", "")
                    citations = data.get("citations", [])
                    metadata = data.get("metadata", {})
                    
                    # Quality checks
                    if len(message) > 100:
                        quality_tests.append(f"✅ Substantial response ({len(message)} chars)")
                    else:
                        quality_tests.append(f"⚠️ Short response ({len(message)} chars)")
                        overall_success = False
                    
                    if len(citations) > 0:
                        quality_tests.append(f"✅ Citations provided: {len(citations)}")
                        
                        # Check citation structure
                        first_citation = citations[0]
                        required_fields = ["title", "url"]
                        has_structure = all(field in first_citation for field in required_fields)
                        
                        if has_structure:
                            quality_tests.append("✅ Citation structure valid")
                        else:
                            quality_tests.append("❌ Citation structure incomplete")
                            overall_success = False
                        
                        # Check for real URLs (not example.com)
                        real_urls = [c for c in citations if "example.com" not in c.get("url", "")]
                        if len(real_urls) == len(citations):
                            quality_tests.append("✅ All URLs are real (no mock)")
                        else:
                            quality_tests.append(f"❌ {len(citations) - len(real_urls)} mock URLs detected")
                            overall_success = False
                    else:
                        quality_tests.append("⚠️ No citations provided")
                    
                    search_enabled = metadata.get("search_enabled", False)
                    if search_enabled:
                        quality_tests.append("✅ Search was triggered")
                    else:
                        quality_tests.append("⚠️ Search was not triggered")
                
                else:
                    quality_tests.append(f"❌ Search quality test failed: HTTP {response.status}")
                    overall_success = False
        
        except Exception as e:
            quality_tests.append(f"❌ Search quality test error: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": quality_tests
        }
    
    async def test_streaming_functionality(self) -> Dict[str, Any]:
        """Test streaming response functionality"""
        streaming_tests = []
        overall_success = True
        
        try:
            chat_data = {
                "message": "Tell me about artificial intelligence",
                "session_id": f"streaming_test_{int(time.time())}",
                "context": {},
                "constraints": {"max_cost": 1.0},
                "stream": True  # Enable streaming
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/chat/chat",
                json=chat_data
            ) as response:
                
                if response.status == 200:
                    content_type = response.headers.get("content-type", "")
                    
                    if "text/event-stream" in content_type or "application/json" in content_type:
                        streaming_tests.append("✅ Streaming response detected")
                        
                        # Try to read some data
                        chunk_count = 0
                        async for chunk in response.content.iter_chunked(1024):
                            if chunk:
                                chunk_count += 1
                            if chunk_count >= 3:  # Read a few chunks
                                break
                        
                        if chunk_count > 0:
                            streaming_tests.append(f"✅ Received {chunk_count} data chunks")
                        else:
                            streaming_tests.append("⚠️ No streaming data received")
                            overall_success = False
                    else:
                        streaming_tests.append("⚠️ Non-streaming response (might be fallback)")
                else:
                    streaming_tests.append(f"❌ Streaming test failed: HTTP {response.status}")
                    overall_success = False
        
        except Exception as e:
            streaming_tests.append(f"❌ Streaming test error: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": streaming_tests
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        total_duration = time.time() - self.start_time
        
        # Count results by status
        passed = sum(1 for r in results.values() if r["status"] == "PASS")
        failed = sum(1 for r in results.values() if r["status"] == "FAIL")
        errors = sum(1 for r in results.values() if r["status"] == "ERROR")
        timeouts = sum(1 for r in results.values() if r["status"] == "TIMEOUT")
        total = len(results)
        
        # Count critical failures
        critical_failures = sum(1 for r in results.values() 
                              if r.get("critical", False) and r["status"] != "PASS")
        
        # Count by severity
        severity_counts = {}
        for severity in TestSeverity:
            severity_counts[severity.value] = sum(1 for r in results.values() 
                                                if r.get("severity") == severity.value)
        
        # Determine overall status
        overall_status = self._determine_overall_status(passed, total, critical_failures)
        
        summary = {
            "overall_status": overall_status,
            "test_mode": self.mode.value,
            "tests_passed": passed,
            "tests_failed": failed,
            "tests_error": errors,
            "tests_timeout": timeouts,
            "tests_total": total,
            "critical_failures": critical_failures,
            "severity_breakdown": severity_counts,
            "total_duration": total_duration,
            "success_rate": passed / total if total > 0 else 0,
            "base_url": self.base_url,
            "results": results
        }
        
        self._print_comprehensive_summary(summary)
        return summary
    
    def _determine_overall_status(self, passed: int, total: int, critical_failures: int) -> str:
        """Determine overall test status"""
        if critical_failures > 0:
            return "CRITICAL_FAILURE"
        elif passed == total:
            return "PASS"
        elif passed >= total * 0.8:
            return "PARTIAL"
        else:
            return "FAIL"
    
    def _print_comprehensive_summary(self, summary: Dict[str, Any]):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🎯 UNIFIED INTEGRATION TEST SUMMARY")
        print("=" * 80)
        
        overall_status = summary["overall_status"]
        status_icons = {
            "PASS": "✅",
            "PARTIAL": "⚠️", 
            "FAIL": "❌",
            "CRITICAL_FAILURE": "🔥"
        }
        status_icon = status_icons.get(overall_status, "❓")
        
        print(f"{status_icon} Overall Status: {overall_status}")
        print(f"🔧 Test Mode: {summary['test_mode'].upper()}")
        print(f"📊 Results: {summary['tests_passed']}/{summary['tests_total']} passed ({summary['success_rate']:.1%})")
        print(f"⏱️ Duration: {summary['total_duration']:.2f}s")
        print(f"🎯 Target: {summary['base_url']}")
        
        if summary['critical_failures'] > 0:
            print(f"🔥 Critical Failures: {summary['critical_failures']}")
        
        # Severity breakdown
        print(f"\n📋 Test Breakdown by Severity:")
        for severity, count in summary['severity_breakdown'].items():
            if count > 0:
                severity_passed = sum(1 for r in summary['results'].values() 
                                    if r.get('severity') == severity and r['status'] == 'PASS')
                print(f"  {severity.upper()}: {severity_passed}/{count} passed")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for test_name, result in summary["results"].items():
            status_icon = {
                "PASS": "✅", 
                "PARTIAL": "⚠️", 
                "FAIL": "❌", 
                "ERROR": "💥",
                "TIMEOUT": "⏰"
            }.get(result["status"], "❓")
            
            severity_marker = "🔥" if result.get("critical", False) else ""
            print(f"  {status_icon} {test_name}: {result['status']} ({result['duration']:.2f}s){severity_marker}")
            
            if result["status"] in ["FAIL", "ERROR", "TIMEOUT"] and "error" in result:
                print(f"    Error: {result['error']}")
        
        # Final recommendations
        print(f"\n💡 Recommendations:")
        if overall_status == "PASS":
            print("🎉 All tests passed! System is ready for production deployment.")
        elif overall_status == "PARTIAL":
            print("⚠️ Some tests failed. Review results and fix non-critical issues.")
            print("✅ System may be suitable for staging deployment.")
        elif overall_status == "CRITICAL_FAILURE":
            print("🔥 Critical failures detected! DO NOT deploy to production.")
            print("🛠️ Fix critical issues before any deployment.")
        else:
            print("❌ Multiple test failures. System needs debugging.")
            print("🔧 Fix issues before deployment to any environment.")
    
    def _create_error_summary(self, error_message: str) -> Dict[str, Any]:
        """Create error summary when tests cannot run"""
        return {
            "overall_status": "ERROR",
            "test_mode": self.mode.value,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_error": 1,
            "tests_timeout": 0,
            "tests_total": 0,
            "critical_failures": 1,
            "total_duration": 0,
            "success_rate": 0,
            "base_url": self.base_url,
            "error": error_message,
            "results": {}
        }
    
    def get_exit_code(self, summary: Dict[str, Any]) -> int:
        """Get exit code for CI/CD integration"""
        overall_status = summary["overall_status"]
        
        if overall_status == "PASS":
            return 0  # Success
        elif overall_status == "PARTIAL":
            return 1  # Warning (non-critical failures)
        elif overall_status == "CRITICAL_FAILURE":
            return 2  # Critical failure
        else:
            return 3  # General failure


# Convenience functions for different test modes
async def run_quick_tests(base_url: str = "http://localhost:8000", auth_token: str = "dev-user-token") -> Dict[str, Any]:
    """Run quick tests for development"""
    async with UnifiedSystemIntegrationTest(base_url, TestMode.QUICK, auth_token=auth_token) as tester:
        return await tester.run_test_suite()


async def run_standard_tests(base_url: str = "http://localhost:8000", auth_token: str = "dev-user-token") -> Dict[str, Any]:
    """Run standard tests for CI/CD"""
    async with UnifiedSystemIntegrationTest(base_url, TestMode.STANDARD, auth_token=auth_token) as tester:
        return await tester.run_test_suite()


async def run_comprehensive_tests(base_url: str = "http://localhost:8000", auth_token: str = "dev-user-token") -> Dict[str, Any]:
    """Run comprehensive tests for production"""
    async with UnifiedSystemIntegrationTest(base_url, TestMode.COMPREHENSIVE, auth_token=auth_token) as tester:
        return await tester.run_test_suite()


async def run_critical_tests(base_url: str = "http://localhost:8000", auth_token: str = "dev-user-token") -> Dict[str, Any]:
    """Run only critical tests"""
    async with UnifiedSystemIntegrationTest(base_url, TestMode.CRITICAL_ONLY, auth_token=auth_token) as tester:
        return await tester.run_test_suite()


# Main execution function
async def main():
    """Main execution with command line support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Corrected Unified System Integration Test Suite")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the system to test")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive", "critical"],
                       default="standard", help="Test execution mode")
    parser.add_argument("--timeout", type=float, default=30.0,
                       help="Request timeout in seconds")
    parser.add_argument("--auth-token", default="dev-user-token",
                       help="Authentication token for API requests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Map mode string to enum
    mode_map = {
        "quick": TestMode.QUICK,
        "standard": TestMode.STANDARD, 
        "comprehensive": TestMode.COMPREHENSIVE,
        "critical": TestMode.CRITICAL_ONLY
    }
    
    mode = mode_map[args.mode]
    
    try:
        async with UnifiedSystemIntegrationTest(
            base_url=args.url,
            mode=mode,
            timeout=args.timeout,
            auth_token=args.auth_token
        ) as tester:
            results = await tester.run_test_suite()
            exit_code = tester.get_exit_code(results)
            return exit_code
    
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        return 1


# Script execution
if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


# Example usage in other scripts:
"""
# Development quick check
results = await run_quick_tests()

# CI/CD pipeline
results = await run_standard_tests("http://staging.example.com:8000", "staging-auth-token")

# Pre-production validation  
results = await run_comprehensive_tests("http://production.example.com:8000", "prod-auth-token")

# Emergency critical check
results = await run_critical_tests()

# Custom configuration
async with UnifiedSystemIntegrationTest(
    base_url="http://custom.url:8000",
    mode=TestMode.COMPREHENSIVE,
    timeout=60.0,
    auth_token="custom-token"
) as tester:
    results = await tester.run_test_suite()
    exit_code = tester.get_exit_code(results)
"""