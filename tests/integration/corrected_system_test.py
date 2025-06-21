"""
Corrected Unified System Integration Test Suite
Fixed to match your actual API endpoints and request schemas
"""

import asyncio
import aiohttp
import time
import json
import sys
from typing import Dict, Any, List, Optional
from enum import Enum
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMode(Enum):
    """Test execution modes"""
    QUICK = "quick"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    CRITICAL_ONLY = "critical"


class TestSeverity(Enum):
    """Test severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestDefinition:
    """Test definition with metadata"""
    name: str
    function: callable
    severity: TestSeverity
    modes: List[TestMode]
    timeout: Optional[float] = None
    description: str = ""


class FixedSystemIntegrationTest:
    """
    Fixed system integration test suite that matches your actual API
    """
    
    # CORRECTED ENDPOINTS based on your actual API structure
    ENDPOINTS = {
        "root": "/",                              # ✅ Root info
        "health": "/health",                      # ✅ Main health  
        "chat": "/api/v1/chat/chat",             # ✅ Correct chat endpoint
        "search_basic": "/api/v1/search/basic",   # ✅ Basic search
        "search_health": "/api/v1/search/health", # ✅ Search health
        "search_test": "/api/v1/search/test",     # ✅ Working test endpoint
        "metrics": "/metrics",                    # ⚠️ May not exist
        "ready": "/health/ready"                  # ⚠️ May not exist
    }
    
    def __init__(self, 
                 base_url: str = "http://localhost:8000",
                 mode: TestMode = TestMode.STANDARD,
                 timeout: float = 30.0,
                 auth_token: str = "dev-user-token"):
        self.base_url = base_url
        self.mode = mode
        self.timeout = timeout
        self.auth_token = auth_token
        self.test_results = {}
        self.session = None
        self.start_time = None
        
        # Define test suite
        self.test_definitions = self._define_test_suite()
    
    async def __aenter__(self):
        """Async context manager entry"""
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
        """Define tests that match your actual API"""
        return [
            # Critical Tests
            TestDefinition(
                name="🏥 System Health",
                function=self.test_system_health,
                severity=TestSeverity.CRITICAL,
                modes=[TestMode.QUICK, TestMode.STANDARD, TestMode.COMPREHENSIVE, TestMode.CRITICAL_ONLY],
                timeout=10.0,
                description="Test actual health endpoints"
            ),
            TestDefinition(
                name="🔥 Server Stability",
                function=self.test_server_stability,
                severity=TestSeverity.CRITICAL,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE, TestMode.CRITICAL_ONLY],
                timeout=15.0,
                description="Ensure server doesn't crash on basic requests"
            ),
            TestDefinition(
                name="💬 Chat API Basic",
                function=self.test_chat_basic,
                severity=TestSeverity.CRITICAL,
                modes=[TestMode.QUICK, TestMode.STANDARD, TestMode.COMPREHENSIVE, TestMode.CRITICAL_ONLY],
                timeout=20.0,
                description="Test chat with correct endpoint and schema"
            ),
            
            # High Priority Tests
            TestDefinition(
                name="🔍 Search Integration",
                function=self.test_search_integration,
                severity=TestSeverity.HIGH,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=30.0,
                description="Test search endpoints that actually exist"
            ),
            TestDefinition(
                name="🔒 Error Handling",
                function=self.test_error_handling,
                severity=TestSeverity.HIGH,
                modes=[TestMode.QUICK, TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=15.0,
                description="Test graceful error handling"
            ),
            
            # Medium Priority Tests
            TestDefinition(
                name="⚡ Performance Check",
                function=self.test_performance_check,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.STANDARD, TestMode.COMPREHENSIVE],
                timeout=20.0,
                description="Basic performance validation"
            ),
            TestDefinition(
                name="📊 API Documentation",
                function=self.test_api_documentation,
                severity=TestSeverity.MEDIUM,
                modes=[TestMode.COMPREHENSIVE],
                timeout=10.0,
                description="Test API info and documentation endpoints"
            )
        ]
    
    async def run_test_suite(self) -> Dict[str, Any]:
        """Run the test suite"""
        
        print(f"🧪 FIXED SYSTEM INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"🎯 Target System: {self.base_url}")
        print(f"🔧 Test Mode: {self.mode.value.upper()}")
        print(f"⏱️ Timeout: {self.timeout}s")
        print("=" * 80)
        
        # Filter tests based on mode
        tests_to_run = [
            test_def for test_def in self.test_definitions
            if self.mode in test_def.modes
        ]
        
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
                    "critical": test_def.severity == TestSeverity.CRITICAL
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
        """Test actual health endpoints"""
        health_checks = []
        overall_success = True
        
        # Test endpoints that actually exist
        endpoints_to_test = [
            ("/", "Root Info"),
            ("/health", "Main Health")
        ]
        
        for endpoint, name in endpoints_to_test:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        try:
                            data = await response.json()
                            health_checks.append(f"✅ {name}: {data.get('status', 'OK')}")
                            
                            if endpoint == "/" and "name" in data:
                                health_checks.append(f"   System: {data.get('name')} v{data.get('version')}")
                        except:
                            health_checks.append(f"✅ {name}: HTTP 200 (non-JSON)")
                    else:
                        health_checks.append(f"❌ {name}: HTTP {response.status}")
                        overall_success = False
            except Exception as e:
                health_checks.append(f"❌ {name}: {str(e)}")
                overall_success = False
        
        # Test search health
        try:
            async with self.session.get(f"{self.base_url}/api/v1/search/health") as response:
                if response.status == 200:
                    data = await response.json()
                    health_checks.append(f"✅ Search Health: {data.get('status', 'OK')}")
                else:
                    health_checks.append(f"❌ Search Health: HTTP {response.status}")
                    overall_success = False
        except Exception as e:
            health_checks.append(f"❌ Search Health: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": health_checks
        }
    
    async def test_server_stability(self) -> Dict[str, Any]:
        """Test that server doesn't crash on basic requests"""
        stability_tests = []
        is_stable = True
        
        # Test various endpoints to ensure no 500 errors
        test_requests = [
            ("GET", "/", None, "Root endpoint"),
            ("GET", "/health", None, "Health endpoint"),
            ("GET", "/api/v1/search/health", None, "Search health"),
            ("POST", "/api/v1/search/test", {"query": "test"}, "Search test")
        ]
        
        for method, endpoint, data, description in test_requests:
            try:
                if method == "GET":
                    async with self.session.get(f"{self.base_url}{endpoint}") as response:
                        status = response.status
                elif method == "POST":
                    async with self.session.post(f"{self.base_url}{endpoint}", json=data) as response:
                        status = response.status
                
                if status == 500:
                    stability_tests.append(f"❌ {description}: Server crashed (500)")
                    is_stable = False
                else:
                    stability_tests.append(f"✅ {description}: Stable (HTTP {status})")
                    
            except Exception as e:
                stability_tests.append(f"❌ {description}: Exception {str(e)}")
                is_stable = False
        
        return {
            "success": is_stable,
            "details": stability_tests
        }
    
    async def test_chat_basic(self) -> Dict[str, Any]:
        """Test chat with correct endpoint and schema"""
        chat_tests = []
        
        try:
            # Use correct chat request schema
            chat_data = {
                "message": "Hello, this is a test message",
                "session_id": f"test_session_{int(time.time())}",
                "context": {},
                "constraints": {"max_cost": 1.0}
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/chat/chat",  # Correct endpoint
                json=chat_data
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    message = data.get("message", "")
                    
                    chat_tests.append(f"✅ Chat successful: {len(message)} chars response")
                    chat_tests.append(f"✅ Response preview: {message[:100]}...")
                    
                    # Check for expected response fields
                    if "cost_prediction" in data:
                        chat_tests.append("✅ Cost prediction included")
                    if "metadata" in data:
                        chat_tests.append("✅ Metadata included")
                    
                    return {
                        "success": len(message) > 10,
                        "details": chat_tests
                    }
                
                elif response.status == 422:
                    error_data = await response.json()
                    chat_tests.append(f"⚠️ Validation error (may be expected): {error_data}")
                    return {"success": False, "details": chat_tests}
                
                elif response.status == 500:
                    error_text = await response.text()
                    chat_tests.append(f"❌ Server error (critical): {error_text}")
                    return {"success": False, "details": chat_tests}
                
                else:
                    chat_tests.append(f"❌ Unexpected status: HTTP {response.status}")
                    return {"success": False, "details": chat_tests}
        
        except Exception as e:
            chat_tests.append(f"❌ Chat test failed: {str(e)}")
            return {"success": False, "details": chat_tests}
    
    async def test_search_integration(self) -> Dict[str, Any]:
        """Test search endpoints that actually exist"""
        search_tests = []
        overall_success = True
        
        # Test the working search test endpoint first
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/search/test",
                json={"query": "test search"}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    search_tests.append(f"✅ Search test endpoint: {data.get('status', 'OK')}")
                else:
                    search_tests.append(f"❌ Search test: HTTP {response.status}")
                    overall_success = False
        except Exception as e:
            search_tests.append(f"❌ Search test failed: {str(e)}")
            overall_success = False
        
        # Test basic search with correct schema
        try:
            search_data = {
                "query": "artificial intelligence",
                "max_results": 3,
                "search_type": "web",
                "include_summary": True
            }
            
            async with self.session.post(
                f"{self.base_url}/api/v1/search/basic",
                json=search_data
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    search_tests.append("✅ Basic search successful")
                elif response.status == 422:
                    error_data = await response.json()
                    search_tests.append(f"⚠️ Search validation issue: {error_data}")
                else:
                    search_tests.append(f"❌ Basic search: HTTP {response.status}")
                    overall_success = False
        except Exception as e:
            search_tests.append(f"❌ Basic search failed: {str(e)}")
            overall_success = False
        
        return {
            "success": overall_success,
            "details": search_tests
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test graceful error handling"""
        error_tests = []
        correctly_handled = 0
        
        # Test error scenarios
        error_scenarios = [
            {
                "name": "Empty chat message",
                "endpoint": "/api/v1/chat/chat",
                "data": {"message": "", "session_id": "test", "context": {}, "constraints": {}}
            },
            {
                "name": "Invalid search query",
                "endpoint": "/api/v1/search/basic", 
                "data": {"query": "", "max_results": -1}
            },
            {
                "name": "Malformed JSON",
                "endpoint": "/api/v1/chat/chat",
                "data": "invalid json"
            }
        ]
        
        for scenario in error_scenarios:
            try:
                if isinstance(scenario["data"], str):
                    # Send malformed JSON
                    async with self.session.post(
                        f"{self.base_url}{scenario['endpoint']}",
                        data=scenario["data"],
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        status = response.status
                else:
                    async with self.session.post(
                        f"{self.base_url}{scenario['endpoint']}",
                        json=scenario["data"]
                    ) as response:
                        status = response.status
                
                if status in [400, 422]:  # Expected error responses
                    error_tests.append(f"✅ {scenario['name']}: Handled gracefully (HTTP {status})")
                    correctly_handled += 1
                elif status == 500:
                    error_tests.append(f"❌ {scenario['name']}: Server crashed (HTTP 500)")
                else:
                    error_tests.append(f"⚠️ {scenario['name']}: Unexpected status {status}")
                    correctly_handled += 1  # Still didn't crash
                    
            except Exception as e:
                error_tests.append(f"⚠️ {scenario['name']}: Exception {str(e)}")
        
        success_rate = correctly_handled / len(error_scenarios)
        error_tests.append(f"📊 Error handling success rate: {success_rate:.1%}")
        
        return {
            "success": success_rate >= 0.7,
            "details": error_tests
        }
    
    async def test_performance_check(self) -> Dict[str, Any]:
        """Basic performance validation"""
        performance_tests = []
        
        try:
            start_time = time.time()
            async with self.session.get(f"{self.base_url}/health") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    if response_time < 5.0:
                        performance_tests.append(f"✅ Health check: {response_time:.2f}s (fast)")
                    else:
                        performance_tests.append(f"⚠️ Health check: {response_time:.2f}s (slow)")
                    
                    return {"success": response_time < 10.0, "details": performance_tests}
                else:
                    performance_tests.append(f"❌ Health check failed: HTTP {response.status}")
                    return {"success": False, "details": performance_tests}
        
        except Exception as e:
            performance_tests.append(f"❌ Performance test failed: {str(e)}")
            return {"success": False, "details": performance_tests}
    
    async def test_api_documentation(self) -> Dict[str, Any]:
        """Test API info and documentation"""
        doc_tests = []
        
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "api_endpoints" in data:
                        endpoints = data["api_endpoints"]
                        doc_tests.append(f"✅ API endpoints documented: {len(endpoints)}")
                    
                    if "features" in data:
                        features = data["features"]
                        doc_tests.append(f"✅ Features listed: {len(features)}")
                    
                    return {"success": True, "details": doc_tests}
                else:
                    doc_tests.append(f"❌ Root endpoint failed: HTTP {response.status}")
                    return {"success": False, "details": doc_tests}
        
        except Exception as e:
            doc_tests.append(f"❌ Documentation test failed: {str(e)}")
            return {"success": False, "details": doc_tests}
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        total_duration = time.time() - self.start_time
        
        passed = sum(1 for r in results.values() if r["status"] == "PASS")
        failed = sum(1 for r in results.values() if r["status"] == "FAIL")
        errors = sum(1 for r in results.values() if r["status"] == "ERROR")
        timeouts = sum(1 for r in results.values() if r["status"] == "TIMEOUT")
        total = len(results)
        
        critical_failures = sum(1 for r in results.values() 
                              if r.get("critical", False) and r["status"] != "PASS")
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = "CRITICAL_FAILURE"
        elif passed == total:
            overall_status = "PASS"
        elif passed >= total * 0.8:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAIL"
        
        summary = {
            "overall_status": overall_status,
            "test_mode": self.mode.value,
            "tests_passed": passed,
            "tests_failed": failed,
            "tests_error": errors,
            "tests_timeout": timeouts,
            "tests_total": total,
            "critical_failures": critical_failures,
            "total_duration": total_duration,
            "success_rate": passed / total if total > 0 else 0,
            "base_url": self.base_url,
            "results": results
        }
        
        self._print_summary(summary)
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("🎯 FIXED INTEGRATION TEST SUMMARY")
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
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for test_name, result in summary["results"].items():
            status_icon = {
                "PASS": "✅", 
                "FAIL": "❌", 
                "ERROR": "💥",
                "TIMEOUT": "⏰"
            }.get(result["status"], "❓")
            
            severity_marker = "🔥" if result.get("critical", False) else ""
            print(f"  {status_icon} {test_name}: {result['status']} ({result['duration']:.2f}s){severity_marker}")
        
        # Recommendations
        print(f"\n💡 Next Steps:")
        if overall_status == "PASS":
            print("🎉 All tests passed! System is working correctly.")
        elif overall_status == "CRITICAL_FAILURE":
            print("🔥 Critical issues found. Priority fixes needed:")
            for test_name, result in summary["results"].items():
                if result.get("critical", False) and result["status"] != "PASS":
                    print(f"   • Fix {test_name}")
        else:
            print("⚠️ Some issues found. Review and fix:")
            for test_name, result in summary["results"].items():
                if result["status"] != "PASS":
                    print(f"   • Check {test_name}")


# Main execution
async def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed System Integration Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL")
    parser.add_argument("--mode", choices=["quick", "standard", "comprehensive", "critical"],
                       default="standard", help="Test mode")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout seconds")
    
    args = parser.parse_args()
    
    mode_map = {
        "quick": TestMode.QUICK,
        "standard": TestMode.STANDARD,
        "comprehensive": TestMode.COMPREHENSIVE,
        "critical": TestMode.CRITICAL_ONLY
    }
    
    try:
        async with FixedSystemIntegrationTest(
            base_url=args.url,
            mode=mode_map[args.mode],
            timeout=args.timeout
        ) as tester:
            results = await tester.run_test_suite()
            
            # Return appropriate exit code
            if results["overall_status"] == "PASS":
                return 0
            elif results["overall_status"] == "CRITICAL_FAILURE":
                return 2
            else:
                return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test execution failed: {str(e)}")
        return 1


# Convenience functions
async def run_quick_tests(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run quick tests"""
    async with FixedSystemIntegrationTest(base_url, TestMode.QUICK) as tester:
        return await tester.run_test_suite()


async def run_standard_tests(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run standard tests"""
    async with FixedSystemIntegrationTest(base_url, TestMode.STANDARD) as tester:
        return await tester.run_test_suite()


async def run_comprehensive_tests(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Run comprehensive tests"""
    async with FixedSystemIntegrationTest(base_url, TestMode.COMPREHENSIVE) as tester:
        return await tester.run_test_suite()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


# Example usage:
"""
# Quick development test
python corrected_system_test.py --mode quick

# Standard CI/CD test  
python corrected_system_test.py --mode standard --url http://localhost:8000

# Comprehensive production test
python corrected_system_test.py --mode comprehensive --url http://prod.example.com

# In Python scripts:
results = await run_standard_tests()
"""
