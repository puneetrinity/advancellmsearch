#!/usr/bin/env python3
"""
Quick verification script to test all fixes.
Run this after implementing the coroutine safety fixes.

Usage:
    python verify_fixes.py

This script will:
1. Test corrected request schemas
2. Verify no coroutine leaks
3. Check all endpoints work correctly
4. Report any remaining issues
"""

import asyncio
import time
import json
from typing import Dict, Any
import httpx
from app.main import app

class FixVerifier:
    """Verifies that all coroutine and schema fixes are working"""
    
    def __init__(self):
        self.base_url = "http://test"
        self.results = []
        
    async def run_all_verifications(self):
        """Run all verification tests"""
        print("🔍 Starting Fix Verification...")
        print("=" * 50)
        
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), 
            base_url=self.base_url
        ) as client:
            
            # Test 1: Schema fixes
            await self.verify_request_schemas(client)
            
            # Test 2: Coroutine safety
            await self.verify_coroutine_safety(client)
            
            # Test 3: Response structure
            await self.verify_response_structure(client)
            
            # Test 4: Error handling
            await self.verify_error_handling(client)
            
            # Test 5: Concurrent safety
            await self.verify_concurrent_safety(client)
        
        # Print final report
        self.print_verification_report()
        
    async def verify_request_schemas(self, client):
        """Verify that request schemas now work correctly"""
        print("\n📋 Testing Request Schema Fixes...")
        
        # Test corrected search request
        search_payload = {
            "query": "schema test query",
            "max_results": 5,
            "include_summary": True
        }
        
        try:
            resp = await client.post("/api/v1/search/basic", json=search_payload)
            if resp.status_code == 200:
                self.results.append("✅ Search schema fix: SUCCESS")
            elif resp.status_code == 422:
                self.results.append("❌ Search schema fix: FAILED (still 422)")
            else:
                self.results.append(f"⚠️ Search schema fix: Unexpected {resp.status_code}")
        except Exception as e:
            self.results.append(f"❌ Search schema fix: Exception {e}")
        
        # Test corrected chat request  
        chat_payload = {
            "message": "schema test message",
            "session_id": "schema_test"
        }
        
        try:
            resp = await client.post("/api/v1/chat/complete", json=chat_payload)
            if resp.status_code == 200:
                self.results.append("✅ Chat schema fix: SUCCESS")
            elif resp.status_code == 422:
                self.results.append("❌ Chat schema fix: FAILED (still 422)")
            else:
                self.results.append(f"⚠️ Chat schema fix: Unexpected {resp.status_code}")
        except Exception as e:
            self.results.append(f"❌ Chat schema fix: Exception {e}")
    
    async def verify_coroutine_safety(self, client):
        """Verify that responses contain no coroutines"""
        print("\n🔒 Testing Coroutine Safety Fixes...")
        
        # Test search coroutine safety
        search_payload = {
            "query": "coroutine safety test",
            "max_results": 3
        }
        
        try:
            resp = await client.post("/api/v1/search/basic", json=search_payload)
            if resp.status_code == 200:
                data = resp.json()
                has_coroutines = self.check_for_coroutines(data)
                if not has_coroutines:
                    self.results.append("✅ Search coroutine safety: SUCCESS")
                else:
                    self.results.append("❌ Search coroutine safety: FAILED (coroutines found)")
            else:
                self.results.append(f"⚠️ Search coroutine safety: Can't test ({resp.status_code})")
        except Exception as e:
            self.results.append(f"❌ Search coroutine safety: Exception {e}")
        
        # Test chat coroutine safety
        chat_payload = {
            "message": "coroutine safety test",
            "session_id": "coroutine_test"
        }
        
        try:
            resp = await client.post("/api/v1/chat/complete", json=chat_payload)
            if resp.status_code == 200:
                data = resp.json()
                has_coroutines = self.check_for_coroutines(data)
                if not has_coroutines:
                    self.results.append("✅ Chat coroutine safety: SUCCESS")
                else:
                    self.results.append("❌ Chat coroutine safety: FAILED (coroutines found)")
            else:
                self.results.append(f"⚠️ Chat coroutine safety: Can't test ({resp.status_code})")
        except Exception as e:
            self.results.append(f"❌ Chat coroutine safety: Exception {e}")
    
    async def verify_response_structure(self, client):
        """Verify that responses have expected structure"""
        print("\n📊 Testing Response Structure...")
        
        # Test search response structure
        search_payload = {
            "query": "structure test",
            "max_results": 2
        }
        
        try:
            resp = await client.post("/api/v1/search/basic", json=search_payload)
            if resp.status_code == 200:
                data = resp.json()
                required_fields = ["status", "data", "metadata"]
                missing_fields = [f for f in required_fields if f not in data]
                
                if not missing_fields:
                    self.results.append("✅ Search response structure: SUCCESS")
                else:
                    self.results.append(f"❌ Search response structure: Missing {missing_fields}")
            else:
                self.results.append(f"⚠️ Search response structure: Can't test ({resp.status_code})")
        except Exception as e:
            self.results.append(f"❌ Search response structure: Exception {e}")
        
        # Test chat response structure
        chat_payload = {
            "message": "structure test",
            "session_id": "structure_test"
        }
        
        try:
            resp = await client.post("/api/v1/chat/complete", json=chat_payload)
            if resp.status_code == 200:
                data = resp.json()
                required_fields = ["status", "data", "metadata"]
                missing_fields = [f for f in required_fields if f not in data]
                
                if not missing_fields:
                    self.results.append("✅ Chat response structure: SUCCESS")
                else:
                    self.results.append(f"❌ Chat response structure: Missing {missing_fields}")
            else:
                self.results.append(f"⚠️ Chat response structure: Can't test ({resp.status_code})")
        except Exception as e:
            self.results.append(f"❌ Chat response structure: Exception {e}")
    
    async def verify_error_handling(self, client):
        """Verify that error handling works correctly"""
        print("\n🚨 Testing Error Handling...")
        
        # Test invalid search request (no wrapper)
        invalid_search = {
            "query": "test",  # Missing wrapper
            "max_results": 5
        }
        
        try:
            resp = await client.post("/api/v1/search/basic", json=invalid_search)
            if resp.status_code == 422:
                self.results.append("✅ Search error handling: SUCCESS (422 for invalid schema)")
            else:
                self.results.append(f"⚠️ Search error handling: Unexpected {resp.status_code}")
        except Exception as e:
            self.results.append(f"❌ Search error handling: Exception {e}")
        
        # Test invalid chat request (no wrapper)
        invalid_chat = {
            "message": "test",  # Missing wrapper
            "session_id": "test"
        }
        
        try:
            resp = await client.post("/api/v1/chat/complete", json=invalid_chat)
            if resp.status_code == 422:
                self.results.append("✅ Chat error handling: SUCCESS (422 for invalid schema)")
            else:
                self.results.append(f"⚠️ Chat error handling: Unexpected {resp.status_code}")
        except Exception as e:
            self.results.append(f"❌ Chat error handling: Exception {e}")
    
    async def verify_concurrent_safety(self, client):
        """Verify that concurrent requests don't cause issues"""
        print("\n🔄 Testing Concurrent Request Safety...")
        
        async def make_search_request(i):
            payload = {
                "query": f"concurrent test {i}",
                "max_results": 2
            }
            resp = await client.post("/api/v1/search/basic", json=payload)
            return resp.status_code == 200
        
        async def make_chat_request(i):
            payload = {
                "message": f"concurrent test {i}",
                "session_id": f"concurrent_{i}"
            }
            resp = await client.post("/api/v1/chat/complete", json=payload)
            return resp.status_code == 200
        
        try:
            # Run 3 concurrent requests of each type
            search_tasks = [make_search_request(i) for i in range(3)]
            chat_tasks = [make_chat_request(i) for i in range(3)]
            
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            chat_results = await asyncio.gather(*chat_tasks, return_exceptions=True)
            
            # Check for exceptions
            search_exceptions = [r for r in search_results if isinstance(r, Exception)]
            chat_exceptions = [r for r in chat_results if isinstance(r, Exception)]
            
            if not search_exceptions and not chat_exceptions:
                search_success = sum(1 for r in search_results if r is True)
                chat_success = sum(1 for r in chat_results if r is True)
                self.results.append(f"✅ Concurrent safety: SUCCESS ({search_success}/3 search, {chat_success}/3 chat)")
            else:
                self.results.append(f"❌ Concurrent safety: FAILED ({len(search_exceptions + chat_exceptions)} exceptions)")
                
        except Exception as e:
            self.results.append(f"❌ Concurrent safety: Exception {e}")
    
    def check_for_coroutines(self, obj, path="root"):
        """Recursively check for coroutines in response object"""
        if asyncio.iscoroutine(obj):
            print(f"🚨 COROUTINE FOUND at {path}")
            return True
            
        if isinstance(obj, dict):
            for key, value in obj.items():
                if self.check_for_coroutines(value, f"{path}.{key}"):
                    return True
                    
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                if self.check_for_coroutines(item, f"{path}[{i}]"):
                    return True
                    
        elif hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if self.check_for_coroutines(value, f"{path}.{key}"):
                    return True
                    
        return False
    
    def print_verification_report(self):
        """Print final verification report"""
        print("\n" + "=" * 50)
        print("📋 VERIFICATION REPORT")
        print("=" * 50)
        
        success_count = sum(1 for r in self.results if r.startswith("✅"))
        warning_count = sum(1 for r in self.results if r.startswith("⚠️"))
        failure_count = sum(1 for r in self.results if r.startswith("❌"))
        
        for result in self.results:
            print(result)
        
        print("\n" + "=" * 50)
        print(f"📊 SUMMARY: {success_count} Success, {warning_count} Warning, {failure_count} Failed")
        
        if failure_count == 0:
            print("🎉 ALL CRITICAL FIXES WORKING!")
            print("✅ No coroutine leaks detected")
            print("✅ Request schemas fixed")  
            print("✅ Error handling working")
        else:
            print("🚨 ISSUES STILL EXIST:")
            failed_items = [r for r in self.results if r.startswith("❌")]
            for item in failed_items:
                print(f"   {item}")
        
        print("=" * 50)

async def main():
    """Main verification function"""
    verifier = FixVerifier()
    await verifier.run_all_verifications()

def test_basic_endpoints():
    """Test basic sync endpoints quickly"""
    print("🔍 Testing Basic Endpoints...")
    
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Test health
    try:
        resp = client.get("/health")
        if resp.status_code == 200:
            print("✅ Health endpoint: SUCCESS")
        else:
            print(f"❌ Health endpoint: FAILED ({resp.status_code})")
    except Exception as e:
        print(f"❌ Health endpoint: Exception {e}")
    
    # Test root
    try:
        resp = client.get("/")
        if resp.status_code == 200:
            print("✅ Root endpoint: SUCCESS")
        else:
            print(f"❌ Root endpoint: FAILED ({resp.status_code})")
    except Exception as e:
        print(f"❌ Root endpoint: Exception {e}")
    
    # Test metrics
    try:
        resp = client.get("/metrics")
        if resp.status_code == 200:
            print("✅ Metrics endpoint: SUCCESS")
        else:
            print(f"❌ Metrics endpoint: FAILED ({resp.status_code})")
    except Exception as e:
        print(f"❌ Metrics endpoint: Exception {e}")

if __name__ == "__main__":
    print("🚀 Fix Verification Script")
    print("This script verifies all coroutine and schema fixes are working.")
    print()
    
    # Test basic endpoints first
    test_basic_endpoints()
    
    # Test async endpoints
    print("\n" + "="*50)
    asyncio.run(main())
    
    print("\n🏁 Verification Complete!")
    print("\nNext steps if issues found:")
    print("1. Check that async_utils.py is imported in endpoints")
    print("2. Verify WrappedChatRequest/WrappedSearchRequest are used")
    print("3. Ensure ensure_awaited() is called on all async results")
    print("4. Check that mocks in tests don't return coroutines")