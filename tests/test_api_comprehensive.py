#!/usr/bin/env python3
"""
Comprehensive API Test Suite for Enso Atlas

Tests all endpoints and functionality against the spec in technical_specification.md.
Run against DGX Spark backend at http://100.111.126.23:8003

Usage:
    python test_api_comprehensive.py [--base-url URL]
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, List, Tuple
import urllib.request
import urllib.error

# Test configuration
DEFAULT_BASE_URL = "http://100.111.126.23:8003"
TIMEOUT = 30

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"

class EnsoAtlasTestSuite:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.results: List[TestResult] = []
        self.slides: List[str] = []
    
    def _request(self, method: str, endpoint: str, data: Dict = None) -> Tuple[int, Any]:
        """Make HTTP request and return status code and response."""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        if data:
            data = json.dumps(data).encode("utf-8")
        
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                try:
                    return resp.status, json.loads(body)
                except json.JSONDecodeError:
                    return resp.status, body
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                return e.code, json.loads(body)
            except:
                return e.code, body
        except Exception as e:
            return 0, str(e)
    
    def add_result(self, result: TestResult):
        self.results.append(result)
        print(result)
    
    # ==================== Health & Status Tests ====================
    
    def test_health_endpoint(self):
        """Test GET /health returns healthy status."""
        status, resp = self._request("GET", "/health")
        
        if status != 200:
            return TestResult("Health Check", False, f"Status {status}", resp)
        
        checks = []
        if resp.get("status") != "healthy":
            checks.append("status not healthy")
        if not resp.get("model_loaded"):
            checks.append("model not loaded")
        if not resp.get("cuda_available"):
            checks.append("CUDA not available")
        if resp.get("slides_available", 0) == 0:
            checks.append("no slides available")
        
        if checks:
            return TestResult("Health Check", False, "; ".join(checks), resp)
        
        return TestResult("Health Check", True, 
            f"Healthy, CUDA={resp.get('cuda_available')}, slides={resp.get('slides_available')}")
    
    def test_api_docs(self):
        """Test GET /api/docs returns documentation."""
        status, resp = self._request("GET", "/api/docs")
        
        if status != 200:
            return TestResult("API Docs", False, f"Status {status}")
        
        return TestResult("API Docs", True, "Documentation accessible")
    
    # ==================== Slide Management Tests ====================
    
    def test_list_slides(self):
        """Test GET /api/slides returns list of available slides."""
        status, resp = self._request("GET", "/api/slides")
        
        if status != 200:
            return TestResult("List Slides", False, f"Status {status}", resp)
        
        if not isinstance(resp, list):
            return TestResult("List Slides", False, "Response not a list", resp)
        
        if len(resp) == 0:
            return TestResult("List Slides", False, "No slides available")
        
        # Store slides for later tests
        self.slides = [s.get("slide_id") for s in resp if s.get("slide_id")]
        
        # Validate slide structure
        required_fields = ["slide_id", "has_embeddings"]
        for slide in resp:
            for field in required_fields:
                if field not in slide:
                    return TestResult("List Slides", False, f"Missing field: {field}", slide)
        
        return TestResult("List Slides", True, 
            f"Found {len(resp)} slides with embeddings")
    
    # ==================== Analysis Tests ====================
    
    def test_analyze_slide_basic(self):
        """Test POST /api/analyze with basic request."""
        if not self.slides:
            return TestResult("Analyze Slide (Basic)", False, "No slides available for testing")
        
        slide_id = self.slides[0]
        status, resp = self._request("POST", "/api/analyze", {"slide_id": slide_id})
        
        if status != 200:
            return TestResult("Analyze Slide (Basic)", False, f"Status {status}", resp)
        
        required_fields = ["slide_id", "prediction", "score", "confidence", 
                          "patches_analyzed", "top_evidence", "similar_cases"]
        
        for field in required_fields:
            if field not in resp:
                return TestResult("Analyze Slide (Basic)", False, f"Missing field: {field}", resp)
        
        # Validate prediction is valid
        if resp["prediction"] not in ["RESPONDER", "NON-RESPONDER"]:
            return TestResult("Analyze Slide (Basic)", False, 
                f"Invalid prediction: {resp['prediction']}")
        
        # Validate score is between 0 and 1
        if not (0 <= resp["score"] <= 1):
            return TestResult("Analyze Slide (Basic)", False, 
                f"Score out of range: {resp['score']}")
        
        return TestResult("Analyze Slide (Basic)", True,
            f"Slide {slide_id}: {resp['prediction']} (score={resp['score']:.3f}, "
            f"patches={resp['patches_analyzed']}, evidence={len(resp['top_evidence'])})")
    
    def test_analyze_all_slides(self):
        """Test POST /api/analyze for all available slides."""
        if not self.slides:
            return TestResult("Analyze All Slides", False, "No slides available")
        
        failures = []
        for slide_id in self.slides:
            status, resp = self._request("POST", "/api/analyze", {"slide_id": slide_id})
            if status != 200:
                failures.append(f"{slide_id}: status {status}")
            elif "prediction" not in resp:
                failures.append(f"{slide_id}: no prediction")
        
        if failures:
            return TestResult("Analyze All Slides", False, 
                f"{len(failures)}/{len(self.slides)} failed", failures)
        
        return TestResult("Analyze All Slides", True, 
            f"All {len(self.slides)} slides analyzed successfully")
    
    def test_analyze_invalid_slide(self):
        """Test POST /api/analyze with non-existent slide returns 404."""
        status, resp = self._request("POST", "/api/analyze", {"slide_id": "nonexistent_slide_xyz"})
        
        if status != 404:
            return TestResult("Analyze Invalid Slide", False, 
                f"Expected 404, got {status}", resp)
        
        return TestResult("Analyze Invalid Slide", True, "Correctly returns 404 for invalid slide")
    
    # ==================== Evidence Tests ====================
    
    def test_evidence_patches(self):
        """Test that evidence patches have valid structure."""
        if not self.slides:
            return TestResult("Evidence Patches", False, "No slides available")
        
        slide_id = self.slides[0]
        status, resp = self._request("POST", "/api/analyze", {"slide_id": slide_id})
        
        if status != 200:
            return TestResult("Evidence Patches", False, f"Analyze failed: {status}")
        
        evidence = resp.get("top_evidence", [])
        if not evidence:
            return TestResult("Evidence Patches", False, "No evidence patches returned")
        
        required_fields = ["rank", "patch_index", "attention_weight"]
        for patch in evidence:
            for field in required_fields:
                if field not in patch:
                    return TestResult("Evidence Patches", False, 
                        f"Missing field in evidence: {field}", patch)
            
            # Validate attention weight is reasonable
            weight = patch.get("attention_weight", 0)
            if not (0 <= weight <= 1):
                return TestResult("Evidence Patches", False, 
                    f"Attention weight out of range: {weight}")
        
        return TestResult("Evidence Patches", True, 
            f"{len(evidence)} patches with valid attention weights")
    
    def test_similar_cases(self):
        """Test that similar cases are returned."""
        if not self.slides:
            return TestResult("Similar Cases", False, "No slides available")
        
        slide_id = self.slides[0]
        status, resp = self._request("POST", "/api/analyze", {"slide_id": slide_id})
        
        if status != 200:
            return TestResult("Similar Cases", False, f"Analyze failed: {status}")
        
        similar = resp.get("similar_cases", [])
        if not similar:
            return TestResult("Similar Cases", False, "No similar cases returned")
        
        # Check that similar cases don't include the query slide
        for case in similar:
            if case.get("slide_id") == slide_id:
                return TestResult("Similar Cases", False, 
                    "Query slide appears in similar cases")
        
        return TestResult("Similar Cases", True, 
            f"{len(similar)} similar cases returned (FAISS working)")
    
    # ==================== Heatmap Tests ====================
    
    def test_heatmap_generation(self):
        """Test GET /api/heatmap/{slide_id} returns valid image."""
        if not self.slides:
            return TestResult("Heatmap Generation", False, "No slides available")
        
        slide_id = self.slides[0]
        url = f"{self.base_url}/api/heatmap/{slide_id}"
        
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "")
                data = resp.read()
                
                if "image" not in content_type and not data.startswith(b"\x89PNG"):
                    return TestResult("Heatmap Generation", False, 
                        f"Not an image: {content_type}")
                
                return TestResult("Heatmap Generation", True, 
                    f"Heatmap generated ({len(data)} bytes)")
        except Exception as e:
            return TestResult("Heatmap Generation", False, str(e))
    
    def test_heatmap_all_slides(self):
        """Test heatmap generation for all slides."""
        if not self.slides:
            return TestResult("Heatmap All Slides", False, "No slides available")
        
        failures = []
        for slide_id in self.slides:
            url = f"{self.base_url}/api/heatmap/{slide_id}"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                    data = resp.read()
                    if len(data) < 100:
                        failures.append(f"{slide_id}: too small ({len(data)} bytes)")
            except Exception as e:
                failures.append(f"{slide_id}: {e}")
        
        if failures:
            return TestResult("Heatmap All Slides", False, 
                f"{len(failures)}/{len(self.slides)} failed", failures)
        
        return TestResult("Heatmap All Slides", True, 
            f"All {len(self.slides)} heatmaps generated")
    
    # ==================== Report Tests ====================
    
    def test_report_generation(self):
        """Test POST /api/report generates structured report."""
        if not self.slides:
            return TestResult("Report Generation", False, "No slides available")
        
        slide_id = self.slides[0]
        status, resp = self._request("POST", "/api/report", {"slide_id": slide_id})
        
        if status != 200:
            return TestResult("Report Generation", False, f"Status {status}", resp)
        
        required_fields = ["slide_id", "report_json", "summary_text"]
        for field in required_fields:
            if field not in resp:
                return TestResult("Report Generation", False, 
                    f"Missing field: {field}", resp)
        
        # Validate report_json structure
        report_json = resp.get("report_json", {})
        json_required = ["case_id", "task", "model_output", "limitations", "safety_statement"]
        for field in json_required:
            if field not in report_json:
                return TestResult("Report Generation", False, 
                    f"Missing JSON field: {field}", report_json)
        
        # Validate summary is not empty
        summary = resp.get("summary_text", "")
        if len(summary) < 50:
            return TestResult("Report Generation", False, 
                "Summary too short", summary)
        
        return TestResult("Report Generation", True, 
            f"Report generated ({len(summary)} chars)")
    
    # ==================== Embedding Tests ====================
    
    def test_embed_status(self):
        """Test GET /api/embed/status returns embedder status."""
        status, resp = self._request("GET", "/api/embed/status")
        
        if status != 200:
            return TestResult("Embedder Status", False, f"Status {status}", resp)
        
        return TestResult("Embedder Status", True, 
            f"Embedder status: {resp.get('status', 'unknown')}")
    
    # ==================== Performance Tests ====================
    
    def test_analyze_performance(self):
        """Test analysis response time."""
        if not self.slides:
            return TestResult("Analyze Performance", False, "No slides available")
        
        slide_id = self.slides[0]
        
        start = time.time()
        status, resp = self._request("POST", "/api/analyze", {"slide_id": slide_id})
        elapsed = time.time() - start
        
        if status != 200:
            return TestResult("Analyze Performance", False, f"Status {status}")
        
        # Should complete in under 5 seconds for cached embeddings
        if elapsed > 5:
            return TestResult("Analyze Performance", False, 
                f"Too slow: {elapsed:.2f}s (expected <5s)")
        
        return TestResult("Analyze Performance", True, 
            f"Analysis completed in {elapsed:.2f}s")
    
    # ==================== Run All Tests ====================
    
    def run_all(self) -> Tuple[int, int]:
        """Run all tests and return (passed, failed) counts."""
        print("=" * 60)
        print("ENSO ATLAS COMPREHENSIVE API TEST SUITE")
        print(f"Target: {self.base_url}")
        print("=" * 60)
        print()
        
        # Health & Status
        print("--- Health & Status ---")
        self.add_result(self.test_health_endpoint())
        self.add_result(self.test_api_docs())
        print()
        
        # Slide Management
        print("--- Slide Management ---")
        self.add_result(self.test_list_slides())
        print()
        
        # Analysis
        print("--- Analysis ---")
        self.add_result(self.test_analyze_slide_basic())
        self.add_result(self.test_analyze_all_slides())
        self.add_result(self.test_analyze_invalid_slide())
        print()
        
        # Evidence
        print("--- Evidence ---")
        self.add_result(self.test_evidence_patches())
        self.add_result(self.test_similar_cases())
        print()
        
        # Heatmap
        print("--- Heatmap ---")
        self.add_result(self.test_heatmap_generation())
        self.add_result(self.test_heatmap_all_slides())
        print()
        
        # Report
        print("--- Report ---")
        self.add_result(self.test_report_generation())
        print()
        
        # Embedding
        print("--- Embedding ---")
        self.add_result(self.test_embed_status())
        print()
        
        # Performance
        print("--- Performance ---")
        self.add_result(self.test_analyze_performance())
        print()
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        
        print("=" * 60)
        print(f"RESULTS: {passed} passed, {failed} failed")
        print("=" * 60)
        
        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
                    if r.details:
                        print(f"    Details: {r.details}")
        
        return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Comprehensive API Test Suite for Enso Atlas")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL of the API")
    args = parser.parse_args()
    
    suite = EnsoAtlasTestSuite(args.base_url)
    passed, failed = suite.run_all()
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
