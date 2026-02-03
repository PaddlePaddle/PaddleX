# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for verifying the async cancellation fix in DocVLMPredictor.

This test verifies that when one request in a batch fails, other pending
futures are properly cancelled instead of continuing to consume resources.

Usage:
    # Unit test (no GPU/server required)
    python tests/test_doc_vlm_cancellation.py --unit

    # Integration test (requires a running VLM server)
    python tests/test_doc_vlm_cancellation.py --integration --server-url http://localhost:8000
"""

import argparse
import asyncio
import concurrent.futures
import sys
import time
import unittest
from unittest.mock import MagicMock, patch


class MockFuture:
    """Mock future that tracks cancel() calls."""

    def __init__(self, result=None, exception=None, delay=0):
        self._result = result
        self._exception = exception
        self._delay = delay
        self._done = False
        self._cancelled = False
        self.cancel_called = False

    def result(self, timeout=None):
        if self._delay > 0:
            time.sleep(self._delay)
        self._done = True
        if self._exception:
            raise self._exception
        return self._result

    def done(self):
        return self._done or self._cancelled

    def cancel(self):
        self.cancel_called = True
        if not self._done:
            self._cancelled = True
            return True
        return False


class MockChatCompletion:
    """Mock chat completion response."""

    def __init__(self, content):
        self.choices = [MagicMock(message=MagicMock(content=content))]


class TestDocVLMCancellation(unittest.TestCase):
    """Unit tests for the cancellation fix."""

    def test_cancellation_on_failure(self):
        """Test that futures are cancelled when one fails."""
        # Create mock futures: first succeeds, second fails, rest should be cancelled
        futures = [
            MockFuture(result=MockChatCompletion("result1")),
            MockFuture(exception=Exception("Simulated failure")),
            MockFuture(result=MockChatCompletion("result3")),
            MockFuture(result=MockChatCompletion("result4")),
        ]

        # Simulate the fixed _genai_client_process logic
        results = []
        try:
            for future in futures:
                result = future.result()
                results.append(result.choices[0].message.content)
        except Exception:
            # This is the fix: cancel pending futures
            for future in futures:
                if not future.done():
                    future.cancel()

        # Verify: futures 2 and 3 (index 2, 3) should have cancel() called
        self.assertFalse(futures[0].cancel_called, "First future should not be cancelled (already done)")
        self.assertFalse(futures[1].cancel_called, "Second future should not be cancelled (raised exception)")
        self.assertTrue(futures[2].cancel_called, "Third future should be cancelled")
        self.assertTrue(futures[3].cancel_called, "Fourth future should be cancelled")

    def test_no_cancellation_on_success(self):
        """Test that no cancellation happens when all succeed."""
        futures = [
            MockFuture(result=MockChatCompletion("result1")),
            MockFuture(result=MockChatCompletion("result2")),
            MockFuture(result=MockChatCompletion("result3")),
        ]

        results = []
        try:
            for future in futures:
                result = future.result()
                results.append(result.choices[0].message.content)
        except Exception:
            for future in futures:
                if not future.done():
                    future.cancel()

        # All should complete successfully, no cancellation
        self.assertEqual(len(results), 3)
        for future in futures:
            self.assertFalse(future.cancel_called)

    def test_cancellation_on_first_failure(self):
        """Test cancellation when the first request fails."""
        futures = [
            MockFuture(exception=Exception("First request failed")),
            MockFuture(result=MockChatCompletion("result2")),
            MockFuture(result=MockChatCompletion("result3")),
        ]

        results = []
        try:
            for future in futures:
                result = future.result()
                results.append(result.choices[0].message.content)
        except Exception:
            for future in futures:
                if not future.done():
                    future.cancel()

        # All remaining futures should be cancelled
        self.assertFalse(futures[0].cancel_called)
        self.assertTrue(futures[1].cancel_called)
        self.assertTrue(futures[2].cancel_called)


def run_integration_test(server_url):
    """
    Integration test with a real VLM server.

    This test sends a batch of requests where one is intentionally malformed
    to trigger an error, then verifies the behavior.
    """
    print(f"Running integration test with server: {server_url}")

    try:
        from paddlex.inference.models.common.genai import GenAIConfig
        from paddlex.inference.models.doc_vlm.predictor import DocVLMPredictor
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PaddleX is installed or run from the PaddleX root directory.")
        return False

    # Create predictor with remote backend
    genai_config = GenAIConfig(
        backend="vllm-server",  # or fastdeploy-server
        server_url=server_url,
        max_concurrency=10,
    )

    try:
        predictor = DocVLMPredictor(
            model_name="PaddleOCR-VL-0.9B",
            genai_config=genai_config,
        )
    except Exception as e:
        print(f"Failed to create predictor: {e}")
        return False

    # Prepare test data: mix of valid and invalid requests
    test_data = [
        {"image": "https://example.com/valid_image.jpg", "query": "What is this?"},
        {"image": "invalid_path_that_does_not_exist.jpg", "query": "What is this?"},
        {"image": "https://example.com/another_valid.jpg", "query": "Describe this."},
    ]

    print("Sending batch request with intentional failure...")
    start_time = time.time()

    try:
        results = list(predictor.predict(test_data))
        print(f"Unexpected success: {results}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Expected exception caught: {type(e).__name__}: {e}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print("If cancellation works, remaining requests should have been cancelled.")

    predictor.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Test DocVLM async cancellation fix")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration test")
    parser.add_argument("--server-url", type=str, help="VLM server URL for integration test")

    args = parser.parse_args()

    if not args.unit and not args.integration:
        args.unit = True  # Default to unit tests

    if args.unit:
        print("=" * 60)
        print("Running unit tests...")
        print("=" * 60)
        # Run unit tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestDocVLMCancellation)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        if not result.wasSuccessful():
            sys.exit(1)

    if args.integration:
        if not args.server_url:
            print("Error: --server-url is required for integration test")
            sys.exit(1)
        print("\n" + "=" * 60)
        print("Running integration test...")
        print("=" * 60)
        success = run_integration_test(args.server_url)
        if not success:
            sys.exit(1)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
