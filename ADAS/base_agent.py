"""
Base Agent and Shared Utilities for ADAS System

This module contains the base agent class and shared utilities used by all agents.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    print("Warning: Cerebras SDK not available")
    Cerebras = None


@dataclass
class TaskAgentVariant:
    """Configuration for a TaskAgent variant"""

    variant_id: str
    generation: int
    parent_id: Optional[str]
    mutation_type: str
    parameters: Dict[str, Any]
    performance_score: float = 0.0
    evaluation_count: int = 0
    created_at: str = ""


@dataclass
class EvaluationResult:
    """Result of evaluating a TaskAgent variant"""

    variant_id: str
    score: float
    reasoning: str
    validation_record_id: str
    timestamp: str


# Global rate limiter instance
_global_rate_limiter = None


class GlobalRateLimiter:
    """Global rate limiter shared across all agents to prevent concurrent API overload"""

    def __init__(self, requests_per_minute: int = 8):  # Conservative limit
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self.last_request_time = 0.0
        self.consecutive_errors = 0
        self.success_count = 0
        self.lock = asyncio.Lock()  # Prevent concurrent access

    async def wait_if_needed(self, agent_name: str = "Unknown"):
        """Wait if necessary to respect rate limits with adaptive backoff"""
        async with self.lock:  # Ensure only one API call at a time
            current_time = time.time()
            time_since_last = current_time - self.last_request_time

            # Adaptive backoff based on consecutive errors
            backoff_multiplier = 1.0 + (
                self.consecutive_errors * 1.0
            )  # More aggressive backoff
            adjusted_interval = self.min_interval * backoff_multiplier

            if time_since_last < adjusted_interval:
                wait_time = adjusted_interval - time_since_last
                if self.consecutive_errors > 0:
                    print(
                        f"⏳ {agent_name}: Global rate limiting with backoff {wait_time:.1f}s (errors: {self.consecutive_errors})"
                    )
                else:
                    print(f"⏳ {agent_name}: Global rate limiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

            self.last_request_time = time.time()

    def record_success(self):
        """Record successful API call"""
        self.consecutive_errors = 0
        self.success_count += 1

    def record_error(self):
        """Record failed API call"""
        self.consecutive_errors += 1


def get_global_rate_limiter() -> GlobalRateLimiter:
    """Get or create global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GlobalRateLimiter(
            requests_per_minute=8
        )  # Very conservative
    return _global_rate_limiter


class BaseAgent:
    """Base class for all ADAS agents with shared functionality"""

    def __init__(self, cerebras_api_key: str, agent_name: str = "Unknown"):
        self.cerebras_client = None
        if Cerebras and cerebras_api_key:
            # Disable TCP warming to reduce API calls and avoid rate limiting
            # Reduce max_retries since we have our own rate limiter
            self.cerebras_client = Cerebras(
                api_key=cerebras_api_key,
                warm_tcp_connection=False,
                max_retries=1,  # Reduce from default 2 to 1
            )

        self.agent_name = agent_name
        self.rate_limiter = get_global_rate_limiter()

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process method")
