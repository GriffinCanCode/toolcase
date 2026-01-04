"""Backoff strategies for batch retry policies.

Provides pluggable delay calculation for batch-level retries.
For tool-level retries, stamina handles backoff internally with exponential + jitter.

These are primarily used by BatchRetryPolicy for batch-level retry coordination.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import reduce
from typing import Protocol, runtime_checkable


@runtime_checkable
class Backoff(Protocol):
    """Protocol for backoff delay calculation.
    
    Implementations compute delay before next retry attempt.
    Attempt numbers are 0-indexed (first retry = attempt 0).
    """
    
    def delay(self, attempt: int) -> float:
        """Calculate delay in seconds for given attempt number."""
        ...


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """Exponential backoff with optional jitter.
    
    Delay = min(base * (multiplier ^ attempt), max_delay) * jitter
    Jitter prevents thundering herd by randomizing delays.
    """
    
    base: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True
    
    def delay(self, attempt: int) -> float:
        d = min(self.base * self.multiplier ** attempt, self.max_delay)
        return d * (0.5 + random.random()) if self.jitter else d


@dataclass(frozen=True, slots=True)
class LinearBackoff:
    """Linear backoff with cap. Delay = min(base + (increment * attempt), max_delay)"""
    
    base: float = 1.0
    increment: float = 1.0
    max_delay: float = 30.0
    
    def delay(self, attempt: int) -> float:
        return min(self.base + self.increment * attempt, self.max_delay)


@dataclass(frozen=True, slots=True)
class ConstantBackoff:
    """Fixed delay between retries. Simple strategy for rate-limited APIs."""
    
    delay_seconds: float = 1.0
    
    def delay(self, attempt: int) -> float:
        return self.delay_seconds


@dataclass(frozen=True, slots=True)
class DecorrelatedJitter:
    """AWS-style decorrelated jitter backoff.
    
    Optimal for distributed systems with many retriers.
    Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """
    
    base: float = 1.0
    max_delay: float = 30.0
    
    def delay(self, attempt: int) -> float:
        return reduce(
            lambda prev, _: min(self.max_delay, random.uniform(self.base, prev * 3)),
            range(attempt),
            self.base,
        )
