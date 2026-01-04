"""Backoff strategies for retry policies.

Provides pluggable delay calculation for retry attempts:
- ExponentialBackoff: Exponential growth with optional jitter
- LinearBackoff: Linear growth with cap
- ConstantBackoff: Fixed delay
- DecorrelatedJitter: AWS-style decorrelated jitter (optimal for many retriers)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Backoff(Protocol):
    """Protocol for backoff delay calculation.
    
    Implementations compute the delay before the next retry attempt.
    Attempt numbers are 0-indexed (first retry = attempt 0).
    """
    
    def delay(self, attempt: int) -> float:
        """Calculate delay in seconds for given attempt number.
        
        Args:
            attempt: 0-indexed retry attempt number
            
        Returns:
            Delay in seconds before next retry
        """
        ...


@dataclass(frozen=True, slots=True)
class ExponentialBackoff:
    """Exponential backoff with optional jitter.
    
    Delay = min(base * (multiplier ^ attempt), max_delay) * jitter
    
    Jitter prevents thundering herd by randomizing delays.
    
    Attributes:
        base: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 30.0)
        multiplier: Exponential growth factor (default: 2.0)
        jitter: Add randomization 0.5-1.5x (default: True)
    """
    
    base: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True
    
    def delay(self, attempt: int) -> float:
        d = min(self.base * (self.multiplier ** attempt), self.max_delay)
        return d * (0.5 + random.random()) if self.jitter else d


@dataclass(frozen=True, slots=True)
class LinearBackoff:
    """Linear backoff with cap.
    
    Delay = min(base + (increment * attempt), max_delay)
    
    Attributes:
        base: Initial delay in seconds (default: 1.0)
        increment: Additional delay per attempt (default: 1.0)
        max_delay: Maximum delay cap (default: 30.0)
    """
    
    base: float = 1.0
    increment: float = 1.0
    max_delay: float = 30.0
    
    def delay(self, attempt: int) -> float:
        return min(self.base + (self.increment * attempt), self.max_delay)


@dataclass(frozen=True, slots=True)
class ConstantBackoff:
    """Fixed delay between retries.
    
    Simple strategy for rate-limited APIs with known cooldown.
    
    Attributes:
        delay_seconds: Fixed delay in seconds (default: 1.0)
    """
    
    delay_seconds: float = 1.0
    
    def delay(self, attempt: int) -> float:
        return self.delay_seconds


@dataclass(frozen=True, slots=True)
class DecorrelatedJitter:
    """AWS-style decorrelated jitter backoff.
    
    Optimal for distributed systems with many retriers.
    Each delay is independent, bounded between base and previous*3.
    
    Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    
    Attributes:
        base: Minimum delay in seconds (default: 1.0)
        max_delay: Maximum delay cap (default: 30.0)
    """
    
    base: float = 1.0
    max_delay: float = 30.0
    _prev: float = 0.0  # Not actually used in frozen, computed fresh
    
    def delay(self, attempt: int) -> float:
        # Compute chain from start for determinism given attempt
        prev = self.base
        for _ in range(attempt):
            prev = min(self.max_delay, random.uniform(self.base, prev * 3))
        return prev
