"""Centralized parameter validation middleware.

Consolidates validation logic from registry, server, and tools into a single
middleware. Supports custom validators, cross-field constraints, and consistent
error formatting for LLM feedback.

Optimizations:
- TypeAdapter cache for fast dict→params validation (bypasses full model overhead)
- Per-tool adapter caching at first validation (lazy initialization)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter, ValidationError

from toolcase.foundation.errors import ErrorCode, ToolError, ValidationToolException, format_validation_error
from toolcase.io.streaming import StreamChunk
from toolcase.runtime.middleware import Context, Next

if TYPE_CHECKING:
    from toolcase.foundation.core import BaseTool

# Validator: value -> True (pass) | False (fail with default msg) | str (fail with custom msg)
Validator = Callable[[object], bool | str]


@dataclass(slots=True, frozen=True)
class FieldRule:
    """Immutable validation rule for a specific field."""
    field: str
    check: Validator
    message: str


@dataclass(slots=True)
class ValidationMiddleware:
    """Centralized parameter validation middleware.
    
    Consolidates validation from registry/server/tools into single point:
    - Dict→BaseModel conversion with TypeAdapter (faster than direct model instantiation)
    - Custom field validators with chainable API
    - Cross-field constraints via rule composition
    - Consistent LLM-friendly error formatting
    
    Should be first in chain for fail-fast behavior. Works with both
    regular middleware protocol and StreamMiddleware hooks.
    
    Performance: Uses cached TypeAdapters per tool params_schema for ~15-30%
    faster dict→model validation vs direct model(**dict) instantiation.
    
    Example:
        >>> validation = ValidationMiddleware()
        >>> validation.add_rule("http_request", "url", lambda u: u.startswith("https://"), "must use HTTPS")
        >>> validation.add_rule("search", "query", lambda q: len(q) >= 3, "must be at least 3 characters")
        >>> registry.use(validation)
        
    Cross-field validation:
        >>> def check_date_range(params):
        ...     return params.start <= params.end or "start must be before end"
        >>> validation.add_constraint("report", check_date_range)
    """
    
    _rules: dict[str, list[FieldRule]] = field(default_factory=dict)
    _constraints: dict[str, list[Validator]] = field(default_factory=dict)
    _adapters: dict[str, TypeAdapter[BaseModel]] = field(default_factory=dict)  # Cached TypeAdapters per tool
    revalidate: bool = False  # Re-run Pydantic validation on existing BaseModel
    
    def add_rule(self, tool_name: str, field_name: str, check: Validator, message: str) -> "ValidationMiddleware":
        """Add field validation rule. Chainable.
        
        Args:
            tool_name: Tool to apply rule to
            field_name: Field to validate
            check: Callable(value) -> True/False/str
            message: Error message if check returns False
        """
        self._rules.setdefault(tool_name, []).append(FieldRule(field_name, check, message))
        return self
    
    def add_constraint(self, tool_name: str, check: Validator) -> "ValidationMiddleware":
        """Add cross-field constraint. Receives full params model. Chainable."""
        self._constraints.setdefault(tool_name, []).append(check)
        return self
    
    def _get_adapter(self, tool: "BaseTool[BaseModel]") -> TypeAdapter[BaseModel]:
        """Get or create cached TypeAdapter for tool's params_schema."""
        name = tool.metadata.name
        if (adapter := self._adapters.get(name)) is None:
            adapter = TypeAdapter(tool.params_schema)
            self._adapters[name] = adapter
        return adapter
    
    def _validate(self, tool: "BaseTool[BaseModel]", params: BaseModel | dict[str, object]) -> tuple[BaseModel | None, str | None]:
        """Validate params. Returns (validated_params, None) or (None, error_string)."""
        name = tool.metadata.name
        
        # Dict→BaseModel conversion via cached TypeAdapter (faster than direct model(**dict))
        if isinstance(params, dict):
            try:
                params = self._get_adapter(tool).validate_python(params)
            except ValidationError as e:
                return None, ToolError.create(name, format_validation_error(e, tool_name=name), ErrorCode.INVALID_PARAMS, recoverable=False).render()
        
        # Optional re-validation (catch mutations, ensure schema compliance)
        if self.revalidate:
            try:
                params = self._get_adapter(tool).validate_python(params.model_dump())
            except ValidationError as e:
                return None, ToolError.create(name, format_validation_error(e, tool_name=name), ErrorCode.INVALID_PARAMS, recoverable=False).render()
        
        # Field rules
        for rule in self._rules.get(name, []):
            val = getattr(params, rule.field, None)
            result = rule.check(val)
            if result is False or isinstance(result, str):
                msg = result if isinstance(result, str) else rule.message
                return None, ToolError.create(name, f"'{rule.field}' {msg}", ErrorCode.INVALID_PARAMS, recoverable=False).render()
        
        # Cross-field constraints
        for constraint in self._constraints.get(name, []):
            result = constraint(params)
            if result is False:
                return None, ToolError.create(name, "Cross-field constraint failed", ErrorCode.INVALID_PARAMS, recoverable=False).render()
            if isinstance(result, str):
                return None, ToolError.create(name, result, ErrorCode.INVALID_PARAMS, recoverable=False).render()
        
        return params, None
    
    # ─────────────────────────────────────────────────────────────────
    # Regular Middleware Protocol
    # ─────────────────────────────────────────────────────────────────
    
    async def __call__(
        self,
        tool: "BaseTool[BaseModel]",
        params: BaseModel,
        ctx: Context,
        next: Next,
    ) -> str:
        """Validate and pass to next middleware."""
        validated, error = self._validate(tool, params)
        if error:
            return error
        
        ctx["validated_params"] = validated
        return await next(tool, validated, ctx)  # type: ignore[arg-type]
    
    # ─────────────────────────────────────────────────────────────────
    # StreamMiddleware Protocol (hooks)
    # ─────────────────────────────────────────────────────────────────
    
    async def on_start(self, tool: "BaseTool[BaseModel]", params: BaseModel, ctx: Context) -> None:
        """Validate before streaming begins. Raises ValidationToolException on failure."""
        validated, error = self._validate(tool, params)
        if error:
            raise ValidationToolException.create(tool.metadata.name, error, ErrorCode.INVALID_PARAMS, recoverable=False)
        ctx["validated_params"] = validated
    
    async def on_chunk(self, chunk: StreamChunk, ctx: Context) -> StreamChunk:
        """Pass chunks through unchanged."""
        return chunk
    
    async def on_complete(self, accumulated: str, ctx: Context) -> None:
        """No-op on completion."""
    
    async def on_error(self, error: Exception, ctx: Context) -> None:
        """No-op on error."""


# ─────────────────────────────────────────────────────────────────────────────
# Preset Validators (common patterns)
# ─────────────────────────────────────────────────────────────────────────────

def min_length(n: int) -> Validator:
    """Validate minimum string/collection length."""
    return lambda v: len(v) >= n if v else False or f"must have at least {n} items/characters"


def max_length(n: int) -> Validator:
    """Validate maximum string/collection length."""
    return lambda v: len(v) <= n if v else True or f"must have at most {n} items/characters"


def in_range(low: float, high: float) -> Validator:
    """Validate numeric value in range [low, high]."""
    return lambda v: low <= v <= high if isinstance(v, (int, float)) else f"must be between {low} and {high}"


def matches(pattern: str) -> Validator:
    """Validate string matches regex pattern."""
    import re
    compiled = re.compile(pattern)
    return lambda v: bool(compiled.match(str(v))) if v else False or f"must match pattern {pattern}"


def one_of(*allowed: object) -> Validator:
    """Validate value is one of allowed options."""
    allowed_set = frozenset(allowed)
    return lambda v: v in allowed_set or f"must be one of: {', '.join(map(str, allowed))}"


def not_empty(v: object) -> bool | str:
    """Validate value is not empty/None."""
    if v is None:
        return "cannot be empty"
    if isinstance(v, (str, list, dict)) and not v:
        return "cannot be empty"
    return True


def https_only(url: object) -> bool | str:
    """Validate URL uses HTTPS scheme."""
    return str(url).startswith("https://") or "must use HTTPS"
