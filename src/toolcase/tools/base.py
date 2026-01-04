"""Base configuration and utilities for built-in tools.

Provides extensible configuration patterns that all built-in tools follow,
enabling customization without subclassing for common use cases.
"""

from __future__ import annotations

from abc import ABC
from typing import Annotated, ClassVar, Generic, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from toolcase.foundation.core import BaseTool, ToolMetadata

TConfig = TypeVar("TConfig", bound="ToolConfig")
TParams = TypeVar("TParams", bound=BaseModel)


class ToolConfig(BaseModel, ABC):
    """Base configuration for built-in tools.
    
    Subclass this to define tool-specific configuration options.
    All configs support runtime updates and validation.
    
    Attributes:
        enabled: Whether the tool is active
        timeout: Operation timeout in seconds
    
    Example:
        >>> class MyToolConfig(ToolConfig):
        ...     max_items: int = 100
        ...     timeout: float = 30.0
    """
    
    model_config = ConfigDict(
        frozen=False,
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        validate_assignment=True,
        json_schema_extra={"title": "Tool Configuration"},
    )
    
    # Private attributes for internal state (not serialized)
    _created_at: float = PrivateAttr(default_factory=lambda: __import__("time").time())
    _update_count: int = PrivateAttr(default=0)
    
    # Common config options
    enabled: bool = Field(default=True, description="Whether the tool is active")
    timeout: Annotated[float, Field(
        default=30.0,
        ge=0.1,
        le=300.0,
        description="Operation timeout in seconds",
    )]
    
    def __setattr__(self, name: str, value: object) -> None:
        """Track config updates via private attr."""
        super().__setattr__(name, value)
        if not name.startswith("_") and hasattr(self, "_update_count"):
            object.__setattr__(self, "_update_count", self._update_count + 1)
    
    @computed_field
    @property
    def is_active(self) -> bool:
        """Alias for enabled (semantic clarity)."""
        return self.enabled


class ConfigurableTool(BaseTool[TParams], Generic[TParams, TConfig]):
    """Base class for tools with runtime-configurable behavior.
    
    Separates tool parameters (per-call inputs) from configuration
    (instance-level settings), enabling:
    - Runtime reconfiguration without re-registration
    - Security constraints (allowed hosts, methods, etc.)
    - Resource limits (timeouts, max sizes)
    - Environment-specific defaults
    
    Example:
        >>> class MyTool(ConfigurableTool[MyParams, MyConfig]):
        ...     config_class = MyConfig
        ...     
        ...     def _run(self, params: MyParams) -> str:
        ...         if self.config.max_items < params.limit:
        ...             return self._error("Limit exceeds max_items")
        ...         ...
    """
    
    config_class: ClassVar[type[ToolConfig]]
    
    __slots__ = ("_config",)
    
    def __init__(self, config: TConfig | None = None) -> None:
        """Initialize with optional config override.
        
        Args:
            config: Configuration instance. If None, uses defaults.
        """
        self._config: TConfig = config or self.config_class()  # type: ignore[assignment]
    
    @property
    def config(self) -> TConfig:
        """Current configuration (read-only access)."""
        return self._config
    
    def configure(self, **updates: object) -> None:
        """Update configuration at runtime.
        
        Validates updates against the config schema.
        
        Args:
            **updates: Configuration fields to update
        
        Example:
            >>> tool.configure(timeout=60.0, max_retries=5)
        """
        # Exclude computed fields from dump (they can't be passed back to constructor)
        data = self._config.model_dump(exclude_unset=False, exclude_defaults=False)
        # Filter out any computed fields that might have been included
        field_names = set(self.config_class.model_fields.keys())
        data = {k: v for k, v in data.items() if k in field_names}
        data.update(updates)
        self._config = self.config_class(**data)  # type: ignore[assignment]
    
    def with_config(self, **updates: object) -> ConfigurableTool[TParams, TConfig]:
        """Create a new instance with updated config.
        
        Immutable alternative to configure() for functional patterns.
        
        Returns:
            New tool instance with updated configuration
        """
        # Exclude computed fields from dump (they can't be passed back to constructor)
        data = self._config.model_dump(exclude_unset=False, exclude_defaults=False)
        # Filter out any computed fields that might have been included
        field_names = set(self.config_class.model_fields.keys())
        data = {k: v for k, v in data.items() if k in field_names}
        data.update(updates)
        return self.__class__(self.config_class(**data))  # type: ignore[arg-type, return-value]
