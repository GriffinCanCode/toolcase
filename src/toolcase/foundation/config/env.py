"""Environment variable loading utilities.

Provides flexible, framework-agnostic env var loading that works with:
- Standard os.environ
- .env files (via pydantic-settings or python-dotenv)
- Multiple env file variants (.env.local, .env.development, etc.)
- Environment-specific overrides
- Direct secret loading with fallback chains

Example:
    >>> from toolcase.foundation.config import load_env, get_env, require_env
    >>>
    >>> # Load .env files with priority ordering
    >>> load_env()  # Loads .env, .env.local, .env.{ENVIRONMENT}
    >>>
    >>> # Get env vars with defaults
    >>> api_key = get_env("OPENAI_API_KEY")
    >>> debug = get_env("DEBUG", default="false", cast=bool)
    >>>
    >>> # Require env vars (raises if missing)
    >>> secret = require_env("DATABASE_URL")
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, TypeVar, overload

T = TypeVar("T")

# Standard env file search order (later files override earlier ones)
_ENV_FILE_PRIORITY: tuple[str, ...] = (
    ".env",
    ".env.local",
    ".env.shared",
)


def _env_files_for_environment(environment: str) -> tuple[str, ...]:
    """Get env file names for a specific environment."""
    return (
        ".env",
        f".env.{environment}",
        f".env.{environment}.local",
        ".env.local",
    )


def _find_env_files(
    base_path: Path | str | None = None,
    environment: str | None = None,
) -> list[Path]:
    """Find env files in priority order (later overrides earlier).
    
    Args:
        base_path: Directory to search for env files. Defaults to cwd.
        environment: Environment name for env-specific files (development, staging, production).
    
    Returns:
        List of existing env file paths in load order.
    """
    base = Path(base_path) if base_path else Path.cwd()
    
    # Determine which files to look for
    if environment:
        file_names = _env_files_for_environment(environment.lower())
    else:
        file_names = _ENV_FILE_PRIORITY
    
    return [f for name in file_names if (f := base / name).is_file()]


def load_env(
    base_path: Path | str | None = None,
    environment: str | None = None,
    override: bool = False,
) -> dict[str, str]:
    """Load environment variables from .env files.
    
    Supports multiple env file formats and priority ordering:
    - .env (base configuration)
    - .env.local (local overrides, gitignored)
    - .env.{environment} (environment-specific: .env.development, .env.production)
    - .env.{environment}.local (environment-specific local overrides)
    
    Compatible with:
    - python-dotenv
    - pydantic-settings
    - django-environ
    - python-decouple
    
    Args:
        base_path: Directory to search for env files. Defaults to cwd.
        environment: Environment name (auto-detected from ENV/ENVIRONMENT if not provided).
        override: Whether to override existing env vars (default: False).
    
    Returns:
        Dict of all loaded environment variables.
    
    Example:
        >>> load_env()  # Auto-detect environment
        >>> load_env(environment="production")
        >>> load_env(base_path="/app", override=True)
    """
    # Auto-detect environment if not provided
    if environment is None:
        environment = os.environ.get("TOOLCASE_ENVIRONMENT") or os.environ.get("ENVIRONMENT") or os.environ.get("ENV")
    
    env_files = _find_env_files(base_path, environment)
    loaded: dict[str, str] = {}
    
    for env_file in env_files:
        loaded.update(_parse_env_file(env_file, override=override))
    
    return loaded


def _parse_env_file(file_path: Path, override: bool = False) -> dict[str, str]:
    """Parse a single .env file and set environment variables.
    
    Supports standard .env format:
    - KEY=value
    - KEY="value with spaces"
    - KEY='value with spaces'
    - # comments
    - export KEY=value (shell-compatible)
    """
    loaded: dict[str, str] = {}
    
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return loaded
    
    for line in content.splitlines():
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        
        # Handle 'export KEY=value' format
        if line.startswith("export "):
            line = line[7:].strip()
        
        # Parse KEY=value
        if "=" not in line:
            continue
        
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        
        # Remove quotes if present
        if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
            value = value[1:-1]
        
        # Handle escape sequences in double-quoted strings
        if value and '"' not in line.partition("=")[2].strip()[0:1]:
            value = value.encode().decode("unicode_escape")
        
        # Set in environment
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    
    return loaded


# Type casting utilities
def _cast_bool(value: str) -> bool:
    """Cast string to boolean (handles common truthy/falsy values)."""
    return value.lower() in ("true", "1", "yes", "on", "enabled")


def _cast_int(value: str) -> int:
    """Cast string to integer."""
    return int(value)


def _cast_float(value: str) -> float:
    """Cast string to float."""
    return float(value)


def _cast_list(value: str, sep: str = ",") -> list[str]:
    """Cast string to list (comma-separated by default)."""
    return [v.strip() for v in value.split(sep) if v.strip()]


_CASTERS: dict[type, Callable[[str], object]] = {
    bool: _cast_bool,
    int: _cast_int,
    float: _cast_float,
    list: _cast_list,
    str: str,
}


@overload
def get_env(key: str) -> str | None: ...
@overload
def get_env(key: str, default: str) -> str: ...
@overload
def get_env(key: str, default: T) -> str | T: ...
@overload
def get_env(key: str, *, cast: Literal[bool]) -> bool | None: ...
@overload
def get_env(key: str, default: str, *, cast: Literal[bool]) -> bool: ...
@overload
def get_env(key: str, *, cast: Literal[int]) -> int | None: ...
@overload
def get_env(key: str, default: str, *, cast: Literal[int]) -> int: ...
@overload
def get_env(key: str, *, cast: Literal[float]) -> float | None: ...
@overload
def get_env(key: str, default: str, *, cast: Literal[float]) -> float: ...
@overload
def get_env(key: str, *, cast: type[T]) -> T | None: ...
@overload
def get_env(key: str, default: str, *, cast: type[T]) -> T: ...


def get_env(
    key: str,
    default: str | T | None = None,
    *,
    cast: type[T] | None = None,
) -> str | T | None:
    """Get environment variable with optional type casting.
    
    Args:
        key: Environment variable name.
        default: Default value if not set (returned as-is, not cast).
        cast: Type to cast the value to (bool, int, float, list, or custom callable).
    
    Returns:
        The environment variable value, cast if specified, or default.
    
    Example:
        >>> get_env("API_KEY")
        >>> get_env("DEBUG", "false", cast=bool)
        >>> get_env("PORT", "8000", cast=int)
        >>> get_env("HOSTS", cast=list)  # Comma-separated
    """
    value = os.environ.get(key)
    
    if value is None:
        return default
    
    if cast is None:
        return value
    
    caster = _CASTERS.get(cast, cast)
    return caster(value)  # type: ignore[return-value]


def require_env(key: str, cast: type[T] | None = None) -> str | T:
    """Get required environment variable (raises if missing).
    
    Args:
        key: Environment variable name.
        cast: Type to cast the value to.
    
    Returns:
        The environment variable value.
    
    Raises:
        EnvironmentError: If the variable is not set.
    
    Example:
        >>> api_key = require_env("OPENAI_API_KEY")
        >>> port = require_env("PORT", cast=int)
    """
    value = os.environ.get(key)
    
    if value is None:
        raise EnvironmentError(f"Required environment variable '{key}' is not set")
    
    if cast is None:
        return value
    
    caster = _CASTERS.get(cast, cast)
    return caster(value)  # type: ignore[return-value]


def get_env_prefix(prefix: str, strip_prefix: bool = True) -> dict[str, str]:
    """Get all environment variables with a given prefix.
    
    Args:
        prefix: Prefix to filter by (e.g., "MYAPP_").
        strip_prefix: Whether to remove the prefix from returned keys.
    
    Returns:
        Dict of matching environment variables.
    
    Example:
        >>> get_env_prefix("REDIS_")  # {"HOST": "...", "PORT": "..."}
        >>> get_env_prefix("REDIS_", strip_prefix=False)  # {"REDIS_HOST": "..."}
    """
    prefix_upper = prefix.upper()
    return {
        (k[len(prefix_upper):] if strip_prefix else k): v
        for k, v in os.environ.items()
        if k.upper().startswith(prefix_upper)
    }


@lru_cache(maxsize=1)
def get_env_files_loaded() -> tuple[Path, ...]:
    """Get list of env files that were loaded (cached)."""
    return tuple(_find_env_files(environment=os.environ.get("TOOLCASE_ENVIRONMENT")))


# Compatibility with python-dotenv
def dotenv_values(dotenv_path: str | Path | None = None) -> dict[str, str]:
    """Load .env file and return values without setting them in os.environ.
    
    Compatible with python-dotenv's dotenv_values() function.
    
    Args:
        dotenv_path: Path to .env file. Defaults to .env in cwd.
    
    Returns:
        Dict of parsed environment variables.
    """
    path = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
    
    if not path.is_file():
        return {}
    
    values: dict[str, str] = {}
    content = path.read_text(encoding="utf-8")
    
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        
        values[key] = value
    
    return values


# Convenience alias
env = get_env
