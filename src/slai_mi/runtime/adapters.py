"""Loading boundary for site-specific real-hardware adapter factories."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

T = TypeVar("T")


class AdapterPluginError(RuntimeError):
    pass


def adapter_plugin_spec(cli_value: str | None, config: Mapping[str, Any]) -> str | None:
    """CLI wins; otherwise read a non-secret plugin import path from hardware config."""
    configured = config.get("adapter_plugin")
    value = cli_value if cli_value is not None else configured
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AdapterPluginError("adapter_plugin must be a non-empty module:factory string")
    return value.strip()


def load_adapter_factory(spec: str) -> Callable[..., T]:
    """Load a callable factory without importing SDKs until real execution is requested."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise AdapterPluginError("adapter plugin must use module:factory syntax")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise AdapterPluginError(f"cannot import adapter module {module_name!r}: {exc}") from exc
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise AdapterPluginError(f"adapter factory {spec!r} is missing or not callable")
    return factory


def build_adapter_dependencies(
    spec: str,
    expected_type: type[T],
    *factory_args: Mapping[str, Any],
) -> T:
    dependencies = load_adapter_factory(spec)(*factory_args)
    if not isinstance(dependencies, expected_type):
        raise AdapterPluginError(
            f"adapter factory {spec!r} returned {type(dependencies).__name__}; "
            f"expected {expected_type.__name__}"
        )
    return dependencies
