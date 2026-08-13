from __future__ import annotations

import sys
import types

import pytest

from slai_mi.runtime import TeleopDependencies
from slai_mi.runtime.adapters import (
    AdapterPluginError,
    adapter_plugin_spec,
    build_adapter_dependencies,
    load_adapter_factory,
)


def test_plugin_import_is_lazy_and_factory_receives_configs(monkeypatch) -> None:
    name = "fake_site_adapters"
    assert name not in sys.modules
    module = types.ModuleType(name)
    received = []

    def factory(hardware, task):
        received.append((hardware, task))
        return TeleopDependencies(lambda *_: None, lambda *_: None, lambda *_: None)

    module.make_teleop = factory
    monkeypatch.setitem(sys.modules, name, module)
    dependencies = build_adapter_dependencies(
        f"{name}:make_teleop", TeleopDependencies, {"site": "lab"}, {"task": "pick"}
    )
    assert isinstance(dependencies, TeleopDependencies)
    assert received == [({"site": "lab"}, {"task": "pick"})]


def test_cli_plugin_takes_precedence_over_config() -> None:
    assert adapter_plugin_spec("cli:factory", {"adapter_plugin": "config:factory"}) == (
        "cli:factory"
    )
    assert adapter_plugin_spec(None, {"adapter_plugin": "config:factory"}) == (
        "config:factory"
    )


@pytest.mark.parametrize("spec", ["missing_separator", ":factory", "module:"])
def test_malformed_plugin_specs_fail_clearly(spec: str) -> None:
    with pytest.raises(AdapterPluginError, match="module:factory"):
        load_adapter_factory(spec)


def test_plugin_must_return_requested_dependency_contract(monkeypatch) -> None:
    module = types.ModuleType("wrong_adapter")
    module.factory = lambda *_args: object()
    monkeypatch.setitem(sys.modules, "wrong_adapter", module)
    with pytest.raises(AdapterPluginError, match="expected TeleopDependencies"):
        build_adapter_dependencies("wrong_adapter:factory", TeleopDependencies, {})
