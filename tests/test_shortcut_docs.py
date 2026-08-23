import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHORTCUTS = ROOT / "shell" / "slai-shortcuts.sh"
README = ROOT / "README.md"

PUBLIC_SHORTCUTS = {
    "s",
    "sr",
    "srx",
    "sim",
    "simx",
    "sc",
    "scx",
    "scc",
    "scs",
    "scsx",
    "su",
    "sul",
    "st",
    "si",
    "se",
    "sp",
    "sd",
    "sdx",
    "sph",
    "sphx",
    "spb",
    "spbx",
    "sv",
    "sl",
    "rec",
    "wuji-check",
    "wrist-park",
    "sm",
}


def _defined_shell_functions() -> set[str]:
    source = SHORTCUTS.read_text(encoding="utf-8")
    return set(re.findall(r"^([A-Za-z][A-Za-z0-9_-]*)\(\)\s*[{(]", source, re.MULTILINE))


def _shortcut_table_commands() -> set[str]:
    readme = README.read_text(encoding="utf-8")
    section = readme.split("## 0. 超短快捷命令（完整清单）", 1)[1].split("## 1.", 1)[0]
    return set(re.findall(r"^\| `([^`]+)` \|", section, re.MULTILINE))


def test_public_shortcut_inventory_matches_shell_script() -> None:
    defined = _defined_shell_functions()
    assert {name for name in defined if not name.startswith("_")} == PUBLIC_SHORTCUTS


def test_readme_table_documents_every_shortcut_once() -> None:
    documented = _shortcut_table_commands()
    expected_rows = (PUBLIC_SHORTCUTS - {"sm"}) | {"sm collect"}
    assert documented == expected_rows


def test_readme_documents_activation_and_collection_environment() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "source ~/.bash_aliases" in readme
    assert ".venv-lerobot-v3/bin/python" in readme
    assert "不是所有快捷命令都走 `uv run`" in readme
    assert "`sm collect`" in readme


def _scx_arguments(*arguments: str) -> list[str]:
    command = (
        f'source "{SHORTCUTS}"\n'
        "_scollect() { printf '%s\\n' \"$@\"; }\n"
        'scx "$@"\n'
    )
    result = subprocess.run(
        ["bash", "-c", command, "scx-test", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def test_scx_defaults_to_unlimited_continuous_collection() -> None:
    arguments = _scx_arguments("--no-open-dashboard")
    assert "--continuous" in arguments
    assert "--execute-real" in arguments


def test_scx_respects_an_explicit_episode_limit() -> None:
    arguments = _scx_arguments("--episodes", "3")
    assert "--continuous" not in arguments
    assert arguments[:2] == ["--episodes", "3"]
