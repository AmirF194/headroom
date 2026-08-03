"""Tests for reversible Claude Code VS Code configuration."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from headroom.providers.claude.vscode import (
    claude_user_settings_path,
    configure_vscode_claude_settings,
    remove_vscode_claude_settings,
    vscode_claude_proxy_url,
)


def test_settings_path_honors_claude_config_dir(tmp_path: Path) -> None:
    assert claude_user_settings_path({"CLAUDE_CONFIG_DIR": str(tmp_path)}) == (
        tmp_path / "settings.json"
    )


def test_settings_path_uses_windows_profile() -> None:
    path = claude_user_settings_path(
        {"HOME": "/wrong", "USERPROFILE": r"C:\\Users\\claude"}, platform="win32"
    )
    assert path == Path(r"C:\\Users\\claude") / ".claude" / "settings.json"


def test_proxy_url_is_project_scoped() -> None:
    assert vscode_claude_proxy_url(8787, "my project").endswith("/p/my%20project")


def test_configure_and_remove_preserve_unrelated_and_previous_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Read"]},
                "env": {
                    "KEEP": "yes",
                    "ANTHROPIC_BASE_URL": "https://gateway.example",
                    "ENABLE_TOOL_SEARCH": "false",
                },
            }
        ),
        encoding="utf-8",
    )

    assert configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo") == "added"
    configured = json.loads(path.read_text(encoding="utf-8"))
    assert configured["env"] == {
        "KEEP": "yes",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:8787/p/demo",
        "ENABLE_TOOL_SEARCH": "true",
    }
    assert configured["permissions"] == {"allow": ["Read"]}

    assert remove_vscode_claude_settings(path)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["env"] == {
        "KEEP": "yes",
        "ANTHROPIC_BASE_URL": "https://gateway.example",
        "ENABLE_TOOL_SEARCH": "false",
    }
    assert restored["permissions"] == {"allow": ["Read"]}
    assert not (tmp_path / ".headroom-vscode-claude.json").exists()


def test_reconfigure_updates_port_without_losing_original_values(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"env":{"ANTHROPIC_BASE_URL":"https://original.example"}}', encoding="utf-8")

    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    assert configure_vscode_claude_settings(path, "http://127.0.0.1:9999/p/demo") == "updated"
    assert remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8"))["env"] == {
        "ANTHROPIC_BASE_URL": "https://original.example"
    }


def test_remove_deletes_settings_created_only_for_headroom(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    assert path.exists()
    assert remove_vscode_claude_settings(path)
    assert not path.exists()


def test_configure_refuses_malformed_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(click.ClickException, match="not valid JSON"):
        configure_vscode_claude_settings(path, "http://127.0.0.1:8787")
    assert path.read_text(encoding="utf-8") == "{broken"


def test_remove_refuses_to_overwrite_changed_managed_value(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    configure_vscode_claude_settings(path, "http://127.0.0.1:8787/p/demo")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["env"]["ANTHROPIC_BASE_URL"] = "https://user-change.example"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(click.ClickException, match="refusing to overwrite"):
        remove_vscode_claude_settings(path)
    assert json.loads(path.read_text(encoding="utf-8"))["env"]["ANTHROPIC_BASE_URL"] == (
        "https://user-change.example"
    )
