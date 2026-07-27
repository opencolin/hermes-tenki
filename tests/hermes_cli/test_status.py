import importlib.machinery
import sys
import types
from types import SimpleNamespace

from hermes_cli.status import show_status


def test_show_status_all_does_not_print_tavily_key_value(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sentinel = "NONSECRET_SENTINEL_VALUE_DO_NOT_PRINT_123456"
    monkeypatch.setenv("TAVILY_API_KEY", sentinel)

    show_status(SimpleNamespace(all=True, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert sentinel not in output


def test_show_status_termux_gateway_section_skips_systemctl(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    def _unexpected_systemctl(*args, **kwargs):
        raise AssertionError("systemctl should not be called in the Termux status view")

    monkeypatch.setattr(status_mod.subprocess, "run", _unexpected_systemctl)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Manager:      Termux / manual process" in output
    assert "Start with:   hermes gateway" in output
    assert "systemd (user)" not in output


def test_show_status_reports_nous_auth_error(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(
        auth_mod,
        "get_nous_auth_status",
        lambda: {
            "logged_in": False,
            "portal_base_url": "https://portal.nousresearch.com",
            "access_expires_at": "2026-04-20T01:00:51+00:00",
            "agent_key_expires_at": "2026-04-20T04:54:24+00:00",
            "has_refresh_token": True,
            "error": "Refresh session has been revoked",
        },
        raising=False,
    )
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Nous Portal   ✗ not logged in (run: hermes portal)" in output
    assert "Error:      Refresh session has been revoked" in output
    assert "Access exp:" in output
    assert "Key exp:" in output


def test_show_status_reports_nous_inference_key_without_portal_login(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    from hermes_cli.nous_account import NousPortalAccountInfo
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(
        auth_mod,
        "get_nous_auth_status",
        lambda: {
            "logged_in": False,
            "inference_credential_present": True,
            "credential_source": "pool:manual opaque key",
            "inference_base_url": "https://inference.example.com/v1",
            "agent_key_expires_at": "2099-01-01T00:00:00+00:00",
        },
        raising=False,
    )
    monkeypatch.setattr(
        status_mod,
        "get_nous_portal_account_info",
        lambda: NousPortalAccountInfo(
            logged_in=False,
            source="inference_key",
            fresh=False,
            inference_credential_present=True,
            inference_base_url="https://inference.example.com/v1",
        ),
        raising=False,
    )
    monkeypatch.setattr(status_mod, "managed_nous_tools_enabled", lambda: False, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Nous Portal   ✗ not logged in (Nous inference key configured)" in output
    assert "Inference:  https://inference.example.com/v1" in output
    assert "Nous inference credentials are configured" in output


# ---------------------------------------------------------------------------
# Helpers shared by xAI OAuth status tests
# ---------------------------------------------------------------------------

def _base_xai_mocks(monkeypatch, tmp_path):
    """Set up the minimal environment for show_status, returning status_mod."""
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
    return status_mod


class TestShowStatusXaiOAuth:
    """xAI OAuth row in hermes status."""

    # ------------------------------------------------------------------
    # Logged-in branch
    # ------------------------------------------------------------------

    def test_logged_in_shows_check_mark_and_label(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True, "auth_store": "/a/auth.json"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "xAI OAuth" in out
        # The logged-in label must appear; the "not logged in" label must not
        assert "✓" in out or "logged in" in out
        assert "not logged in" not in out.split("xAI OAuth", 1)[1].split("\n")[0]

    def test_logged_in_shows_auth_store(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True, "auth_store": "/home/u/.hermes/auth.json"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Auth file:  /home/u/.hermes/auth.json" in out

    def test_logged_in_shows_last_refresh(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {
                                "logged_in": True,
                                "auth_store": "/a/auth.json",
                                "last_refresh": "2026-05-17T10:00:00+00:00",
                            },
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Refreshed:" in out

    def test_logged_in_does_not_show_error_line(self, monkeypatch, capsys, tmp_path):
        """Error field must be suppressed when logged_in is True."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {
                                "logged_in": True,
                                "auth_store": "/a/auth.json",
                                "error": "stale-error-must-not-appear",
                            },
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1]
        assert "stale-error-must-not-appear" not in xai_section

    def test_no_auth_store_line_when_field_absent(self, monkeypatch, capsys, tmp_path):
        """Auth file line must not appear when auth_store is missing."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1].split("◆", 1)[0]
        assert "Auth file:" not in xai_section

    def test_no_refreshed_line_when_last_refresh_absent(self, monkeypatch, capsys, tmp_path):
        """Refreshed line must not appear when last_refresh is not present."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": True, "auth_store": "/a/auth.json"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1].split("◆", 1)[0]
        assert "Refreshed:" not in xai_section

    # ------------------------------------------------------------------
    # Not-logged-in branch
    # ------------------------------------------------------------------

    def test_not_logged_in_shows_login_command(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": False, "error": "no credentials"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "not logged in (run: hermes auth add xai-oauth)" in out

    def test_not_logged_in_shows_error(self, monkeypatch, capsys, tmp_path):
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": False, "error": "Token has expired"},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Error:      Token has expired" in out

    def test_not_logged_in_omits_error_line_when_error_absent(self, monkeypatch, capsys, tmp_path):
        """No Error: line when not logged in but error key is missing."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: {"logged_in": False},
                            raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        xai_section = out.split("xAI OAuth", 1)[1].split("◆", 1)[0]
        assert "Error:" not in xai_section

    # ------------------------------------------------------------------
    # Resilience: import failure and runtime exception
    # ------------------------------------------------------------------

    def test_import_failure_does_not_crash_show_status(self, monkeypatch, capsys, tmp_path):
        """show_status must complete even when get_xai_oauth_auth_status cannot be imported."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.delattr(auth_mod, "get_xai_oauth_auth_status", raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "◆ Auth Providers" in out

    def test_import_failure_does_not_break_other_oauth_providers(self, monkeypatch, capsys, tmp_path):
        """Nous/Codex/MiniMax rows must still appear when xAI import fails."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_nous_auth_status",
                            lambda: {"logged_in": True}, raising=False)
        monkeypatch.delattr(auth_mod, "get_xai_oauth_auth_status", raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "Nous Portal" in out
        assert "MiniMax OAuth" in out

    def test_status_function_exception_does_not_crash(self, monkeypatch, capsys, tmp_path):
        """show_status must not propagate an exception raised by get_xai_oauth_auth_status."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)

        def _raises():
            raise RuntimeError("backend unreachable")

        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", _raises, raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "◆ Auth Providers" in out

    def test_status_function_returns_none_does_not_crash(self, monkeypatch, capsys, tmp_path):
        """get_xai_oauth_auth_status returning None must be handled gracefully."""
        import hermes_cli.auth as auth_mod
        status_mod = _base_xai_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status",
                            lambda: None, raising=False)

        status_mod.show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "xAI OAuth" in out
        assert "not logged in (run: hermes auth add xai-oauth)" in out


# ---------------------------------------------------------------------------
# Tenki workspace sandbox usage rows
# ---------------------------------------------------------------------------

class _FakeUsageLimit:
    """Stand-in for tenki_sandbox.WorkspaceUsageLimit."""

    def __init__(self, key, label, unit, current, maximum=None, help_text=""):
        self.key = key
        self.label = label
        self.unit = unit
        self.current = current
        self.max = maximum
        self.help_text = help_text


def _base_tenki_mocks(monkeypatch, tmp_path, *, backend="tenki",
                      workspace="ws-abc123", project="proj-def456"):
    """show_status environment with the Tenki terminal backend selected."""
    import tools.tenki_config as tenki_config_mod

    status_mod = _base_xai_mocks(monkeypatch, tmp_path)
    import hermes_cli.auth as auth_mod
    monkeypatch.setattr(auth_mod, "get_xai_oauth_auth_status", lambda: {}, raising=False)

    monkeypatch.setenv("TERMINAL_ENV", backend)
    monkeypatch.delenv("TERMINAL_TENKI_IMAGE", raising=False)
    monkeypatch.delenv("TERMINAL_TENKI_SYNC_HERMES_HOME", raising=False)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_api_endpoint",
                        lambda explicit="": "https://api.tenki.test", raising=False)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_workspace_id",
                        lambda explicit="": workspace, raising=False)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_project_id",
                        lambda explicit="": project, raising=False)
    return status_mod


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("_tenki_sandbox_usage must not be called here")


def test_tenki_usage_renders_count_and_humanized_bytes(monkeypatch, capsys, tmp_path):
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path)
    limits = [
        _FakeUsageLimit("sandboxes", "Sandboxes", "count", 2, 5),
        _FakeUsageLimit("snapshot_bytes", "Snapshot storage", "bytes",
                        5 * 1024 * 1024 * 1024, None),
    ]
    monkeypatch.setattr(
        status_mod, "_tenki_sandbox_usage",
        lambda endpoint, workspace_id, project_id, timeout=3.0: (limits, None),
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out

    assert "Sandboxes: 2 active / 5 limit" in out
    assert "Snapshot storage: 5.0 GB used (unlimited)" in out
    # First row carries the label; the continuation row aligns under it
    # without repeating the label column.
    lines = out.splitlines()
    first = [ln for ln in lines if "Sandboxes: 2 active" in ln][0]
    second = [ln for ln in lines if "Snapshot storage:" in ln][0]
    assert first.startswith("  Usage:")
    assert not second.strip().startswith("Usage:")
    assert second.index("Snapshot storage:") == first.index("Sandboxes:")


def test_tenki_usage_passes_resolved_workspace_and_project(monkeypatch, capsys, tmp_path):
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path)
    seen = {}

    def _capture(endpoint, workspace_id, project_id, timeout=3.0):
        seen.update(endpoint=endpoint, workspace_id=workspace_id,
                    project_id=project_id, timeout=timeout)
        return [], None

    monkeypatch.setattr(status_mod, "_tenki_sandbox_usage", _capture, raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    capsys.readouterr()

    assert seen == {
        "endpoint": "https://api.tenki.test",
        "workspace_id": "ws-abc123",
        "project_id": "proj-def456",
        "timeout": 3.0,
    }


def test_tenki_usage_error_degrades_and_report_continues(monkeypatch, capsys, tmp_path):
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(
        status_mod, "_tenki_sandbox_usage",
        lambda endpoint, workspace_id, project_id, timeout=3.0: ([], "deadline exceeded"),
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out

    assert "(unavailable: deadline exceeded)" in out
    assert "◆ Messaging Platforms" in out
    assert "◆ Gateway Service" in out


def test_tenki_usage_helper_exception_degrades_to_dim_row(monkeypatch, capsys, tmp_path):
    """Even a helper that breaks its contract must not kill the rest of the report."""
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path)

    def _raises(endpoint, workspace_id, project_id, timeout=3.0):
        raise RuntimeError("boom")

    monkeypatch.setattr(status_mod, "_tenki_sandbox_usage", _raises, raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out

    assert "Usage:" in out
    assert "◆ Gateway Service" in out


def test_tenki_usage_empty_list_shows_no_usage_data(monkeypatch, capsys, tmp_path):
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(
        status_mod, "_tenki_sandbox_usage",
        lambda endpoint, workspace_id, project_id, timeout=3.0: ([], None),
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out

    assert "(no usage data)" in out


def test_tenki_usage_unresolved_workspace_skips_network_call(monkeypatch, capsys, tmp_path):
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path, workspace="")
    monkeypatch.setattr(status_mod, "_tenki_sandbox_usage", _fail_if_called, raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out

    assert "(workspace not resolved)" in out


def test_non_tenki_backend_never_queries_usage(monkeypatch, capsys, tmp_path):
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path, backend="local")
    monkeypatch.setattr(status_mod, "_tenki_sandbox_usage", _fail_if_called, raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))
    out = capsys.readouterr().out

    assert "Usage:" not in out
    assert "◆ Terminal Backend" in out


def test_tenki_usage_never_prints_the_auth_token(monkeypatch, capsys, tmp_path):
    sentinel = "NONSECRET_TENKI_TOKEN_DO_NOT_PRINT_987654"
    monkeypatch.setenv("TENKI_AUTH_TOKEN", sentinel)
    status_mod = _base_tenki_mocks(monkeypatch, tmp_path)
    monkeypatch.setattr(
        status_mod, "_tenki_sandbox_usage",
        lambda endpoint, workspace_id, project_id, timeout=3.0: (
            [_FakeUsageLimit("sandboxes", "Sandboxes", "count", 1, 3)], None
        ),
        raising=False,
    )

    status_mod.show_status(SimpleNamespace(all=True, deep=False))
    out = capsys.readouterr().out

    assert "Sandboxes: 1 active / 3 limit" in out
    assert sentinel not in out


# --- _tenki_sandbox_usage itself -------------------------------------------


def _install_fake_tenki_sdk(monkeypatch, client_cls):
    module = types.ModuleType("tenki_sandbox")
    module.__spec__ = importlib.machinery.ModuleSpec("tenki_sandbox", None)
    module.Client = client_cls
    monkeypatch.setitem(sys.modules, "tenki_sandbox", module)


def test_tenki_sandbox_usage_without_token_returns_error(monkeypatch):
    from hermes_cli import status as status_mod
    import tools.tenki_config as tenki_config_mod

    class _FakeClient:
        def __init__(self, **kwargs):
            raise AssertionError("Client must not be built without a token")

    _install_fake_tenki_sdk(monkeypatch, _FakeClient)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_auth_token",
                        lambda explicit="": "", raising=False)

    limits, error = status_mod._tenki_sandbox_usage("https://api.tenki.test", "ws-1", "")

    assert limits == []
    assert "not signed in" in error


def test_tenki_sandbox_usage_reports_missing_sdk(monkeypatch):
    """No SDK must not trigger a lazy install prompt — find_spec gates the import."""
    import importlib.util

    from hermes_cli import status as status_mod

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    limits, error = status_mod._tenki_sandbox_usage("", "ws-1", "")

    assert limits == []
    assert "not installed" in error


def test_tenki_sandbox_usage_sets_deadline_and_closes_client(monkeypatch):
    """Regression guard: no explicit timeout means the RPC has no gRPC deadline."""
    from hermes_cli import status as status_mod
    import tools.tenki_config as tenki_config_mod

    recorded = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            recorded["init"] = kwargs
            recorded["closed"] = False

        def get_workspace_sandbox_usage(self, workspace_id, *, project_id=None):
            recorded["call"] = (workspace_id, project_id)
            return [_FakeUsageLimit("sandboxes", "Sandboxes", "count", 1, 4)]

        def close(self):
            recorded["closed"] = True

    _install_fake_tenki_sdk(monkeypatch, _FakeClient)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_auth_token",
                        lambda explicit="": "cookie:token", raising=False)

    # Deliberately no timeout kwarg: the helper's DEFAULT must carry the
    # deadline, otherwise removing it would ship green while re-introducing
    # the unbounded RPC.
    limits, error = status_mod._tenki_sandbox_usage(
        "https://api.tenki.test", "ws-1", "proj-1"
    )

    assert error is None
    assert len(limits) == 1
    assert recorded["init"]["timeout"] == 3.0
    assert recorded["init"]["base_url"] == "https://api.tenki.test"
    assert recorded["call"] == ("ws-1", "proj-1")
    assert recorded["closed"] is True


def test_tenki_sandbox_usage_closes_client_and_reports_rpc_failure(monkeypatch):
    from hermes_cli import status as status_mod
    import tools.tenki_config as tenki_config_mod

    recorded = {"closed": False}

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_workspace_sandbox_usage(self, workspace_id, *, project_id=None):
            raise RuntimeError("sandbox: unauthorized")

        def close(self):
            recorded["closed"] = True

    _install_fake_tenki_sdk(monkeypatch, _FakeClient)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_auth_token",
                        lambda explicit="": "cookie:token", raising=False)

    limits, error = status_mod._tenki_sandbox_usage("", "ws-1", "")

    assert limits == []
    assert error == "sandbox: unauthorized"
    assert recorded["closed"] is True


def test_tenki_sandbox_usage_reports_constructor_failure(monkeypatch):
    """Client() raises MissingAuthTokenError-style errors — the ctor is inside the try."""
    from hermes_cli import status as status_mod
    import tools.tenki_config as tenki_config_mod

    class _FakeClient:
        def __init__(self, **kwargs):
            raise ValueError("")

    _install_fake_tenki_sdk(monkeypatch, _FakeClient)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_auth_token",
                        lambda explicit="": "cookie:token", raising=False)

    limits, error = status_mod._tenki_sandbox_usage("", "ws-1", "")

    assert limits == []
    assert error == "ValueError"


def test_tenki_sandbox_usage_close_failure_keeps_result(monkeypatch):
    """A close() failure after a successful RPC must not discard the result."""
    from hermes_cli import status as status_mod
    import tools.tenki_config as tenki_config_mod

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_workspace_sandbox_usage(self, workspace_id, *, project_id=None):
            return [_FakeUsageLimit("sandboxes", "Sandboxes", "count", 2, 5)]

        def close(self):
            raise RuntimeError("channel already torn down")

    _install_fake_tenki_sdk(monkeypatch, _FakeClient)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_auth_token",
                        lambda explicit="": "cookie:token", raising=False)

    limits, error = status_mod._tenki_sandbox_usage("", "ws-1", "")

    assert error is None
    assert len(limits) == 1


def test_tenki_sandbox_usage_redacts_token_from_error(monkeypatch):
    """An upstream error that echoes the credential must not reach the row."""
    from hermes_cli import status as status_mod
    import tools.tenki_config as tenki_config_mod

    sentinel = "cookie:NONSECRET_SENTINEL_TOKEN_13579"

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_workspace_sandbox_usage(self, workspace_id, *, project_id=None):
            raise RuntimeError(f"auth rejected for {sentinel}\nretry later")

        def close(self):
            pass

    _install_fake_tenki_sdk(monkeypatch, _FakeClient)
    monkeypatch.setattr(tenki_config_mod, "resolve_tenki_auth_token",
                        lambda explicit="": sentinel, raising=False)

    limits, error = status_mod._tenki_sandbox_usage("", "ws-1", "")

    assert limits == []
    assert sentinel not in error
    assert "<redacted>" in error
    assert "\n" not in error  # single line for the fixed-width row


def test_format_tenki_usage_handles_unknown_unit():
    from hermes_cli import status as status_mod

    text = status_mod._format_tenki_usage(
        _FakeUsageLimit("weird", "Widgets", "unspecified", 7, 9)
    )

    # Unknown units get no qualifier — "active" is only known-true for counts.
    assert text == "Widgets: 7 / 9 limit"
