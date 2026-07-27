"""Profile-scoping of the process-global terminal environment cache.

The multiplexing gateway serves every profile from ONE process, and almost
every task_id collapses to the literal ``"default"`` container
(``_resolve_container_task_id``).  Before profile scoping that meant profile B
was handed the live sandbox — and the forwarded credentials baked into it —
that profile A's turn had created.

The fix keys ``_active_environments`` / ``_last_activity`` / ``_creation_locks``
(and file_tools' ``_file_ops_cache`` / ``_last_known_cwd``) by
``"<profile token>\\x00<container id>"``.  The profile half is bound while the
turn's ``HERMES_HOME`` contextvar is live; cleanup — which runs on bare threads
with a fresh ``contextvars.Context`` — is driven purely off keys it already
holds.
"""

import hashlib
import threading

import pytest

import hermes_constants
from tools import file_tools, terminal_tool


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    """Give every test its own cache dicts and no stray env overrides."""
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks_lock", threading.Lock())
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.setattr(file_tools, "_file_ops_cache", {})
    monkeypatch.setattr(file_tools, "_last_known_cwd", {})
    yield


class _FakeEnv:
    """Minimal stand-in for a BaseEnvironment."""

    def __init__(self, label: str):
        self.label = label
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def _with_home(path):
    """Context-manager-ish helper: set the HERMES_HOME override, return token."""
    return hermes_constants.set_hermes_home_override(path)


# ── the token itself ────────────────────────────────────────────────────────

def test_token_matches_legacy_tenki_formula(tmp_path, monkeypatch):
    """Byte-identical to what tenki's ``_profile_token`` emitted before, so
    live sandbox names and ``hermes_profile`` metadata don't churn."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    token = _with_home(tmp_path)
    try:
        expected = hashlib.sha1(
            str(tmp_path.resolve()).encode("utf-8")
        ).hexdigest()[:10]
        assert hermes_constants.active_profile_token() == expected
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_tenki_profile_token_delegates(tmp_path, monkeypatch):
    from tools.environments import tenki

    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    token = _with_home(tmp_path)
    try:
        assert tenki._profile_token() == hermes_constants.active_profile_token()
    finally:
        hermes_constants.reset_hermes_home_override(token)


def test_hermes_profile_env_preferred_when_not_multiplexing(tmp_path, monkeypatch):
    """Non-multiplex deployments keep the legacy HERMES_PROFILE basis, so the
    token survives a home-directory move."""
    monkeypatch.setattr(
        "agent.secret_scope.is_multiplex_active", lambda: False, raising=True
    )
    monkeypatch.setenv("HERMES_PROFILE", "coder")

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    token = _with_home(home_a)
    try:
        first = hermes_constants.active_profile_token()
    finally:
        hermes_constants.reset_hermes_home_override(token)
    token = _with_home(home_b)
    try:
        second = hermes_constants.active_profile_token()
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert first == second == hashlib.sha1(b"profile:coder").hexdigest()[:10]


# ── key composition ─────────────────────────────────────────────────────────

def test_distinct_homes_give_distinct_keys_for_one_task_id(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    home_a, home_b = tmp_path / "a", tmp_path / "b"

    token = _with_home(home_a)
    try:
        key_a = terminal_tool._env_cache_key(None)
        key_a_again = terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)

    token = _with_home(home_b)
    try:
        key_b = terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert key_a == key_a_again, "same home must yield a stable key"
    assert key_a != key_b, "two profiles must not share one cache entry"
    # Both still collapse to the same shared container id.
    assert terminal_tool._container_id_from_cache_key(key_a) == "default"
    assert terminal_tool._container_id_from_cache_key(key_b) == "default"


def test_container_id_round_trips(tmp_path):
    token = _with_home(tmp_path)
    try:
        assert terminal_tool._container_id_from_cache_key(
            terminal_tool._env_cache_key(None)
        ) == "default"

        terminal_tool.register_task_env_overrides("rl-7", {"docker_image": "x:y"})
        assert terminal_tool._container_id_from_cache_key(
            terminal_tool._env_cache_key("rl-7")
        ) == "rl-7"
    finally:
        hermes_constants.reset_hermes_home_override(token)

    # An un-prefixed (legacy/raw) key round-trips unchanged so cleanup can
    # decompose keys it did not compose.
    assert terminal_tool._container_id_from_cache_key("default") == "default"


def test_multiplex_ignores_process_global_hermes_profile(tmp_path, monkeypatch):
    """HERMES_PROFILE lives in the process-global ``os.environ`` (see
    ``secret_scope._GLOBAL_ENV_EXACT``), so under multiplexing it would collapse
    every served profile onto one token.  It must be ignored there."""
    monkeypatch.setattr(
        "agent.secret_scope.is_multiplex_active", lambda: True, raising=True
    )
    monkeypatch.setenv("HERMES_PROFILE", "shared-process-value")

    home_a, home_b = tmp_path / "a", tmp_path / "b"
    token = _with_home(home_a)
    try:
        key_a = terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)
    token = _with_home(home_b)
    try:
        key_b = terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert key_a != key_b


# ── the isolation the keys buy ──────────────────────────────────────────────

def test_profile_b_does_not_see_profile_a_env(tmp_path, monkeypatch):
    """Simulated two-profile gateway flow: A creates the shared ``"default"``
    sandbox; B's turn must miss it and be told to build its own."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    env_a = _FakeEnv("profile-a")

    token = _with_home(home_a)
    try:
        key_a = terminal_tool._env_cache_key(None)
        terminal_tool._active_environments[key_a] = env_a
        assert terminal_tool.get_active_env(None) is env_a
    finally:
        hermes_constants.reset_hermes_home_override(token)

    token = _with_home(home_b)
    try:
        assert terminal_tool._env_cache_key(None) not in terminal_tool._active_environments
        assert terminal_tool.get_active_env(None) is None
        assert terminal_tool.get_active_env("subagent-3-cafe") is None
    finally:
        hermes_constants.reset_hermes_home_override(token)

    # A's entry is untouched — B missing must not evict A.
    assert terminal_tool._active_environments[key_a] is env_a


def test_file_ops_cache_is_profile_scoped(tmp_path, monkeypatch):
    """file_tools must derive the SAME composite keys as terminal_tool
    (lockstep scoping) — asserting file_tools' own derivation, not dict
    semantics — and two profiles must never share a key."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    home_a, home_b = tmp_path / "a", tmp_path / "b"

    token = _with_home(home_a)
    try:
        keys_a = file_tools._env_cache_keys_for(None)
        assert keys_a[0] == terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)

    token = _with_home(home_b)
    try:
        keys_b = file_tools._env_cache_keys_for(None)
        assert keys_b[0] == terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert set(keys_a).isdisjoint(set(keys_b))


# ── cleanup stays key-driven ────────────────────────────────────────────────

def test_cleanup_vm_evicts_via_raw_task_id_backcompat(tmp_path):
    """``AIAgent.close()`` passes a raw session id that predates composite
    keying; that entry must still be reclaimed."""
    env = _FakeEnv("legacy")
    terminal_tool._active_environments["legacy-session-id"] = env
    terminal_tool._last_activity["legacy-session-id"] = 0.0

    token = _with_home(tmp_path)
    try:
        terminal_tool.cleanup_vm("legacy-session-id")
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert "legacy-session-id" not in terminal_tool._active_environments
    assert "legacy-session-id" not in terminal_tool._last_activity
    assert env.cleaned is True


def test_cleanup_vm_falsy_task_id_is_a_noop(tmp_path, monkeypatch):
    """Pre-composite behavior preserved: a falsy task_id never matched a cache
    entry, so it must NOT resolve to the shared container and tear it down."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    env = _FakeEnv("a")
    token = _with_home(tmp_path / "a")
    try:
        key = terminal_tool._env_cache_key(None)
        terminal_tool._active_environments[key] = env
        terminal_tool._last_activity[key] = 0.0
        terminal_tool.cleanup_vm(None)
        assert key in terminal_tool._active_environments  # untouched
        terminal_tool._active_environments.pop(key, None)
        terminal_tool._last_activity.pop(key, None)
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert env.cleaned is False


def test_cleanup_vm_evicts_composite_key_from_turn_context(tmp_path, monkeypatch):
    """An explicit shared-container id from inside the turn's context evicts
    the profile's composite entry."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    env = _FakeEnv("a")
    token = _with_home(tmp_path / "a")
    try:
        key = terminal_tool._env_cache_key(None)
        terminal_tool._active_environments[key] = env
        terminal_tool._last_activity[key] = 0.0
        terminal_tool.cleanup_vm("default")
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert terminal_tool._active_environments == {}
    assert env.cleaned is True


def test_cleanup_vm_with_per_turn_uuid_does_not_evict_shared_env(tmp_path, monkeypatch):
    """``cleanup_task_resources`` runs after EVERY turn with a fresh uuid
    task_id. That id collapses to the shared ``"default"`` container, but
    ``cleanup_vm`` must not follow the collapse — otherwise the shared sandbox
    (and the shell state local/docker keep across turns) is destroyed every
    turn. Same semantics as before profile scoping, just profile-scoped."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    env = _FakeEnv("shared")
    token = _with_home(tmp_path / "a")
    try:
        key = terminal_tool._env_cache_key(None)
        terminal_tool._active_environments[key] = env
        terminal_tool.cleanup_vm("f2f0c0de-0000-4000-8000-000000000001")
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert terminal_tool._active_environments[key] is env
    assert env.cleaned is False


def test_cleanup_all_sweeps_every_profile_without_contextvar(tmp_path, monkeypatch):
    """Gateway shutdown / atexit run with NO HERMES_HOME override — they must
    still reclaim every profile's sandbox."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.setenv("TERMINAL_SCRATCH_DIR", str(tmp_path / "scratch"))

    envs = {}
    for name in ("a", "b"):
        token = _with_home(tmp_path / name)
        try:
            envs[terminal_tool._env_cache_key(None)] = _FakeEnv(name)
        finally:
            hermes_constants.reset_hermes_home_override(token)
    assert len(envs) == 2, "the two profiles must have produced two keys"
    terminal_tool._active_environments.update(envs)

    # No override installed here — exactly the atexit situation.
    cleaned = terminal_tool.cleanup_all_environments()

    assert cleaned == 2
    assert terminal_tool._active_environments == {}
    assert all(env.cleaned for env in envs.values())


def test_idle_reaper_queries_process_registry_with_bare_container_id(tmp_path, monkeypatch):
    """``process_registry`` sessions are registered with the plain container id,
    so the reaper must strip the profile prefix before asking — otherwise envs
    with live background processes get reaped out from under them."""
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    seen = []

    class _Registry:
        def has_active_processes(self, task_id, *, env_cache_key=None):
            seen.append(task_id)
            return True

    monkeypatch.setattr(
        "tools.process_registry.process_registry", _Registry(), raising=True
    )

    token = _with_home(tmp_path / "a")
    try:
        key = terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)
    env = _FakeEnv("a")
    terminal_tool._active_environments[key] = env
    terminal_tool._last_activity[key] = 0.0

    # Runs with no HERMES_HOME override, like the daemon cleanup thread.
    terminal_tool._cleanup_inactive_envs(lifetime_seconds=0)

    assert seen == ["default"]
    # has_active_processes() returned True, so the env is kept alive.
    assert terminal_tool._active_environments[key] is env
    assert env.cleaned is False


def test_idle_reaper_clears_file_ops_with_composite_key(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    class _Registry:
        def has_active_processes(self, task_id, *, env_cache_key=None):
            return False

    monkeypatch.setattr(
        "tools.process_registry.process_registry", _Registry(), raising=True
    )

    token = _with_home(tmp_path / "a")
    try:
        key = terminal_tool._env_cache_key(None)
    finally:
        hermes_constants.reset_hermes_home_override(token)
    env = _FakeEnv("a")
    terminal_tool._active_environments[key] = env
    terminal_tool._last_activity[key] = 0.0
    file_tools._file_ops_cache[key] = object()

    terminal_tool._cleanup_inactive_envs(lifetime_seconds=0)

    assert terminal_tool._active_environments == {}
    assert file_tools._file_ops_cache == {}
    assert env.cleaned is True


def test_has_active_processes_filters_by_env_cache_key(monkeypatch):
    """The keep-alive probe must not let profile A's background process pin
    profile B's idle sandbox: stamped sessions match only their own composite
    key, while legacy unstamped sessions conservatively match every profile."""
    from tools.process_registry import process_registry

    class _Session:
        def __init__(self, task_id, env_cache_key):
            self.task_id = task_id
            self.exited = False
            self.env_cache_key = env_cache_key

    monkeypatch.setattr(process_registry, "_refresh_detached_session", lambda s: None)

    stamped = _Session("default", "tokA\x00default")
    monkeypatch.setattr(process_registry, "_running", {"1": stamped})
    assert process_registry.has_active_processes("default", env_cache_key="tokA\x00default")
    assert not process_registry.has_active_processes("default", env_cache_key="tokB\x00default")
    assert process_registry.has_active_processes("default")  # bare query unchanged

    legacy = _Session("default", None)
    monkeypatch.setattr(process_registry, "_running", {"1": legacy})
    assert process_registry.has_active_processes("default", env_cache_key="tokB\x00default")
