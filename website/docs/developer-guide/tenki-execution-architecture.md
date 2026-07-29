---
title: "Tenki Execution Architecture"
description: "How a long-running Hermes agent runs on ephemeral Tenki sandboxes — process topology, sandbox lifecycle, persistence, readiness, and multi-profile isolation"
---

# Tenki Execution Architecture

Hermes is a long-running agent. Tenki sandboxes are short-lived by design. This page explains why that is not a contradiction, and how the `tenki` terminal backend layers session reuse, snapshot persistence, and pause/resume on top of ephemeral sandboxes. It documents the supported SDK range `tenki-sandbox>=0.4.0,<0.6`.

## Process Topology

The apparent mismatch dissolves once the process topology is clear: **the Hermes agent process never runs inside the Tenki sandbox.** The agent loop, model calls, memory, config, credentials, gateway adapters, and cron all run on the host. Tenki is the execution substrate for the terminal tool, the file tools, and `execute_code` — plus delegated subagents, each of which gets its own task and therefore its own sandbox.

```text
┌───────────────────── Host (long-running) ─────────────────────┐
│                                                               │
│  AIAgent loop · gateway · memory · cron · credentials         │
│  HERMES_HOME (config, sessions, snapshot registry)            │
│                                                               │
│  terminal / file tools / execute_code / subagents             │
│        │                                                      │
│        │  tenki-sandbox SDK (control-plane token host-side)   │
└────────┼──────────────────────────────────────────────────────┘
         ▼
   Tenki API (api.tenki.cloud)
         │
┌────────▼──────────── Sandbox (replaceable) ───────────────────┐
│  /home/tenki · shell exec · file transfer · code execution    │
│  guest env = env_passthrough vars + tenki_forward_env entries │
└───────────────────────────────────────────────────────────────┘
```

A sandbox dying can never take the agent down. The worst case is one failed command; the next command transparently creates or restores a sandbox. The guest environment receives the generic `terminal.env_passthrough` allowlist (config-based and skill-registered) plus anything explicitly listed in `terminal.tenki_forward_env`. The supervisor's Tenki control-plane token is **not forwarded by default** — forwarding it is an explicit (and warned) `tenki_forward_env` opt-in for nested-sandbox workflows; see [Security](../user-guide/security.md) for the full credential boundary.

Everything below is implemented in `tools/environments/tenki.py`, with credential/workspace resolution in `tools/tenki_config.py` and host-side caching in `tools/terminal_tool.py`.

## Sandbox Lifecycle

| Tier | Mechanism | Survives |
|---|---|---|
| Within a session | cached environment, reused per command | idle gaps up to `lifetime_seconds`; sandbox lifetime capped by `tenki_max_duration` |
| Across sessions | `container_persistent: true` → durable snapshot at cleanup, restore on next session | process restarts, host reboots, sandbox loss |
| Durability fallback | pause instead of terminate when a snapshot cannot be confirmed durable | pause-retention window |

The first terminal/file/code call creates a sandbox on demand and caches the environment host-side. Subsequent commands reuse it, so shell state and the filesystem persist across the whole session. An idle reaper reclaims environments after `lifetime_seconds` of inactivity (environments with live background processes are kept alive), and `tenki_max_duration` caps any single sandbox's wall-clock life. If a sandbox ages out mid-session, the next command simply gets a fresh one.

With the default `container_persistent: false`, cleanup **terminates** the sandbox — ephemeral, nothing retained.

## Cross-Session Persistence

With `container_persistent: true`, the primary mechanism is **snapshot, then terminate**: cleanup takes a filesystem snapshot, waits for it to be confirmed durable, records the pointer in the profile's snapshot registry (`<HERMES_HOME>/tenki_snapshots.json`), and only then terminates the sandbox. The next session boots a new sandbox *from that snapshot* — persistent state does not depend on any particular sandbox staying alive.

The **durability gate** governs every edge of this flow:

- A pointer is only ever recorded for a snapshot confirmed durable. If the snapshot cannot be taken or confirmed — quota exhaustion is the common cause — cleanup **pauses** the sandbox instead of terminating it, preserving the live filesystem and the previous pointer. The next session rediscovers the paused sandbox by its deterministic name and resumes it. (`tenki_idle_timeout` server-side auto-pause and non-graceful process exits produce paused/live sandboxes the same rediscovery handles.)
- On restore, a *transient* failure (network, rate limit, ambiguous error) re-raises with the pointer intact, so a later attempt can still recover the state.
- Only a *confirmed-unrecoverable* failure — snapshot gone, backing image gone (both the old and new SDK spellings of the registry error are handled), or explicitly non-durable — discards the pointer and boots a clean base image.

State degradation is therefore always explicit, never silent: persistent state is either recovered, still recoverable, or deliberately reset.

Cancellation (an interrupted or timed-out command) prefers pause for a persistent sandbox but falls back to terminate if pause is unavailable or fails — unlike `cleanup()`, which never terminates un-snapshotted persistent state and instead leaves the sandbox live.

## Readiness

`wait=True` returns when the sandbox reaches RUNNING — the VM is up, but the image's workload may still be booting. `wait_for_runtime` holds the create call until the image's *declared template runtime* (entrypoint / `ready_when` probes) is actually ready, eliminating the race between create and first exec.

The backend requests it optimistically, with three provisos discovered against the live API:

- The kwarg is signature-gated on `Client.create` (not `Sandbox.create`, whose `(cls, **kwargs)` signature would defeat the gate). The older registry-error spelling is likewise kept as defensive back-compat, though no supported install runs a pre-0.4.0 SDK.
- Images with **no** declared template runtime — including the Tenki default image — reject the entire create with `wait_for_runtime requires a template runtime`. The backend retries once without the kwarg and latches per (endpoint, image), so later creates skip the doomed attempt.
- A genuine `TemplateRuntimeFailedError` (runtime declared but failed to start) does **not** terminate the sandbox server-side; the backend reaps the leaked sandbox best-effort, then retries once without the runtime gate — matching pre-gate behavior, where a broken runtime surfaced at first exec instead of blocking boot.

Create calls carry a real gRPC deadline on both the persistent and ephemeral paths (the server now holds the call through boot *and* runtime start). Resume has no runtime awareness in the SDK, so resumed sandboxes remain best-effort — `wait_ready` only confirms RUNNING.

## Multi-Profile Isolation

One multiplexing gateway process can serve many profiles, and nearly every task collapses to one shared execution container per backend. Isolation is enforced by a **profile token** (`hermes_constants.active_profile_token()`):

- Under multiplexing the token derives only from the per-turn `HERMES_HOME` contextvar. `HERMES_PROFILE` is deliberately ignored there — it is process-global and would collapse every profile onto one token.
- The host-side environment cache is keyed `"profile-token + container-id"`, in lockstep across the terminal tool, file tools, and `execute_code`, so profile B can never attach to profile A's live sandbox or the credentials baked into it.
- Identity is bound **at construction** — cleanup runs on bare threads and `atexit` handlers that have no contextvar, so teardown is driven purely off keys and tokens captured while the turn's context was live. The shutdown sweep skips no profile's keys. (Non-graceful exits bypass `atexit` entirely, and a persistent sandbox whose state is not yet snapshotted is *deliberately* left running rather than torn down.)
- Tenki sandbox names and reuse matching are namespaced by the token; the snapshot registry is isolated by living under each profile's own `HERMES_HOME`, bound at construction for the same cleanup-thread reason.

Two residual limits are documented rather than hidden: terminal *configuration* (backend, image, forward lists) is still bridged from the startup profile's environment, and the Docker backend's forwarded-env capture still reads process-global `os.environ`.

## Quotas and Observability

- `hermes status` (tenki backend) shows live workspace usage — active sessions, snapshots, storage, and other dimensions — via a bounded, timeout-guarded RPC that degrades to a single dim row on any failure. Counts follow Tenki's quota semantics: only resources holding live VMs count, so paused sandboxes do not appear in `Active sessions`.
- `hermes doctor` verifies SDK presence, auth, and workspace/project resolution.
- Watch the **snapshot quota** when running persistent workloads: it is what cross-session persistence spends. A full quota does not lose state — the durability gate pauses the sandbox instead — but it silently shifts persistence from snapshots onto pause retention.
- Delegated subagents each hold their own sandbox, so a wide fan-out multiplies `Active sessions` consumption accordingly.

## Ephemeral State

- **In-memory state** — running processes, exposed ports — does not survive a pause, snapshot restore, or recreate. This is true of any sandbox provider. Long-lived state belongs on the filesystem under `/home/tenki`, where the persistence tiers protect it.
- **Selected host files** can be synced into the sandbox (`tenki_sync_hermes_home`) and synced back on cleanup, for workloads that want `~/.hermes` artifacts to round-trip.
- Treat sandboxes as replaceable capacity, not pets: the agent's durable state — sessions, memory, snapshot registry, credentials — lives in `HERMES_HOME` on the host, and the sandbox's durable state lives in its filesystem under the persistence tiers.

## Source Map

| Concern | Where |
|---|---|
| Sandbox lifecycle, create/pause/resume, durability gate, readiness | `tools/environments/tenki.py` |
| Auth/workspace/project resolution (profile-scope-aware) | `tools/tenki_config.py` |
| Host-side environment cache, idle reaper, profile-scoped keys | `tools/terminal_tool.py` (shared with `tools/file_tools.py`, `tools/code_execution_tool.py`) |
| Profile token | `hermes_constants.active_profile_token()` |
| Status/doctor surfaces | `hermes_cli/status.py`, `hermes_cli/doctor.py` |

## Related Docs

- [Configuration](../user-guide/configuration.md) — all `tenki_*` keys (see also `cli-config.yaml.example` OPTION 7)
- [Security](../user-guide/security.md) — credential boundaries and env forwarding
- [Tools Runtime](./tools-runtime.md) — terminal environments and tool dispatch
