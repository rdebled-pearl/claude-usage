# Claude Code usage dashboard

A local, single-user tool that ingests your Claude Code session transcripts into
a SQLite database and serves an interactive Streamlit dashboard over your token
usage, cost, PRs, and trends.

## Components

| File | Role |
|------|------|
| `ingest.py` | Reads `~/.claude/projects/**` transcripts into `usage.db`. Stdlib-only; run on a schedule by launchd. |
| `dashboard.py` | Streamlit app: Overview, Explore, Insights, PRs, Trends, Tools, and an "Ask" tab (Claude API or Claude Code login). |
| `ask_mcp_server.py` | Read-only SQL tool for the "Ask" tab, used directly by the API backend and as a stdio MCP server by the Claude Code backend. Stdlib-only. |
| `run_dashboard.sh` | Launches the Streamlit server (headless, port 8501). |
| `bin/claude-usage` | User command: start/open the dashboard, plus `--status` / `--stop` / `--update` / `--help`. |
| `requirements.txt` | Pinned dashboard dependencies, synced on every install/update. |
| `install.sh` | Installs the program, schedules ingest via launchd, and adds the `claude-usage` command. |
| `uninstall.sh` | Removes the launchd job, command, and (with `--purge`) the program and data. |

## Install

```bash
./install.sh
```

This:
- copies the program to `~/.local/share/claude-usage/` (code + venv),
- keeps mutable data (`usage.db`, logs) in `~/.local/state/claude-usage/`,
- schedules the ingest job (every 30 minutes) via launchd -- launchd polls
  `ingest.py --every-minutes 30` each minute and ingest runs only when due, so
  the dashboard's refresh button (top right, beside the countdown) resets the
  timer,
- installs a `claude-usage` command at `~/.local/bin/claude-usage`,
- optionally raises Claude Code's `cleanupPeriodDays` so more history survives.

If `~/.local/bin` isn't on your `PATH`, the installer prints how to add it (or
set an alias).

## Usage

```bash
claude-usage            # start the server (if needed) and open the browser
claude-usage --status   # is the dashboard running?
claude-usage --stop     # stop the dashboard server
claude-usage --update   # pull latest code and reinstall
claude-usage --help
```

The server is a persistent local process; it keeps running until you `--stop`
it, uninstall, or reboot.

## Updates

`claude-usage --update` pulls the latest code from the source checkout you
installed from (recorded at install time) and re-runs `install.sh`, which:
- copies the new code into the install dir,
- syncs dependencies from the pinned `requirements.txt` (so bumped packages are
  actually picked up),
- restarts the ingest job and stops any running dashboard so it reloads.

Your `usage.db` and logs in the data dir are left untouched. If you'd rather do
it by hand: `git pull` in your checkout, then `./install.sh`.

The installer tracks the exact set of shipped files (a manifest) and prunes any
left behind by a previous version that renamed or removed a file, so the install
dir never accumulates stale code.

### Database schema versioning

`usage.db` carries a schema version in SQLite's `PRAGMA user_version`. On every
ingest run, `init_db()` upgrades an existing DB through ordered migrations up to
the code's `SCHEMA_VERSION`; fresh DBs are created at the latest shape directly.
A DB newer than the running code is detected and refused rather than corrupted.
To evolve the schema: update `SCHEMA`, add a `_migrate_to_N()` function, register
it in `MIGRATIONS`, and bump `SCHEMA_VERSION` (see the comments in `ingest.py`).

## The "Tools" tab

Transcripts don't bill tool calls separately -- a tool's result is paid for as
input on the turns that follow it -- so tool cost is estimated:

- **Cost & context by tool** (`tool_calls` table, one row per call): each call's
  estimated cost = writing its input (output tokens on the issuing turn) + the
  cache write of its result on the next turn + re-reading that result from cache
  on every later turn in the same context until a compaction (`context_resets`).
  Token counts are estimated from result size (chars / 4). Recomputed after every
  ingest, since re-read cost grows while a session continues. Grouped by tool,
  MCP server, or built-in vs MCP, with a cost/tokens toggle.
- **Turns attributed to skills & MCP tools** (`usage_events.attribution_*`):
  Claude Code tags a turn with the active skill, or the MCP tool whose result it
  was reading. Only recent Claude Code versions record this, and a turn's cost
  includes re-reading the whole context, so treat these totals as relative, not
  marginal.

The same tool stats appear per PR in the PRs tab's lookup, and the Overview cost
chart overlays estimated tool cost as a dashed line.

These came with schema v2 (tool calls, attribution) and v3 (cost estimates,
compaction points); each migration rewinds ingest's per-file offsets so the next
run backfills from existing transcripts.

## The "Ask" tab

Answers questions by letting Claude run read-only SQL against `usage.db`. Two
backends, picked automatically:

- **Anthropic API** (preferred) -- used when `ANTHROPIC_API_KEY` is set. Calls the
  real Claude API (small cost, not logged in `usage.db`).
- **Claude Code login** (fallback) -- with no key, it shells out to the `claude`
  CLI (`claude -p`) using whatever that CLI is logged in with, e.g. a Claude
  Enterprise seat. The CLI is sandboxed to a single read-only `run_sql` tool
  (served by `ask_mcp_server.py`; all built-in tools disabled) and runs with
  `--no-session-persistence`, so Ask's own turns don't end up in your usage
  history. `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` are stripped from its
  environment so it can't silently bill a key instead. The tab checks
  `claude auth status` first and shows a specific error (CLI missing, not
  logged in, login rejected, model unavailable, usage limit, timeout, tool
  server failed) rather than a generic failure. Slower than the API since each
  question starts a CLI process, which is why the tab suggests setting a key.

## Configuration

- `CLAUDE_USAGE_DATA_DIR` — override where `usage.db` and logs live.
- `CLAUDE_USAGE_INSTALL_DIR` — override where the program is installed.

Both default per the resolution described in `install.sh`; a pre-existing
`~/.claude-usage/usage.db` is detected and reused (and migrated on reinstall).

## Uninstall

```bash
./uninstall.sh            # remove job + command, keep program and data
./uninstall.sh --purge    # also delete the program and your usage history
```
