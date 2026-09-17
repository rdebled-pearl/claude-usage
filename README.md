# Claude Code usage dashboard

A local, single-user tool that ingests your Claude Code session transcripts into
a SQLite database and serves an interactive Streamlit dashboard over your token
usage, cost, PRs, and trends.

## Components

| File | Role |
|------|------|
| `ingest.py` | Reads `~/.claude/projects/**` transcripts into `usage.db`. Stdlib-only; run on a schedule by launchd. |
| `dashboard.py` | Streamlit app: Overview, Explore, Insights, PRs, Trends, and an "Ask" tab (Claude API). |
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
- schedules the ingest job (every 4h) via launchd,
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

## The "Ask" tab

Requires an `ANTHROPIC_API_KEY` in the environment. Without it, that tab shows a
message instead of the input box. It calls the real Claude API (small cost, not
logged in `usage.db`).

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
