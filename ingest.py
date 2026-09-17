#!/usr/bin/env python3
import fcntl
import glob
import json
import os
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, timezone

def _data_dir():
    """Resolve where usage.db/locks/logs live, kept separate from the
    installed code so reinstalls never clobber collected history."""
    env = os.environ.get("CLAUDE_USAGE_DATA_DIR")
    if env:
        return os.path.expanduser(env)
    legacy = os.path.expanduser("~/.claude-usage")
    xdg = os.path.expanduser("~/.local/state/claude-usage")
    if os.path.exists(os.path.join(legacy, "usage.db")) and not os.path.exists(
        os.path.join(xdg, "usage.db")
    ):
        return legacy
    return xdg


BASE_DIR = _data_dir()
os.makedirs(BASE_DIR, exist_ok=True)
DB_PATH = os.path.join(BASE_DIR, "usage.db")
LOCK_PATH = os.path.join(BASE_DIR, "ingest.lock")
LOG_PATH = os.path.join(BASE_DIR, "ingest.log")
PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
PR_LOOKUP_LIMIT_PER_RUN = 50
PR_CACHE_TTL_SECONDS = 24 * 3600

# USD per million tokens, by model family and the date each rate took effect.
# Cache read/write are multiples of the base input rate: read = 0.1x input,
# write = 1.25x input @5m TTL / 2x input @1h TTL.
#
# Cost is computed and stored PER ROW at ingest time using the rate in effect
# on that row's date, so historical rows keep the price actually billed even
# after a rate changes here. When a price changes, APPEND a new
# (model_prefix, effective_from, input, output) entry; never edit/remove an
# old one.
PRICING_HISTORY = [
    # (model_prefix, effective_from, input_$_per_mtok, output_$_per_mtok)
    ("claude-opus-5", "2000-01-01", 5.00, 25.00),
    ("claude-sonnet-5", "2000-01-01", 2.00, 10.00),
    ("claude-sonnet-4-6", "2000-01-01", 3.00, 15.00),
    ("claude-haiku-4-5", "2000-01-01", 1.00, 5.00),
]
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0


def resolve_price(model, event_date):
    """Latest (input, output) rate for `model` with effective_from <= event_date."""
    if not model:
        return (None, None)
    best = None
    for prefix, effective_from, in_price, out_price in PRICING_HISTORY:
        if model.startswith(prefix) and effective_from <= event_date:
            if best is None or effective_from > best[0]:
                best = (effective_from, in_price, out_price)
    if best is None:
        return (None, None)
    return (best[1], best[2])


def compute_costs(input_price, output_price, input_tokens, output_tokens,
                   cache_5m, cache_1h, cache_read):
    if input_price is None or output_price is None:
        return (None, None, None, None, None)
    input_cost = input_tokens * input_price / 1e6
    output_cost = output_tokens * output_price / 1e6
    cache_write_cost = (
        cache_5m * input_price * CACHE_WRITE_5M_MULTIPLIER
        + cache_1h * input_price * CACHE_WRITE_1H_MULTIPLIER
    ) / 1e6
    cache_read_cost = cache_read * input_price * CACHE_READ_MULTIPLIER / 1e6
    total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost
    return (input_cost, output_cost, cache_write_cost, cache_read_cost, total_cost)

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    date TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model TEXT,
    repo TEXT,
    branch TEXT,
    cwd TEXT,
    pr_number INTEGER,
    pr_url TEXT,
    is_subagent INTEGER NOT NULL DEFAULT 0,
    agent_id TEXT,
    agent_type TEXT,
    agent_description TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_input_tokens INTEGER DEFAULT 0,
    cache_creation_input_tokens INTEGER DEFAULT 0,
    cache_creation_5m_tokens INTEGER DEFAULT 0,
    cache_creation_1h_tokens INTEGER DEFAULT 0,
    thinking_tokens INTEGER DEFAULT 0,
    input_price_per_mtok REAL,
    output_price_per_mtok REAL,
    input_cost REAL,
    output_cost REAL,
    cache_write_cost REAL,
    cache_read_cost REAL,
    total_cost REAL
);
CREATE INDEX IF NOT EXISTS idx_usage_events_date ON usage_events(date);
CREATE INDEX IF NOT EXISTS idx_usage_events_repo ON usage_events(repo);

CREATE TABLE IF NOT EXISTS ingest_state (
    file_path TEXT PRIMARY KEY,
    byte_offset INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS repo_cache (
    cwd TEXT PRIMARY KEY,
    repo TEXT
);

CREATE TABLE IF NOT EXISTS pr_cache (
    repo TEXT NOT NULL,
    branch TEXT NOT NULL,
    pr_number INTEGER,
    pr_url TEXT,
    resolved_at REAL NOT NULL,
    PRIMARY KEY (repo, branch)
);
"""


# Schema versioning: SCHEMA always reflects the latest shape (CREATE ... IF
# NOT EXISTS). Ordered migrations below bring an existing older DB up to
# SCHEMA_VERSION, tracked via SQLite's PRAGMA user_version. To evolve the
# schema: edit SCHEMA, add a _migrate_to_N(conn), register it in MIGRATIONS,
# and bump SCHEMA_VERSION.


def _add_columns(conn, table, columns):
    """Add any of `columns` (name -> DDL type) not already present."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, ddl_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}")


def _migrate_to_1(conn):
    """Baseline: subagent columns added after the very first schema shipped."""
    _add_columns(conn, "usage_events", {
        "is_subagent": "INTEGER NOT NULL DEFAULT 0",
        "agent_id": "TEXT",
        "agent_type": "TEXT",
        "agent_description": "TEXT",
    })


# version number -> function that upgrades a DB from (version-1) to (version).
MIGRATIONS = {
    1: _migrate_to_1,
}
SCHEMA_VERSION = max(MIGRATIONS)


def _db_has_tables(conn):
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='usage_events'"
    ).fetchone()
    return row[0] > 0


def init_db(conn):
    """Create/upgrade the schema and stamp the version. Fresh DBs stamp
    straight to SCHEMA_VERSION; existing DBs run pending migrations in order."""
    fresh = not _db_has_tables(conn)
    conn.executescript(SCHEMA)

    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if fresh:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        return

    if current > SCHEMA_VERSION:
        raise SystemExit(
            f"usage.db is at schema version {current}, but this code only knows "
            f"up to {SCHEMA_VERSION}. Update the application before running ingest."
        )

    for version in range(current + 1, SCHEMA_VERSION + 1):
        log(f"migrating usage.db schema {version - 1} -> {version}")
        MIGRATIONS[version](conn)
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()


def log(msg):
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def acquire_lock():
    fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another ingest run holds the lock, exiting")
        sys.exit(0)
    return fd


WORKTREE_MARKERS = ("/.claude/worktrees/", "/.worktrees/")


def resolve_repo(conn, cwd):
    if not cwd:
        return None
    row = conn.execute("SELECT repo FROM repo_cache WHERE cwd = ?", (cwd,)).fetchone()
    if row:
        return row[0]

    repo = None
    # Worktree sessions often outlive the worktree itself, so `git -C <cwd>`
    # fails -- fall back to the path segment before the worktrees marker.
    for marker in WORKTREE_MARKERS:
        if marker in cwd:
            repo = cwd.split(marker, 1)[0].rstrip("/").rsplit("/", 1)[-1]
            break

    if not repo:
        try:
            url = subprocess.run(
                ["git", "-C", cwd, "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5,
            )
            if url.returncode == 0 and url.stdout.strip():
                name = url.stdout.strip().rstrip("/")
                name = name.rsplit("/", 1)[-1]
                if name.endswith(".git"):
                    name = name[:-4]
                repo = name
        except (subprocess.SubprocessError, OSError):
            pass
    if not repo:
        try:
            top = subprocess.run(
                ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
            )
            if top.returncode == 0 and top.stdout.strip():
                repo = os.path.basename(top.stdout.strip())
        except (subprocess.SubprocessError, OSError):
            pass
    if not repo and cwd.rstrip("/") != os.path.expanduser("~").rstrip("/"):
        repo = os.path.basename(cwd.rstrip("/")) or None
    conn.execute("INSERT OR REPLACE INTO repo_cache (cwd, repo) VALUES (?, ?)", (cwd, repo))
    return repo


def session_files():
    return glob.glob(os.path.join(PROJECTS_DIR, "*", "*.jsonl"))


def subagent_files():
    # <project>/<session_id>/subagents/agent-<agentId>.jsonl; sidecar
    # agent-<agentId>.meta.json carries agentType/description.
    return glob.glob(os.path.join(PROJECTS_DIR, "*", "*", "subagents", "*.jsonl"))


def load_agent_meta(jsonl_path):
    meta_path = os.path.splitext(jsonl_path)[0] + ".meta.json"
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def ingest_file(conn, path, is_subagent=False, agent_type=None, agent_description=None):
    row = conn.execute(
        "SELECT byte_offset FROM ingest_state WHERE file_path = ?", (path,)
    ).fetchone()
    offset = row[0] if row else 0
    try:
        size = os.path.getsize(path)
    except OSError:
        return 0
    if size < offset:
        offset = 0  # rotated/truncated; restart from the top

    inserted = 0
    new_offset = offset
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        while True:
            raw_line = f.readline()
            if not raw_line:
                break
            line_end = f.tell()
            line = raw_line.strip()
            if not line:
                new_offset = line_end
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                break  # incomplete trailing line (write in progress)
            new_offset = line_end

            if obj.get("type") != "assistant":
                continue
            message = obj.get("message") or {}
            usage = message.get("usage")
            message_id = message.get("id")
            if not usage or not message_id:
                continue
            ts = obj.get("timestamp")
            try:
                dt_utc = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            local_date = dt_utc.astimezone().date().isoformat()
            cwd = obj.get("cwd")
            repo = resolve_repo(conn, cwd)
            branch = obj.get("gitBranch")
            thinking_tokens = (usage.get("output_tokens_details") or {}).get(
                "thinking_tokens", 0
            )
            cache_creation = usage.get("cache_creation") or {}
            cache_5m = cache_creation.get("ephemeral_5m_input_tokens", 0)
            cache_1h = cache_creation.get("ephemeral_1h_input_tokens", 0)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            model = message.get("model")

            input_price, output_price = resolve_price(model, local_date)
            (input_cost, output_cost, cache_write_cost, cache_read_cost,
             total_cost) = compute_costs(
                input_price, output_price, input_tokens, output_tokens,
                cache_5m, cache_1h, cache_read,
            )

            agent_id = obj.get("agentId") if is_subagent else None

            before = conn.total_changes
            conn.execute(
                """INSERT OR IGNORE INTO usage_events
                   (message_id, session_id, date, timestamp, model, repo, branch, cwd,
                    is_subagent, agent_id, agent_type, agent_description,
                    input_tokens, output_tokens, cache_read_input_tokens,
                    cache_creation_input_tokens, cache_creation_5m_tokens,
                    cache_creation_1h_tokens, thinking_tokens,
                    input_price_per_mtok, output_price_per_mtok,
                    input_cost, output_cost, cache_write_cost, cache_read_cost, total_cost)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    message_id,
                    obj.get("sessionId"),
                    local_date,
                    ts,
                    model,
                    repo,
                    branch,
                    cwd,
                    1 if is_subagent else 0,
                    agent_id,
                    agent_type,
                    agent_description,
                    input_tokens,
                    output_tokens,
                    cache_read,
                    usage.get("cache_creation_input_tokens", 0),
                    cache_5m,
                    cache_1h,
                    thinking_tokens,
                    input_price,
                    output_price,
                    input_cost,
                    output_cost,
                    cache_write_cost,
                    cache_read_cost,
                    total_cost,
                ),
            )
            if conn.total_changes > before:
                inserted += 1

    conn.execute(
        """INSERT INTO ingest_state (file_path, byte_offset) VALUES (?, ?)
           ON CONFLICT(file_path) DO UPDATE SET byte_offset = excluded.byte_offset""",
        (path, new_offset),
    )
    return inserted


def lookup_pr(cwd, branch):
    """(number, url) for the PR on this branch, via `gh` -- more reliable
    than reconstructing owner/repo from the local path."""
    if not cwd or not os.path.isdir(cwd):
        return None, None
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--head", branch, "--json", "number,url",
             "--state", "all", "--limit", "1"],
            cwd=cwd, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return None, None
        data = json.loads(result.stdout or "[]")
        if not data:
            return None, None
        return data[0]["number"], data[0]["url"]
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError, KeyError, IndexError):
        return None, None


def enrich_prs(conn):
    now = time.time()
    rows = conn.execute(
        """SELECT DISTINCT repo, branch, cwd FROM usage_events
           WHERE pr_number IS NULL AND branch IS NOT NULL AND branch != 'HEAD'"""
    ).fetchall()
    checked = 0
    for repo, branch, cwd in rows:
        if checked >= PR_LOOKUP_LIMIT_PER_RUN:
            break
        cache_row = conn.execute(
            "SELECT pr_number, pr_url, resolved_at FROM pr_cache WHERE repo = ? AND branch = ?",
            (repo, branch),
        ).fetchone()
        if cache_row and (now - cache_row[2]) < PR_CACHE_TTL_SECONDS:
            pr_number, pr_url = cache_row[0], cache_row[1]
        else:
            pr_number, pr_url = lookup_pr(cwd, branch)
            checked += 1
            conn.execute(
                """INSERT OR REPLACE INTO pr_cache (repo, branch, pr_number, pr_url, resolved_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (repo, branch, pr_number, pr_url, now),
            )
        if pr_number is not None:
            conn.execute(
                """UPDATE usage_events SET pr_number = ?, pr_url = ?
                   WHERE repo = ? AND branch = ? AND pr_number IS NULL""",
                (pr_number, pr_url, repo, branch),
            )


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    lock_fd = acquire_lock()
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    total_inserted = 0
    files_seen = 0
    for path in session_files():
        files_seen += 1
        total_inserted += ingest_file(conn, path)
    for path in subagent_files():
        files_seen += 1
        meta = load_agent_meta(path)
        total_inserted += ingest_file(
            conn, path, is_subagent=True,
            agent_type=meta.get("agentType"), agent_description=meta.get("description"),
        )
    conn.commit()

    enrich_prs(conn)
    conn.commit()
    conn.close()

    log(f"ingest complete: {files_seen} files scanned, {total_inserted} new rows")


if __name__ == "__main__":
    main()
