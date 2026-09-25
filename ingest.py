#!/usr/bin/env python3
import fcntl
import glob
import json
import os
import shutil
import subprocess
import sqlite3
import sys
import time
from datetime import datetime, timezone


def _find_gh():
    """Absolute path to the `gh` CLI. launchd runs with a minimal PATH that
    excludes Homebrew, so relying on bare `gh` breaks scheduled ingests --
    resolve it explicitly against PATH plus common install locations."""
    found = shutil.which("gh")
    if found:
        return found
    for candidate in (
        "/opt/homebrew/bin/gh",
        "/usr/local/bin/gh",
        "/home/linuxbrew/.linuxbrew/bin/gh",
        os.path.expanduser("~/.local/bin/gh"),
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


GH_BIN = _find_gh()

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
PROGRESS_PATH = os.path.join(BASE_DIR, "ingest.progress")
# {"last_run": epoch secs, "interval_minutes": int}. ingest owns the schedule
# (launchd just polls with --every-minutes), so any run -- scheduled, manual,
# or from the dashboard -- resets the countdown the dashboard shows.
SCHEDULE_PATH = os.path.join(BASE_DIR, "ingest.schedule.json")
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
    total_cost REAL,
    attribution_mcp_server TEXT,
    attribution_mcp_tool TEXT,
    attribution_skill TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_events_date ON usage_events(date);
CREATE INDEX IF NOT EXISTS idx_usage_events_repo ON usage_events(repo);

-- One row per tool invocation (tool_use block), with the size of what it
-- returned. A call's own cost isn't billed separately: its result becomes
-- input on the following turn(s), so result size is the per-call signal.
CREATE TABLE IF NOT EXISTS tool_calls (
    tool_use_id TEXT PRIMARY KEY,
    message_id TEXT,
    session_id TEXT,
    date TEXT,
    timestamp TEXT,
    model TEXT,
    repo TEXT,
    is_subagent INTEGER NOT NULL DEFAULT 0,
    tool_name TEXT NOT NULL,
    mcp_server TEXT,
    mcp_tool TEXT,
    skill TEXT,
    input_chars INTEGER DEFAULT 0,
    result_chars INTEGER,
    result_images INTEGER,
    result_tokens_est INTEGER,
    is_error INTEGER,
    agent_id TEXT,
    -- Estimated cost of the call, recomputed after every ingest (see
    -- compute_tool_costs): writing the tool input (output tokens), the cache
    -- write of the result on the next turn, and re-reading it on every later
    -- turn in the same context until a compaction.
    est_input_cost REAL,
    est_write_cost REAL,
    est_reread_cost REAL,
    est_cost REAL,
    rereads INTEGER
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_date ON tool_calls(date);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);

-- Compaction points (system/compact_boundary): the context is replaced by a
-- summary, so earlier tool results stop being re-read after this.
CREATE TABLE IF NOT EXISTS context_resets (
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    PRIMARY KEY (session_id, agent_id, timestamp)
);

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


def _migrate_to_2(conn):
    """Tool/skill attribution columns + the tool_calls table (created by
    SCHEMA). Rewinds every file's read offset so the next ingest re-reads
    history to backfill them; usage rows are deduped by message_id, so the
    re-read doesn't double-count."""
    _add_columns(conn, "usage_events", {
        "attribution_mcp_server": "TEXT",
        "attribution_mcp_tool": "TEXT",
        "attribution_skill": "TEXT",
    })
    conn.execute("UPDATE ingest_state SET byte_offset = 0")


def _migrate_to_3(conn):
    """Per-call cost estimate columns, the subagent a call ran in, and the
    context_resets table (created by SCHEMA). Rewinds offsets again so the
    re-read backfills agent_id and compaction points."""
    _add_columns(conn, "tool_calls", {
        "agent_id": "TEXT",
        "est_input_cost": "REAL",
        "est_write_cost": "REAL",
        "est_reread_cost": "REAL",
        "est_cost": "REAL",
        "rereads": "INTEGER",
    })
    conn.execute("UPDATE ingest_state SET byte_offset = 0")


# version number -> function that upgrades a DB from (version-1) to (version).
MIGRATIONS = {
    1: _migrate_to_1,
    2: _migrate_to_2,
    3: _migrate_to_3,
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


# Rough chars-per-token for tool output (mixed prose/code/JSON). Transcripts
# don't carry per-result token counts, so result_tokens_est is an estimate.
CHARS_PER_TOKEN = 4


def split_mcp_name(tool_name):
    """'mcp__<server>__<tool>' -> (server, tool); (None, None) otherwise."""
    if not tool_name or not tool_name.startswith("mcp__"):
        return None, None
    server, sep, tool = tool_name[len("mcp__"):].partition("__")
    return (server, tool) if sep else (server, None)


def measure_tool_result(content):
    """(text_chars, image_count) for a tool_result's content, which is a
    string or a list of text/image/tool_reference blocks."""
    if isinstance(content, str):
        return len(content), 0
    chars = images = 0
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            chars += len(block.get("text") or "")
        elif block.get("type") == "image":
            images += 1
        else:
            chars += len(json.dumps(block))
    return chars, images


def record_tool_uses(conn, obj, message, local_date, repo, is_subagent):
    for block in message.get("content") or []:
        if not isinstance(block, dict) or block.get("type") != "tool_use" or not block.get("id"):
            continue
        name = block.get("name") or ""
        tool_input = block.get("input") if isinstance(block.get("input"), dict) else {}
        server, tool = split_mcp_name(name)
        agent_id = obj.get("agentId") if is_subagent else None
        conn.execute(
            """INSERT OR IGNORE INTO tool_calls
               (tool_use_id, message_id, session_id, date, timestamp, model, repo,
                is_subagent, tool_name, mcp_server, mcp_tool, skill, input_chars, agent_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                block["id"], message.get("id"), obj.get("sessionId"), local_date,
                obj.get("timestamp"), message.get("model"), repo,
                1 if is_subagent else 0, name, server, tool,
                tool_input.get("skill") if name == "Skill" else None,
                len(json.dumps(tool_input)), agent_id,
            ),
        )
        # Rows from before agent_id existed: fill it in on the backfill re-read.
        if agent_id:
            conn.execute(
                "UPDATE tool_calls SET agent_id = ? WHERE tool_use_id = ? AND agent_id IS NULL AND is_subagent = 1",
                (agent_id, block["id"]),
            )


def record_context_reset(conn, obj, is_subagent):
    if obj.get("sessionId") and obj.get("timestamp"):
        conn.execute(
            "INSERT OR IGNORE INTO context_resets (session_id, agent_id, timestamp) VALUES (?, ?, ?)",
            (obj["sessionId"], (obj.get("agentId") or "") if is_subagent else "", obj["timestamp"]),
        )


def compute_tool_costs(conn):
    """Estimate each tool call's cost and store it on tool_calls.

    A call isn't billed on its own. Its result becomes new input on the next
    turn in the same context (a cache write at that turn's rate: 2x input for
    the 1h cache, 1.25x for 5m, 1x if uncached), and is then re-read from
    cache (0.1x input) on every later turn until a compaction replaces the
    context. The call's input was generated as output on the issuing turn.
    All token counts are the chars/CHARS_PER_TOKEN estimates. Recomputed in
    full each run (cheap) because re-read cost keeps growing while a session
    continues.
    """
    from bisect import bisect_right

    resets = {}
    for session_id, agent_id, ts in conn.execute(
        "SELECT session_id, agent_id, timestamp FROM context_resets ORDER BY timestamp"
    ):
        resets.setdefault((session_id, agent_id), []).append(ts)

    # Per context: turns in time order, with the segment (compactions so far)
    # each falls in and the suffix sum of re-read rates within its segment.
    contexts = {}
    output_price = {}
    for (session_id, agent_id, ts, message_id, in_price, out_price, c1h, c5m) in conn.execute(
        """SELECT session_id, COALESCE(agent_id, ''), timestamp, message_id,
                  input_price_per_mtok, output_price_per_mtok,
                  cache_creation_1h_tokens, cache_creation_5m_tokens
           FROM usage_events ORDER BY timestamp"""
    ):
        output_price[message_id] = out_price
        ctx = contexts.setdefault((session_id, agent_id), {"ts": [], "turns": []})
        ctx["ts"].append(ts)
        write_mult = (CACHE_WRITE_1H_MULTIPLIER if c1h else
                      CACHE_WRITE_5M_MULTIPLIER if c5m else 1.0)
        ctx["turns"].append([in_price, write_mult, 0, 0.0, 0])  # price, mult, seg, suffix, count
    for key, ctx in contexts.items():
        ctx_resets = resets.get(key, [])
        turns = ctx["turns"]
        for i, ts in enumerate(ctx["ts"]):
            turns[i][2] = bisect_right(ctx_resets, ts)
        suffix, count, seg = 0.0, 0, None
        for turn in reversed(turns):
            if turn[2] != seg:
                suffix, count, seg = 0.0, 0, turn[2]
            if turn[0] is not None:
                suffix += turn[0] * CACHE_READ_MULTIPLIER
                count += 1
            turn[3], turn[4] = suffix, count

    updates = []
    for (tool_use_id, session_id, agent_id, ts, tokens, input_chars, message_id) in conn.execute(
        """SELECT tool_use_id, session_id, COALESCE(agent_id, ''), timestamp,
                  result_tokens_est, input_chars, message_id FROM tool_calls"""
    ).fetchall():
        out_price = output_price.get(message_id)
        input_cost = (
            -(-(input_chars or 0) // CHARS_PER_TOKEN) * out_price / 1e6
            if out_price is not None else None
        )
        write_cost = reread_cost = None
        rereads = 0
        ctx = contexts.get((session_id, agent_id))
        if tokens is not None and ctx and ts:
            ctx_resets = resets.get((session_id, agent_id), [])
            call_seg = bisect_right(ctx_resets, ts)
            j = bisect_right(ctx["ts"], ts)  # first turn after the call = reads its result
            write_cost = reread_cost = 0.0
            if j < len(ctx["turns"]) and ctx["turns"][j][2] == call_seg and ctx["turns"][j][0] is not None:
                price, mult = ctx["turns"][j][0], ctx["turns"][j][1]
                write_cost = tokens * price * mult / 1e6
                if j + 1 < len(ctx["turns"]) and ctx["turns"][j + 1][2] == call_seg:
                    reread_cost = tokens * ctx["turns"][j + 1][3] / 1e6
                    rereads = ctx["turns"][j + 1][4]
        parts = [c for c in (input_cost, write_cost, reread_cost) if c is not None]
        updates.append((
            input_cost, write_cost, reread_cost, sum(parts) if parts else None, rereads,
            tool_use_id,
        ))
    conn.executemany(
        """UPDATE tool_calls SET est_input_cost = ?, est_write_cost = ?,
               est_reread_cost = ?, est_cost = ?, rereads = ? WHERE tool_use_id = ?""",
        updates,
    )


def record_tool_results(conn, obj):
    """Attach result size/error to the matching tool_calls row. Results always
    follow their tool_use in the transcript, so the row already exists (even
    when the two land in different ingest runs)."""
    content = (obj.get("message") or {}).get("content")
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        chars, images = measure_tool_result(block.get("content"))
        conn.execute(
            """UPDATE tool_calls SET result_chars = ?, result_images = ?,
                   result_tokens_est = ?, is_error = ?
               WHERE tool_use_id = ?""",
            (chars, images, -(-chars // CHARS_PER_TOKEN),
             1 if block.get("is_error") else 0, block.get("tool_use_id")),
        )


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

            if obj.get("type") == "user":
                record_tool_results(conn, obj)
                continue
            if obj.get("type") == "system" and obj.get("subtype") == "compact_boundary":
                record_context_reset(conn, obj, is_subagent)
                continue
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

            # A turn tagged by Claude Code as driven by an MCP tool (the turn
            # reading that tool's result) or by an active skill. Filled in
            # separately so re-reading history backfills rows ingested before
            # these columns existed.
            attribution = (obj.get("attributionMcpServer"), obj.get("attributionMcpTool"),
                           obj.get("attributionSkill"))
            if any(attribution):
                conn.execute(
                    """UPDATE usage_events SET attribution_mcp_server = ?,
                           attribution_mcp_tool = ?, attribution_skill = ?
                       WHERE message_id = ? AND attribution_mcp_server IS NULL
                         AND attribution_mcp_tool IS NULL AND attribution_skill IS NULL""",
                    (*attribution, message_id),
                )
            record_tool_uses(conn, obj, message, local_date, repo, is_subagent)

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
    if not GH_BIN:
        return None, None
    try:
        result = subprocess.run(
            [GH_BIN, "pr", "list", "--head", branch, "--json", "number,url",
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


def write_progress(fraction, label, done=False):
    """Report ingest progress to a small JSON file the dashboard polls. Written
    atomically (tmp + rename) so a concurrent read never sees a partial line."""
    try:
        tmp = PROGRESS_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(
                {"fraction": max(0.0, min(1.0, fraction)), "label": label,
                 "done": done, "ts": time.time()},
                fh,
            )
        os.replace(tmp, PROGRESS_PATH)
    except OSError:
        pass


def enrich_prs(conn, progress=None):
    now = time.time()
    if not GH_BIN:
        log("warning: `gh` CLI not found; skipping PR resolution")
        return
    rows = conn.execute(
        """SELECT DISTINCT repo, branch, cwd FROM usage_events
           WHERE pr_number IS NULL AND branch IS NOT NULL AND branch != 'HEAD'"""
    ).fetchall()
    total = len(rows) or 1
    checked = 0
    for idx, (repo, branch, cwd) in enumerate(rows):
        if progress:
            progress(0.82 + 0.18 * (idx + 1) / total,
                     f"Resolving pull requests ({idx + 1}/{total})")
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


def read_schedule():
    try:
        with open(SCHEDULE_PATH) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_schedule(**updates):
    """Merge `updates` into the schedule file atomically (tmp + rename)."""
    data = {**read_schedule(), **updates}
    tmp = SCHEDULE_PATH + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, SCHEDULE_PATH)
    except OSError:
        pass


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Ingest Claude Code usage into usage.db.")
    parser.add_argument(
        "--every-minutes", type=int, metavar="N",
        help="Scheduled mode: exit without doing anything unless N minutes "
             "have passed since the last run. launchd polls with this.",
    )
    return parser.parse_args(argv)


def ingest_is_due(every_minutes):
    """Record the configured interval and report whether a scheduled run
    should go ahead now."""
    schedule = read_schedule()
    if schedule.get("interval_minutes") != every_minutes:
        write_schedule(interval_minutes=every_minutes)
    last_run = schedule.get("last_run")
    if not isinstance(last_run, (int, float)):
        return True
    # A clock that jumped backwards shouldn't stall ingest indefinitely.
    elapsed = time.time() - last_run
    return elapsed < 0 or elapsed >= every_minutes * 60


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    os.makedirs(BASE_DIR, exist_ok=True)
    if args.every_minutes is not None and not ingest_is_due(args.every_minutes):
        return
    lock_fd = acquire_lock()
    # Stamped at start (not completion) so a run that keeps failing retries
    # on the normal cadence rather than on every poll.
    write_schedule(last_run=time.time())
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    total_inserted = 0
    files_seen = 0
    sessions = list(session_files())
    subagents = list(subagent_files())
    total_files = len(sessions) + len(subagents) or 1
    write_progress(0.0, "Scanning Claude Code transcripts\u2026")
    for path in sessions:
        files_seen += 1
        total_inserted += ingest_file(conn, path)
        write_progress(0.8 * files_seen / total_files,
                       f"Parsing transcripts ({files_seen}/{total_files})")
    for path in subagents:
        files_seen += 1
        meta = load_agent_meta(path)
        total_inserted += ingest_file(
            conn, path, is_subagent=True,
            agent_type=meta.get("agentType"), agent_description=meta.get("description"),
        )
        write_progress(0.8 * files_seen / total_files,
                       f"Parsing transcripts ({files_seen}/{total_files})")
    conn.commit()

    write_progress(0.81, "Estimating tool costs\u2026")
    compute_tool_costs(conn)
    conn.commit()

    write_progress(0.82, "Resolving pull requests\u2026")
    enrich_prs(conn, progress=write_progress)
    conn.commit()
    conn.close()

    write_progress(1.0, "Finalizing\u2026", done=True)
    log(f"ingest complete: {files_seen} files scanned, {total_inserted} new rows")


if __name__ == "__main__":
    main()
