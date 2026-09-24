#!/usr/bin/env python3
"""Read-only SQL access to usage.db, shared by both "Ask" backends.

- The dashboard imports `run_sql_query` for the Anthropic API tool runner.
- Run as a script, this is a minimal stdio MCP server exposing the same
  query as a `run_sql` tool, which the Claude Code CLI backend loads via
  --mcp-config. Stdlib-only and hand-rolled (newline-delimited JSON-RPC) so
  it needs no extra dependency.

Usage: ask_mcp_server.py <path to usage.db>
"""
import json
import sqlite3
import sys

MAX_ROWS = 500

# Column list mirrors ingest.py's usage_events schema exactly -- keep in sync
# if that schema changes, since this is Claude's only description of the
# table (it never sees the DB schema directly, only this string).
USAGE_EVENTS_SCHEMA = """
Table usage_events (one row per Claude Code assistant turn):
  message_id TEXT primary key, session_id TEXT, date TEXT ('YYYY-MM-DD', local day),
  timestamp TEXT (raw UTC ISO8601), model TEXT, repo TEXT, branch TEXT, cwd TEXT,
  pr_number INTEGER (nullable), pr_url TEXT (nullable),
  input_tokens INTEGER, output_tokens INTEGER,
  cache_read_input_tokens INTEGER, cache_creation_input_tokens INTEGER,
  cache_creation_5m_tokens INTEGER, cache_creation_1h_tokens INTEGER,
  thinking_tokens INTEGER,
  input_price_per_mtok REAL, output_price_per_mtok REAL,
  input_cost REAL, output_cost REAL, cache_write_cost REAL, cache_read_cost REAL,
  total_cost REAL (sum of the four cost columns; the number to use for "cost" questions)
"""

RUN_SQL_DESCRIPTION = (
    "Run a read-only SQL query against the usage_events table and return the "
    "results as JSON. Pass a single SELECT (or WITH ... SELECT) statement; no "
    "INSERT/UPDATE/DELETE/DDL and no multiple statements."
)

_SQL_FORBIDDEN_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "attach", "detach",
    "pragma", "create", "replace", "vacuum", "reindex",
)


def run_sql_query(db_path, sql):
    """Validate and run one read-only query; always returns a JSON string
    (errors included, so the model can correct itself)."""
    normalized = (sql or "").strip().rstrip(";")
    if ";" in normalized:
        return json.dumps({"error": "Only a single statement is allowed."})
    first_word = normalized.split(None, 1)[0].lower() if normalized else ""
    if first_word not in ("select", "with"):
        return json.dumps({"error": "Only SELECT queries are allowed."})
    lowered = f" {normalized.lower()} "
    if any(f" {kw} " in lowered for kw in _SQL_FORBIDDEN_KEYWORDS):
        return json.dumps({"error": "Only read-only SELECT queries are allowed."})

    try:
        # mode=ro is enforced by SQLite itself, independent of the string
        # checks above.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(normalized)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS)
        finally:
            conn.close()
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})

    return json.dumps({
        "columns": cols,
        "rows": [list(r) for r in rows],
        "truncated": len(rows) == MAX_ROWS,
    })


# --- MCP stdio server -------------------------------------------------------

_TOOL = {
    "name": "run_sql",
    "description": RUN_SQL_DESCRIPTION,
    "inputSchema": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "A single SELECT (or WITH ... SELECT) statement against usage_events.",
            },
        },
        "required": ["sql"],
    },
}


def _handle(db_path, msg):
    """Response dict for a request, or None for notifications."""
    method, req_id = msg.get("method"), msg.get("id")
    if req_id is None:
        return None  # notification (e.g. notifications/initialized)

    if method == "initialize":
        params = msg.get("params") or {}
        result = {
            "protocolVersion": params.get("protocolVersion", "2025-06-18"),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "claude-usage", "version": "1"},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [_TOOL]}
    elif method == "tools/call":
        params = msg.get("params") or {}
        if params.get("name") != "run_sql":
            return {"jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32602, "message": f"Unknown tool: {params.get('name')}"}}
        text = run_sql_query(db_path, (params.get("arguments") or {}).get("sql", ""))
        result = {"content": [{"type": "text", "text": text}],
                  "isError": '"error"' in text[:10]}
    else:
        return {"jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def serve(db_path):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        response = _handle(db_path, msg) if isinstance(msg, dict) else None
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: ask_mcp_server.py <path to usage.db>")
    serve(sys.argv[1])
