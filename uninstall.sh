#!/bin/bash
# Removes the launchd job, the `claude-usage` command, and stops the
# dashboard. Pass --purge to also delete the program and usage data.
set -uo pipefail

INSTALL_DIR="${CLAUDE_USAGE_INSTALL_DIR:-$HOME/.local/share/claude-usage}"
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
# Match install.sh / ingest.py data-dir resolution.
LEGACY_DIR="$HOME/.claude-usage"
if [ -n "${CLAUDE_USAGE_DATA_DIR:-}" ]; then
    DATA_DIR="${CLAUDE_USAGE_DATA_DIR/#\~/$HOME}"
elif [ -f "$LEGACY_DIR/usage.db" ] && [ ! -f "$HOME/.local/state/claude-usage/usage.db" ]; then
    DATA_DIR="$LEGACY_DIR"
else
    DATA_DIR="$HOME/.local/state/claude-usage"
fi

PLIST_LABEL="com.robindebled.claude-usage-ingest"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
BIN_LINK="$HOME/.local/bin/claude-usage"
PURGE=0

for arg in "$@"; do
    [ "$arg" = "--purge" ] && PURGE=1
done

echo "==> Unloading launchd job: ${PLIST_LABEL}"
launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" >/dev/null 2>&1 || true
rm -f "$PLIST_PATH"

echo "==> Removing 'claude-usage' command"
rm -f "$BIN_LINK"

echo "==> Stopping the dashboard server, if running"
DASH_PID="$(pgrep -f "${INSTALL_DIR}/venv/bin/streamlit" || true)"
if [ -n "$DASH_PID" ]; then
    kill "$DASH_PID"
    echo "    stopped (pid ${DASH_PID})"
else
    echo "    not running"
fi

echo
echo "Note: cleanupPeriodDays in ~/.claude/settings.json was raised during"
echo "install and is left as-is (lowering it would delete your Claude Code"
echo "transcript history on next launch). Edit it yourself if you want it back."

DB_SIZE="$(du -h "${DATA_DIR}/usage.db" 2>/dev/null | cut -f1)"

if [ "$PURGE" -eq 1 ]; then
    echo
    echo "--purge: this will permanently delete"
    echo "    the installed program at ${INSTALL_DIR} (code + venv), and"
    echo "    your data at ${DATA_DIR}, including usage.db (${DB_SIZE:-unknown size})."
    read -r -p "Type 'yes' to confirm: " CONFIRM
    if [ "$CONFIRM" = "yes" ]; then
        rm -rf "$INSTALL_DIR"
        echo "==> Deleted ${INSTALL_DIR}"
        rm -rf "$DATA_DIR"
        echo "==> Deleted ${DATA_DIR}"
    else
        echo "==> Skipped deletion of program and data"
    fi
else
    echo
    echo "Program kept:  ${INSTALL_DIR} (code + venv)"
    echo "Data kept:     ${DATA_DIR} (usage.db: ${DB_SIZE:-not found})"
    echo "Re-run with --purge to delete both."
fi
