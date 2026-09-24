#!/bin/bash
# Installs the program, schedules ingest via launchd, and adds the
# `claude-usage` command. See README.md for details.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${CLAUDE_USAGE_INSTALL_DIR:-$HOME/.local/share/claude-usage}"
INSTALL_DIR="${INSTALL_DIR/#\~/$HOME}"
# Must match ingest.py / dashboard.py's resolver.
LEGACY_DIR="$HOME/.claude-usage"
if [ -n "${CLAUDE_USAGE_DATA_DIR:-}" ]; then
    DATA_DIR="${CLAUDE_USAGE_DATA_DIR/#\~/$HOME}"
elif [ -f "$LEGACY_DIR/usage.db" ]; then
    DATA_DIR="$LEGACY_DIR"
else
    DATA_DIR="$HOME/.local/state/claude-usage"
fi

PLIST_LABEL="com.robindebled.claude-usage-ingest"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
BIN_DIR="$HOME/.local/bin"
BIN_LINK="${BIN_DIR}/claude-usage"
SETTINGS_PATH="$HOME/.claude/settings.json"
SUGGESTED_RETENTION_DAYS=365
# How often ingest runs. Ingest is incremental (per-file byte offsets, cached
# PR lookups), so frequent runs are cheap. launchd polls every
# INGEST_POLL_SECONDS and ingest.py itself decides whether it's due, which lets
# a manual run (e.g. the dashboard's "ingest now") reset the countdown.
# Changing either and running `claude-usage --update` rewrites and reloads the
# plist on existing installs.
INGEST_INTERVAL_MINUTES=30
INGEST_POLL_SECONDS=60
DEFAULT_RETENTION_DAYS=30

echo "==> Installing Claude usage dashboard"
echo "    source:  ${SRC_DIR}"
echo "    program: ${INSTALL_DIR}   (code + venv)"
echo "    data:    ${DATA_DIR}   (usage.db + logs, preserved across updates)"
echo
echo "    This sets up: a scheduled job that ingests your Claude Code session"
echo "    logs into a local SQLite database, a 'claude-usage' command to open"
echo "    the dashboard, and (optionally) a longer transcript retention window."
echo

# --- 1. Copy the program into the stable install dir ----------------------
# Explicit manifest (never the venv/DB/logs) so stale files from an older
# version can be pruned below.
MANIFEST=(
    dashboard.py
    ingest.py
    run_dashboard.sh
    uninstall.sh
    requirements.txt
    bin/claude-usage
    .streamlit/config.toml
)

echo "==> Syncing program files into ${INSTALL_DIR}"
for rel in "${MANIFEST[@]}"; do
    mkdir -p "${INSTALL_DIR}/$(dirname "$rel")"
    cp "${SRC_DIR}/${rel}" "${INSTALL_DIR}/${rel}"
done

# Prune files from a previous install no longer in the manifest. Only
# touches recorded paths -- venv/.source/usage.db are never at risk.
PREV_MANIFEST="${INSTALL_DIR}/.manifest"
if [ -f "${PREV_MANIFEST}" ]; then
    while IFS= read -r rel; do
        [ -z "$rel" ] && continue
        still_shipped=0
        for keep in "${MANIFEST[@]}"; do
            [ "$keep" = "$rel" ] && { still_shipped=1; break; }
        done
        if [ "$still_shipped" -eq 0 ] && [ -e "${INSTALL_DIR}/${rel}" ]; then
            echo "    pruning stale file from a previous version: ${rel}"
            rm -f "${INSTALL_DIR}/${rel}"
        fi
    done < "${PREV_MANIFEST}"
fi
printf '%s\n' "${MANIFEST[@]}" > "${PREV_MANIFEST}"

chmod +x "${INSTALL_DIR}/run_dashboard.sh" "${INSTALL_DIR}/bin/claude-usage" \
         "${INSTALL_DIR}/uninstall.sh"
# Recorded so 'claude-usage --update' can pull new code from the same checkout.
printf '%s\n' "${SRC_DIR}" > "${INSTALL_DIR}/.source"

# Stop any running dashboard so a reopen picks up the freshly-copied code.
if pgrep -f "${INSTALL_DIR}/venv/bin/streamlit run dashboard.py" >/dev/null 2>&1; then
    echo "    stopping the running dashboard so it reloads the new code..."
    pkill -f "${INSTALL_DIR}/venv/bin/streamlit run dashboard.py" 2>/dev/null || true
fi

# --- 2. Data dir + one-time migration of an existing DB -------------------
mkdir -p "${DATA_DIR}"
# Migrate an old ~/.claude-usage/usage.db onto the new XDG data dir, if any.
if [ "${DATA_DIR}" != "${LEGACY_DIR}" ] && [ -f "${LEGACY_DIR}/usage.db" ] \
   && [ ! -f "${DATA_DIR}/usage.db" ]; then
    echo "==> Migrating existing usage.db from ${LEGACY_DIR} to ${DATA_DIR}"
    cp "${LEGACY_DIR}/usage.db" "${DATA_DIR}/usage.db"
    echo "    (original left in place at ${LEGACY_DIR}/usage.db as a backup)"
fi

# --- 3. Python venv for the dashboard (ingest.py is stdlib-only) ----------
if [ ! -x "${INSTALL_DIR}/venv/bin/pip" ]; then
    echo "==> Creating a venv for the dashboard's dependencies."
    echo "    The ingest job itself only needs the stdlib, so it runs on the system python3."
    python3 -m venv "${INSTALL_DIR}/venv"
    "${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade pip
fi
echo "==> Syncing dashboard dependencies from requirements.txt"
"${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade -r "${INSTALL_DIR}/requirements.txt"

# --- 4. Offer to raise Claude Code's transcript retention -----------------
echo
echo "==> Transcript retention (cleanupPeriodDays)"
echo "    Claude Code deletes session transcripts under ~/.claude/projects/ after"
echo "    cleanupPeriodDays (${DEFAULT_RETENTION_DAYS} days if unset). The ingest job reads those files every"
echo "    ${INGEST_INTERVAL_MINUTES} minutes, so day-to-day usage is unaffected either way - but anything pruned before"
echo "    ingest ever sees it is lost from your history for good (e.g. after the Mac"
echo "    was asleep/off across a boundary, or before you first installed this tool)."
echo "    Longer retention means more raw transcripts kept on disk in the meantime."

CURRENT_RETENTION="$(python3 -c "
import json, os
p = '${SETTINGS_PATH}'
try:
    v = json.load(open(p)).get('cleanupPeriodDays')
except (FileNotFoundError, json.JSONDecodeError):
    v = None
print(v if v is not None else '')
" 2>/dev/null || true)"
DISPLAY_CURRENT="${CURRENT_RETENTION:-${DEFAULT_RETENTION_DAYS} (default, unset)}"
echo "    Current setting: ${DISPLAY_CURRENT}"

TARGET_RETENTION_DAYS="$SUGGESTED_RETENTION_DAYS"
if [ -t 0 ]; then
    read -r -p "    Raise retention to ${SUGGESTED_RETENTION_DAYS} days? [Y/n] " ANSWER
else
    ANSWER=""
    echo "    (non-interactive shell, defaulting to yes)"
fi
case "$ANSWER" in
    [nN]*)
        TARGET_RETENTION_DAYS="$DEFAULT_RETENTION_DAYS"
        echo "    Keeping retention at ${DEFAULT_RETENTION_DAYS} days (or whatever it's currently set to)."
        ;;
esac

python3 - "$SETTINGS_PATH" "$TARGET_RETENTION_DAYS" <<'PYEOF'
import json
import os
import sys

settings_path, target = sys.argv[1], int(sys.argv[2])

if os.path.exists(settings_path):
    with open(settings_path) as f:
        raw = f.read()
    try:
        settings = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(f"    settings.json does not parse ({e}); leaving retention untouched.")
        sys.exit(0)
else:
    settings = {}

current = settings.get("cleanupPeriodDays")
# Never lower it -- a lower value would delete transcripts sooner than
# whatever is already in place.
new_value = max(current or 0, target)

if current == new_value:
    print(f"    cleanupPeriodDays already {current}, leaving as-is")
    sys.exit(0)

settings["cleanupPeriodDays"] = new_value
tmp_path = settings_path + ".tmp"
os.makedirs(os.path.dirname(settings_path), exist_ok=True)
with open(tmp_path, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")
os.replace(tmp_path, settings_path)
print(f"    cleanupPeriodDays: {current!r} -> {new_value}")
PYEOF

# --- 5. Schedule the ingest job via launchd -------------------------------
echo
echo "==> Scheduling the ingest job with launchd (label: ${PLIST_LABEL})."
echo "    Runs ${INSTALL_DIR}/ingest.py every ${INGEST_INTERVAL_MINUTES} minutes (and now, if due) to keep usage.db current."
mkdir -p "$HOME/Library/LaunchAgents"
# launchd starts jobs with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin) that
# omits Homebrew, so `gh` (used for PR resolution) wouldn't be found. Build a
# PATH that includes wherever `gh` actually lives, plus the usual locations.
GH_PATH="$(command -v gh || true)"
LAUNCHD_PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
if [ -n "$GH_PATH" ]; then
    LAUNCHD_PATH="$(dirname "$GH_PATH"):${LAUNCHD_PATH}"
fi

cat > "$PLIST_PATH" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>${INSTALL_DIR}/ingest.py</string>
        <string>--every-minutes</string>
        <string>${INGEST_INTERVAL_MINUTES}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>CLAUDE_USAGE_DATA_DIR</key>
        <string>${DATA_DIR}</string>
        <key>PATH</key>
        <string>${LAUNCHD_PATH}</string>
    </dict>
    <key>StartInterval</key>
    <integer>${INGEST_POLL_SECONDS}</integer>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${DATA_DIR}/launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>${DATA_DIR}/launchd.err.log</string>
</dict>
</plist>
PLISTEOF

launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
echo "    installed and loaded."

# --- 6. `claude-usage` command to open the dashboard ----------------------
echo
echo "==> Installing the 'claude-usage' command at ${BIN_LINK}."
echo "    It starts the dashboard server (if not already running) and opens your"
echo "    browser to it - the server itself is NOT kept running permanently in the"
echo "    background; it only starts when you ask for it."
mkdir -p "$BIN_DIR"
ln -sf "${INSTALL_DIR}/bin/claude-usage" "$BIN_LINK"

echo
case ":${PATH}:" in
    *":${BIN_DIR}:"*)
        echo "==> Done. Run 'claude-usage' to open the dashboard."
        ;;
    *)
        echo "==> Almost done. '${BIN_DIR}' is not on your PATH yet, so the"
        echo "    'claude-usage' command won't be found until you add it."
        echo
        echo "    Add this line to your shell profile (~/.zshrc for zsh, the"
        echo "    macOS default; ~/.bashrc for bash), then restart your terminal:"
        echo
        echo "        export PATH=\"${BIN_DIR}:\$PATH\""
        echo
        echo "    Or, if you'd rather not touch PATH, add an alias instead:"
        echo
        echo "        alias claude-usage=\"${BIN_LINK}\""
        echo
        echo "    Either way, you can run it right now with the full path:"
        echo "        ${BIN_LINK}"
        ;;
esac
