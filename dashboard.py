import datetime as dt
import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.parse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from anthropic import Anthropic, beta_tool

from ask_mcp_server import USAGE_EVENTS_SCHEMA, run_sql_query

def _data_dir():
    """Resolve the usage.db location. Mirrors ingest.py's resolver."""
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


DB_PATH = os.path.join(_data_dir(), "usage.db")
LOCK_PATH = os.path.join(_data_dir(), "ingest.lock")
PROGRESS_PATH = os.path.join(_data_dir(), "ingest.progress")
SCHEDULE_PATH = os.path.join(_data_dir(), "ingest.schedule.json")
INGEST_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest.py")

_LIGHT = dict(
    surface="#ffffff",
    page="#ffffff",
    ink_primary="#08051f",
    ink_secondary="#5b5470",
    muted="#8b85a8",
    gridline="#e6e3f7",
    border="rgba(11,6,32,0.10)",
    categorical=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
)

# UI accent (buttons, focus rings, hero gradient) -- separate from the
# colorblind-validated categorical palette used for chart series.
ACCENT = "#6f5cff"
ACCENT_DEEP = "#4736c9"
ACCENT_INK = "#160a70"
ACCENT_MAGENTA = "#d14fff"
LAVENDER_PALE = "#f3f2ff"
LAVENDER_MIST = "#f5f3ff"
LAVENDER_TINT = "#d6d4ff"
ACCENT_GRADIENT = f"linear-gradient(90deg, {ACCENT}, {ACCENT_DEEP}, {ACCENT_INK})"

# Delta-chip colors: WCAG-legible foreground for text/border, plus a
# separate full-saturation variant used only for the soft outer glow.
SYNTH_PINK = "#e6009c"
SYNTH_PINK_GLOW = "#ff2ec4"
SYNTH_CYAN = "#0089b3"
SYNTH_CYAN_GLOW = "#00e5ff"
SYNTH_YELLOW = "#a67c00"
SYNTH_YELLOW_GLOW = "#f9f871"
_DARK = dict(
    surface="#1a1a19",
    page="#0d0d0d",
    ink_primary="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    gridline="#2c2c2a",
    border="rgba(255,255,255,0.10)",
    categorical=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
)
# Sequential ramp for magnitude encodings (heatmap shading); never used as
# a categorical/identity color.
SEQUENTIAL_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

st.set_page_config(
    page_title="Claude Code Usage",
    layout="wide",
    menu_items={
        "Get Help": "https://github.com/rdebled-pearl/claude-usage",
        "Report a bug": "https://github.com/rdebled-pearl/claude-usage/issues",
        "About": (
            "**Claude Code usage dashboard** \u2014 a local, single-user tool "
            "for token/cost analytics over your Claude Code sessions.\n\n"
            "[Source on GitHub](https://github.com/rdebled-pearl/claude-usage)"
        ),
    },
)


def resolve_theme_type():
    """Always "light" -- st.context.theme reflects a client-cached choice
    (localStorage) that a server-side reset can't clear, so detection was
    dropped in favor of forcing light unconditionally."""
    return "light"


def apply_theme():
    """Set the module-level color tokens every chart/CSS helper reads."""
    global BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, CATEGORICAL
    global INK_PRIMARY, INK_SECONDARY, MUTED, GRIDLINE, SURFACE, PAGE, BORDER
    global PLOTLY_LAYOUT

    tokens = _DARK if resolve_theme_type() == "dark" else _LIGHT
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = tokens["categorical"]
    CATEGORICAL = tokens["categorical"]
    INK_PRIMARY = tokens["ink_primary"]
    INK_SECONDARY = tokens["ink_secondary"]
    MUTED = tokens["muted"]
    GRIDLINE = tokens["gridline"]
    SURFACE = tokens["surface"]
    PAGE = tokens["page"]
    BORDER = tokens["border"]

    PLOTLY_LAYOUT = dict(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(family="'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_PRIMARY, size=13),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=INK_PRIMARY, size=12),
        ),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRIDLINE, font=dict(color=INK_PRIMARY, size=12)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return tokens


apply_theme()


def style_axes(fig, hovermode="closest", spikes=True):
    """`hovermode="x"` for multi-series charts sharing an x-axis (stacked
    area, multi-line), so the tooltip lists every series, not just the
    nearest painted pixel. `spikes=False` for grid/cell forms (heatmap)."""
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor=GRIDLINE, ticks="",
        tickfont=dict(color=INK_PRIMARY, size=12),
        title_font=dict(color=INK_PRIMARY, size=12),
        showspikes=spikes, spikemode="across", spikesnap="cursor",
        spikedash="solid", spikethickness=1, spikecolor=INK_SECONDARY,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRIDLINE, zeroline=False, ticks="",
        tickfont=dict(color=INK_PRIMARY, size=12),
        title_font=dict(color=INK_PRIMARY, size=12),
        showspikes=spikes, spikemode="across", spikesnap="cursor",
        spikedash="solid", spikethickness=1, spikecolor=INK_SECONDARY,
    )
    fig.update_layout(hovermode=hovermode, **PLOTLY_LAYOUT)
    return fig


@st.cache_data(ttl=60)
def load_data(db_mtime):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT date, timestamp, session_id, model, repo, branch, pr_number, pr_url,
               is_subagent, agent_id, agent_type, agent_description,
               input_tokens, output_tokens, cache_read_input_tokens,
               cache_creation_5m_tokens, cache_creation_1h_tokens, thinking_tokens,
               input_cost, output_cost, cache_write_cost, cache_read_cost, total_cost
        FROM usage_events
        """,
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["total_cost"] = df["total_cost"].fillna(0.0)
    df["is_subagent"] = df["is_subagent"].fillna(0).astype(bool)

    # timestamp is stored as raw UTC; convert to local tz so hour-of-day
    # reflects wall-clock time, not UTC.
    local_tz = dt.datetime.now().astimezone().tzinfo
    ts_local = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601").dt.tz_convert(local_tz)
    df["ts_local"] = ts_local
    df["hour"] = ts_local.dt.hour
    df["dow"] = ts_local.dt.day_name()
    df["cache_write_tokens"] = df["cache_creation_5m_tokens"] + df["cache_creation_1h_tokens"]
    return df


def get_db_mtime():
    try:
        return os.path.getmtime(DB_PATH)
    except OSError:
        return 0


def fold_top_n(df, col, n=None, top_values=None, other_label="Other"):
    """Collapse every value outside the top N into `other_label`. Pass
    `top_values` computed from the full unfiltered dataset to keep
    membership (and thus color assignment) stable across filter changes."""
    if top_values is None:
        top_values = df.groupby(col)["total_cost"].sum().sort_values(ascending=False).head(n).index
    out = df.copy()
    out[col] = out[col].where(out[col].isin(top_values), other_label)
    return out


def build_model_color_map(df_all, n=3, other_label="Other"):
    """One color per model, assigned from the full dataset's cost ranking so
    a filtered view never reassigns colors."""
    ranked = df_all.groupby("model")["total_cost"].sum().sort_values(ascending=False)
    top_models = list(ranked.head(n).index)
    color_map = {model: CATEGORICAL[i] for i, model in enumerate(top_models)}
    color_map[other_label] = MUTED
    return top_models, color_map


def emphasis_colors(n, highlight="max"):
    """One bar in full accent, the rest in a quiet tint. `highlight="max"`
    marks the last bar (largest, since horizontal bars sort ascending)."""
    if n <= 0:
        return []
    idx = (n - 1) if highlight == "max" else highlight
    return [ACCENT if i == idx else LAVENDER_TINT for i in range(n)]


def inject_css():
    """Force-overrides page chrome too, not just widgets -- Streamlit can
    persist a client-cached "dark" choice that a server-side default can't
    clear."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

        /* Plex Sans everywhere; Plex Mono is reserved for dollar/token figures. */
        [data-testid="stApp"], [data-testid="stApp"] * {{
            font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
        }}
        /* The universal font rule above also caught Streamlit's icon glyphs,
           which render via a ligature font (Material Symbols) -- exempt them. */
        [data-testid*="Icon"] {{
            font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', sans-serif !important;
        }}

        [data-testid="stApp"], [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], [data-testid="stMain"], body {{
            background-color: {PAGE} !important;
            color: {INK_PRIMARY} !important;
        }}
        [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span,
        h1, h2, h3, h4, label {{ color: {INK_PRIMARY} !important; }}

        /* Streamlit's header toolbar overlays content rather than pushing it
           down; clear it and leave room for the eyebrow line above the title. */
        .block-container {{ padding-top: 4.5rem; }}

        /* Stat tiles: no card chrome. Governs every st.metric() in the app. */
        div[data-testid="stMetric"] {{
            background: transparent;
            border: none;
            padding: 0;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', ui-monospace, monospace;
            font-size: 1.25rem;
            font-variant-numeric: tabular-nums;
            overflow: visible;
            white-space: normal;
            line-height: 1.3;
            color: {ACCENT} !important;
        }}
        /* Streamlit's inner value div carries its own nowrap/ellipsis that
           the outer rule above doesn't reach -- override it too. */
        div[data-testid="stMetricValue"] > div {{
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }}
        div[data-testid="stMetricLabel"] {{ font-size: 0.8rem; color: {INK_SECONDARY} !important; }}

        /* Hero band: the one bold reading on the page, filled with a
           gradient via background-clip: text. */
        .cost-hero {{
            padding: 1.25rem 1.5rem 1.4rem;
            margin-bottom: 1.5rem;
            border-radius: 20px;
            background:
                radial-gradient(circle at 12% 15%, rgba(111,92,255,0.14), transparent 60%),
                linear-gradient(135deg, {LAVENDER_MIST} 0%, #ffffff 65%);
        }}
        .cost-hero__span {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.85rem;
            color: {MUTED};
            margin-bottom: 0.2rem;
        }}
        .cost-hero__value {{
            font-family: 'IBM Plex Mono', ui-monospace, monospace;
            font-weight: 600;
            font-size: 3.25rem;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
            background-image: {ACCENT_GRADIENT};
            background-clip: text;
            -webkit-background-clip: text;
            color: transparent;
            -webkit-text-fill-color: transparent;
        }}
        .cost-hero__label {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.9rem;
            color: {INK_SECONDARY};
            margin-bottom: 0.7rem;
        }}
        .ledger-line {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.9rem;
            color: {INK_SECONDARY};
            margin-top: 0.3rem;
        }}
        .ledger-line b {{
            font-family: 'IBM Plex Mono', ui-monospace, monospace;
            color: {ACCENT};
            font-weight: 600;
        }}

        /* Secondary KPI tier: bold stat tiles bridging the hero and the charts.
           A consistent card grammar (subtle border, matching radius, generous
           padding) so this row scans as one family and the hero stops being the
           page's only filled surface. */
        .kpi-band {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.9rem;
            margin-bottom: 1.6rem;
        }}
        .kpi {{
            background: {LAVENDER_PALE};
            border: 1px solid {LAVENDER_TINT};
            border-radius: 16px;
            padding: 1rem 1.15rem 0.9rem;
        }}
        .kpi__label {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {MUTED};
            margin-bottom: 0.35rem;
        }}
        .kpi__value {{
            font-family: 'IBM Plex Mono', ui-monospace, monospace;
            font-weight: 600;
            font-size: 1.9rem;
            line-height: 1.1;
            font-variant-numeric: tabular-nums;
            color: {ACCENT_INK};
        }}
        .kpi__delta {{ margin-top: 0.4rem; min-height: 1.1rem; }}

        /* Period-over-period change chips: direction only, not good/bad --
           more usage isn't inherently bad. */
        .delta {{
            display: inline-flex;
            align-items: center;
            gap: 0.2rem;
            font-family: 'IBM Plex Mono', ui-monospace, monospace;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            white-space: nowrap;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: #ffffff;
            vertical-align: middle;
        }}
        /* Two classes (not one) needed to outrank the global !important rule
           pinning every stMarkdownContainer span color to INK_PRIMARY. */
        .delta.delta--up {{
            color: {SYNTH_PINK} !important;
            border: 1.5px solid {SYNTH_PINK_GLOW};
            box-shadow: 0 0 8px rgba(255,46,196,0.45);
        }}
        .delta.delta--down {{
            color: {SYNTH_CYAN} !important;
            border: 1.5px solid {SYNTH_CYAN_GLOW};
            box-shadow: 0 0 8px rgba(0,229,255,0.45);
        }}
        .delta.delta--flat {{
            color: {SYNTH_YELLOW} !important;
            border: 1.5px solid {SYNTH_YELLOW_GLOW};
            box-shadow: 0 0 6px rgba(249,248,113,0.4);
        }}

        /* When a delta chip rides inside the hero's label line, give it a touch
           of separation from the "spend for this range" text. */
        .cost-hero__label .delta {{ margin-left: 0.5rem; }}

        @media (max-width: 900px) {{
            .kpi-band {{ grid-template-columns: repeat(2, 1fr); }}
        }}

        /* Override Streamlit's default theme red on the selected segmented-
           control pill, active tab label, and tab-selection underline. */
        button[data-variant="segmented_control"][aria-checked="true"] {{
            color: {ACCENT} !important;
            border-color: {ACCENT} !important;
            background-color: rgba(111,92,255,0.10) !important;
            outline-color: {ACCENT} !important;
        }}
        [data-testid="stTab"][aria-selected="true"] {{
            color: {ACCENT} !important;
        }}
        [data-testid="stTab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stTab"][aria-selected="true"] [data-testid="stMarkdownContainer"] span {{
            color: {ACCENT} !important;
        }}
        [data-testid="stTab"] .react-aria-SelectionIndicator {{
            background-color: {ACCENT} !important;
        }}

        .eyebrow {{
            font-family: 'IBM Plex Sans', sans-serif;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: {ACCENT};
            margin-bottom: 0.3rem;
        }}

        [data-testid="stCaptionContainer"], .stCaption {{ color: {INK_SECONDARY} !important; }}

        h3 {{ font-size: 1.1rem !important; margin-bottom: 0.4rem; }}

        /* Bottom-align only the filter bar row (mismatched widget label
           heights); every other st.columns() pair top-aligns by default. */
        div[data-testid="stHorizontalBlock"] {{ align-items: start; }}
        .st-key-filter-bar-row div[data-testid="stHorizontalBlock"] {{ align-items: end; }}
        /* Headless localStorage bridge; takes no space in the layout. */
        .st-key-filter-prefs-host {{ display: none; }}

        /* Multiselect/date-input fields render as React Aria components, not
           BaseWeb -- [data-baseweb=...] selectors don't match here. */
        [data-testid="stMultiSelect"] [role="group"],
        [data-testid="stDateInputField"] {{
            background-color: {SURFACE} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stMultiSelect"] [role="group"]:hover,
        [data-testid="stDateInputField"]:hover {{
            border-color: {ACCENT} !important;
        }}
        [data-testid="stMultiSelect"] input {{
            background-color: transparent !important;
            color: {INK_PRIMARY} !important;
        }}
        /* Selected-value chip: <span data-tag> with a nested label span
           and a currentColor remove-icon button. */
        [data-tag] {{
            background-color: {LAVENDER_TINT} !important;
            color: {ACCENT_INK} !important;
            border-radius: 6px !important;
        }}
        [data-tag] span {{ color: {ACCENT_INK} !important; }}
        [data-tag] button {{ color: {ACCENT_INK} !important; }}
        [data-testid="stMultiSelectDropdown"] {{
            border: 1px solid {BORDER} !important;
        }}
        [data-testid="stMultiSelectDropdown"] [role="option"]:hover,
        [data-testid="stMultiSelectDropdown"] [role="option"][aria-selected="true"] {{
            background-color: {LAVENDER_PALE} !important;
        }}

        /* Scoped to stButton/stFormSubmitButton specifically -- excludes the
           header Deploy button and dataframe icon buttons. */
        div[data-testid="stButton"] button,
        div[data-testid="stFormSubmitButton"] button {{
            background-color: {ACCENT} !important;
            background-image: none !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.5rem 1.25rem !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 2px rgba(22,10,112,0.20) !important;
            transition: background-image 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease !important;
        }}
        div[data-testid="stButton"] button p,
        div[data-testid="stFormSubmitButton"] button p {{
            color: #ffffff !important;
        }}
        div[data-testid="stButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {{
            background-image: linear-gradient(90deg, {ACCENT}, {ACCENT_MAGENTA}) !important;
            box-shadow: 0 6px 16px rgba(111,92,255,0.35) !important;
            transform: translateY(-1px) !important;
        }}
        div[data-testid="stButton"] button:active,
        div[data-testid="stFormSubmitButton"] button:active {{
            background-image: none !important;
            background-color: {ACCENT_DEEP} !important;
            transform: translateY(0) !important;
            box-shadow: 0 1px 2px rgba(22,10,112,0.20) !important;
        }}
        div[data-testid="stButton"] button:focus-visible,
        div[data-testid="stFormSubmitButton"] button:focus-visible {{
            outline: 2px solid {ACCENT} !important;
            outline-offset: 2px !important;
        }}
        div[data-testid="stButton"] button:disabled,
        div[data-testid="stFormSubmitButton"] button:disabled {{
            background-color: {MUTED} !important;
            background-image: none !important;
            color: #ffffff !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


RANGE_OPTIONS = ["Today", "7d", "30d", "90d", "All", "Custom"]
FILTER_PARAMS = ("range", "repo", "model", "from", "to")
FILTER_WIDGET_KEYS = ("f_range", "f_repos", "f_models", "f_custom")
FILTER_PREFS_KEY = "filter-prefs"

# Headless bridge between the URL-held filter state and localStorage, so a
# bare URL (e.g. from `claude-usage`) resumes the last selection. On the first
# render of a page load with no filters in the URL, it hands the saved query
# string back to Python; otherwise it saves the current one.
_filter_prefs_component = st.components.v2.component(
    "filter_prefs",
    js="""
    export default function({ data, setTriggerValue }) {
        const KEY = "claude-usage:filters";
        let saved = null;
        try { saved = localStorage.getItem(KEY); } catch (e) {}
        if (!window.__claudeUsageFiltersChecked) {
            window.__claudeUsageFiltersChecked = true;
            if (!data.url_had_filters && saved && saved !== data.qs) {
                setTriggerValue("restore", saved);
                return;
            }
        }
        try { localStorage.setItem(KEY, data.qs); } catch (e) {}
    }
    """,
)


def _restore_filters():
    """Apply the localStorage-saved query string, then drop the widget state
    so the next run re-seeds (and re-validates) every filter from the URL."""
    saved = (st.session_state.get(FILTER_PREFS_KEY) or {}).get("restore")
    if not saved:
        return
    parsed = urllib.parse.parse_qs(saved)
    st.query_params.from_dict(
        {k: v if k in ("repo", "model") else v[0] for k, v in parsed.items() if k in FILTER_PARAMS}
    )
    for key in FILTER_WIDGET_KEYS:
        st.session_state.pop(key, None)


def _parse_iso_date(value):
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _seed_filter_state(repo_options, model_options, min_date, max_date):
    """Initialise filter widget state from the URL, once per session (or after
    a restore). Values are validated so a stale/hand-edited URL can't feed a
    widget an option it doesn't have."""
    qp = st.query_params
    if "_url_had_filters" not in st.session_state:
        st.session_state["_url_had_filters"] = any(k in qp for k in FILTER_PARAMS)

    if "f_range" not in st.session_state:
        preset = qp.get("range")
        st.session_state.f_range = preset if preset in RANGE_OPTIONS else "All"
    if "f_repos" not in st.session_state:
        st.session_state.f_repos = [r for r in qp.get_all("repo") if r in repo_options]
    if "f_models" not in st.session_state:
        st.session_state.f_models = [m for m in qp.get_all("model") if m in model_options]
    if "f_custom" not in st.session_state:
        start = _parse_iso_date(qp.get("from")) or min_date
        end = _parse_iso_date(qp.get("to")) or max_date
        start = min(max(start, min_date), max_date)
        end = min(max(end, start), max_date)
        st.session_state.f_custom = (start, end)


def _persist_filters(preset, repos, models, custom_range):
    """Mirror the current selection into the URL and localStorage."""
    params = {}
    if preset:
        params["range"] = preset
    if repos:
        params["repo"] = repos
    if models:
        params["model"] = models
    if preset == "Custom" and isinstance(custom_range, tuple) and len(custom_range) == 2:
        params["from"], params["to"] = (d.isoformat() for d in custom_range)

    # Only touch our own keys so unrelated params survive.
    for key in FILTER_PARAMS:
        if key not in params and key in st.query_params:
            del st.query_params[key]
    for key, value in params.items():
        if (st.query_params.get_all(key) if isinstance(value, list) else st.query_params.get(key)) != value:
            st.query_params[key] = value

    with st.container(key="filter-prefs-host"):
        _filter_prefs_component(
            key=FILTER_PREFS_KEY,
            data={
                "qs": urllib.parse.urlencode(params, doseq=True),
                "url_had_filters": st.session_state["_url_had_filters"],
            },
            on_restore_change=_restore_filters,
        )


def filter_bar(df):
    """One always-visible row above every chart; every tab reads its output."""
    min_date, max_date = df["date"].min().date(), df["date"].max().date()
    repo_options = sorted(df["repo"].dropna().unique())
    model_options = sorted(df["model"].dropna().unique())
    _seed_filter_state(repo_options, model_options, min_date, max_date)

    # Two stacked rows: the custom-range picker only appears for "Custom",
    # directly under the preset control rather than off to the side.
    filter_row = st.container(key="filter-bar-row")
    c_preset, c_repo, c_model = filter_row.columns([2, 1.5, 1.5])
    with c_preset:
        preset = st.segmented_control(
            "Date range", RANGE_OPTIONS,
            key="f_range", label_visibility="collapsed",
        )

    with c_repo:
        repos = st.multiselect(
            "Repo", repo_options, key="f_repos",
            placeholder="All repos", label_visibility="collapsed",
        )
    with c_model:
        models = st.multiselect(
            "Model", model_options, key="f_models",
            placeholder="All models", label_visibility="collapsed",
        )

    custom_range = None
    if preset == "Custom":
        c_custom, _, _ = filter_row.columns([2, 1.5, 1.5])
        with c_custom:
            # Fixed width instead of the column-filling default, sized to
            # hold just the date-range content.
            custom_range = st.date_input(
                "Custom range", key="f_custom",
                min_value=min_date, max_value=max_date,
                label_visibility="collapsed", width=300,
            )

    _persist_filters(preset, repos, models, custom_range)

    preset_days = {"7d": 7, "30d": 30, "90d": 90}
    if preset == "Today":
        # Actual calendar day (dates are stored in local time), not the
        # latest day with data, so an idle day shows as empty.
        start = end = dt.date.today()
    elif preset in preset_days:
        start = max_date - pd.Timedelta(days=preset_days[preset])
        end = max_date
    elif preset == "Custom" and isinstance(custom_range, tuple) and len(custom_range) == 2:
        start, end = custom_range
    else:
        start, end = min_date, max_date

    out = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    if repos:
        out = out[out["repo"].isin(repos)]
    if models:
        out = out[out["model"].isin(models)]

    st.caption(
        f"Showing **{start} to {end}** · {len(out):,} events · "
        f"{out['session_id'].nunique():,} sessions"
        + (f" · repo: {', '.join(repos)}" if repos else "")
        + (f" · model: {', '.join(models)}" if models else "")
    )
    meta = {"start": start, "end": end, "repos": repos, "models": models}
    return out, meta


def _summary_metrics(df):
    """The handful of headline numbers the hero and KPI band both read, so the
    two never drift out of agreement."""
    total_cost = df["total_cost"].sum()
    tokens = (
        df["input_tokens"].sum()
        + df["output_tokens"].sum()
        + df["cache_read_input_tokens"].sum()
        + df["cache_creation_5m_tokens"].sum()
        + df["cache_creation_1h_tokens"].sum()
    )
    sessions = df["session_id"].nunique()
    subagent_cost = df.loc[df["is_subagent"], "total_cost"].sum()
    return dict(
        cost=total_cost,
        tokens=tokens,
        sessions=sessions,
        subagent_cost=subagent_cost,
        avg_cost=(total_cost / sessions) if sessions else 0.0,
        subagent_share=(subagent_cost / total_cost * 100) if total_cost else 0.0,
    )


def previous_period(df_all, meta):
    """Equal-length window immediately preceding the selected range, same
    filters applied. Empty frame means no prior data / no delta."""
    start, end = meta["start"], meta["end"]
    length = end - start
    prior_end = start - pd.Timedelta(days=1)
    prior_start = prior_end - length
    out = df_all[
        (df_all["date"].dt.date >= prior_start) & (df_all["date"].dt.date <= prior_end)
    ]
    if meta.get("repos"):
        out = out[out["repo"].isin(meta["repos"])]
    if meta.get("models"):
        out = out[out["model"].isin(meta["models"])]
    return out


def _delta_html(cur, prev):
    """Period-over-period change chip. Color encodes direction, not good/bad.
    Returns "" when there's no comparable prior value."""
    if prev is None or prev == 0:
        return ""
    pct = (cur - prev) / prev * 100
    if abs(pct) < 0.5:
        return f'<span class="delta delta--flat">→ no change vs prev</span>'
    if pct > 0:
        return f'<span class="delta delta--up">▲ {pct:,.0f}% vs prev</span>'
    return f'<span class="delta delta--down">▼ {abs(pct):,.0f}% vs prev</span>'


def render_cost_hero(df, prior_df=None):
    """Hero reading below the filter bar, scoped to the current filters."""
    m = _summary_metrics(df)
    prev = _summary_metrics(prior_df) if prior_df is not None and not prior_df.empty else None
    span = f"{df['date'].min().date()} – {df['date'].max().date()}"
    delta = _delta_html(m["cost"], prev["cost"] if prev else None)

    # Single-line HTML: indented/blank-line-separated HTML gets parsed as a
    # markdown code block instead of rendering.
    st.markdown(
        f'<div class="cost-hero">'
        f'<div class="cost-hero__span">{span}</div>'
        f'<div class="cost-hero__value">${m["cost"]:,.2f}</div>'
        f'<div class="cost-hero__label">spend for this range {delta}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_kpi_band(df, prior_df=None):
    """Secondary stat tier between the hero and the charts, each tile with a
    period-over-period delta."""
    m = _summary_metrics(df)
    prev = _summary_metrics(prior_df) if prior_df is not None and not prior_df.empty else None

    def p(key):
        return prev[key] if prev else None

    tiles = [
        ("Tokens", f"{m['tokens']/1e6:,.1f}M", _delta_html(m["tokens"], p("tokens"))),
        ("Sessions", f"{m['sessions']:,}", _delta_html(m["sessions"], p("sessions"))),
        ("Avg $ / session", f"${m['avg_cost']:,.2f}", _delta_html(m["avg_cost"], p("avg_cost"))),
        ("Subagent share", f"{m['subagent_share']:,.0f}%", _delta_html(m["subagent_share"], p("subagent_share"))),
    ]
    # Single-line HTML -- see render_cost_hero.
    cells = "".join(
        f'<div class="kpi">'
        f'<div class="kpi__label">{label}</div>'
        f'<div class="kpi__value">{value}</div>'
        f'<div class="kpi__delta">{delta}</div>'
        f'</div>'
        for label, value, delta in tiles
    )
    st.markdown(f'<div class="kpi-band">{cells}</div>', unsafe_allow_html=True)


def tab_overview(df, model_color_map, top_models):
    st.subheader("Cost over time")
    daily = fold_top_n(df, "model", top_values=top_models)
    daily = daily.groupby(["date", "model"], as_index=False)["total_cost"].sum()
    fig = px.area(
        daily, x="date", y="total_cost", color="model",
        color_discrete_map=model_color_map,
        category_orders={"model": top_models + ["Other"]},
        labels={"total_cost": "Cost ($)", "date": "Date", "model": "Model"},
    )
    fig.update_traces(line=dict(width=2))
    style_axes(fig, hovermode="x")
    st.plotly_chart(fig, width='stretch')

    left, right = st.columns(2)

    with left:
        st.subheader("Cost by model")
        by_model = df.groupby("model", as_index=False)["total_cost"].sum().sort_values(
            "total_cost", ascending=True
        )
        fig = px.bar(
            by_model, x="total_cost", y="model", orientation="h",
            labels={"total_cost": "Cost ($)", "model": ""},
        )
        fig.update_traces(marker_color=emphasis_colors(len(by_model)))
        style_axes(fig)
        fig.update_layout(showlegend=False, height=max(220, 28 * len(by_model)))
        st.plotly_chart(fig, width='stretch')

    with right:
        st.subheader("Cost by repo")
        by_repo = df.groupby("repo", as_index=False)["total_cost"].sum().sort_values(
            "total_cost", ascending=True
        )
        fig = px.bar(
            by_repo, x="total_cost", y="repo", orientation="h",
            labels={"total_cost": "Cost ($)", "repo": ""},
        )
        fig.update_traces(marker_color=emphasis_colors(len(by_repo)))
        style_axes(fig)
        fig.update_layout(showlegend=False, height=max(220, 28 * len(by_repo)))
        st.plotly_chart(fig, width='stretch')


def tab_explore(df):
    st.subheader("Session leaderboard")
    subagent_by_session = (
        df[df["is_subagent"]].groupby("session_id", as_index=False)
        .agg(subagent_cost=("total_cost", "sum"), subagent_turns=("total_cost", "count"))
    )
    sessions = (
        df.groupby(["session_id", "repo"], as_index=False)
        .agg(
            turns=("model", "count"),
            cost=("total_cost", "sum"),
            cache_read_mtok=("cache_read_input_tokens", lambda s: s.sum() / 1e6),
            first_seen=("timestamp", "min"),
            pr_number=("pr_number", "first"),
        )
        .merge(subagent_by_session, on="session_id", how="left")
    )
    sessions[["subagent_cost", "subagent_turns"]] = (
        sessions[["subagent_cost", "subagent_turns"]].fillna(0)
    )
    sessions = sessions.sort_values("cost", ascending=False)
    st.dataframe(sessions, width='stretch', height=320)

    st.divider()
    st.subheader("Session detail")
    session_options = sessions["session_id"].tolist()
    if session_options:
        # Label with repo + PR so the search-by-typing dropdown can find a
        # session -- the raw UUID alone is unsearchable.
        session_labels = {
            row.session_id: (
                f"{row.repo}" + (f" · PR #{int(row.pr_number)}" if pd.notna(row.pr_number) else "")
                + f" · {row.session_id[:8]}"
            )
            for row in sessions.itertuples()
        }
        chosen = st.selectbox(
            "Pick a session", options=session_options,
            format_func=lambda sid: session_labels[sid],
            index=None, placeholder="Search by repo, PR, or session id...",
        )
        if chosen:
            sess_all = df[df["session_id"] == chosen]
            # Main-loop turns only -- subagent turns share the parent's
            # session_id and can overlap in wall-clock time, so mixing them
            # into one "turn N" axis would be meaningless.
            sess_df = sess_all[~sess_all["is_subagent"]].sort_values("timestamp").reset_index(drop=True)
            sess_df["turn"] = sess_df.index + 1
            sess_df["cumulative_cost"] = sess_df["total_cost"].cumsum()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sess_df["turn"], y=sess_df["cumulative_cost"],
                mode="lines", line=dict(color=ACCENT, width=2), name="Cumulative cost",
            ))
            fig.update_layout(
                title=f"Cumulative cost across {len(sess_df)} main-loop turns",
                xaxis_title="Turn", yaxis_title="Cumulative cost ($)",
            )
            style_axes(fig)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

            subagents = sess_all[sess_all["is_subagent"]]
            if not subagents.empty:
                st.caption(
                    f"Spawned {subagents['agent_id'].nunique()} subagent(s), "
                    f"costing ${subagents['total_cost'].sum():,.2f} total."
                )
                agent_summary = (
                    subagents.groupby("agent_id", as_index=False)
                    .agg(
                        description=("agent_description", "first"),
                        agent_type=("agent_type", "first"),
                        model=("model", "first"),
                        turns=("total_cost", "count"),
                        cost=("total_cost", "sum"),
                    )
                    .sort_values("cost", ascending=False)
                )
                agent_summary["description"] = agent_summary["description"].fillna("(no description)")
                st.dataframe(
                    agent_summary[["description", "agent_type", "model", "turns", "cost"]],
                    width='stretch', height=min(320, 60 + 35 * len(agent_summary)),
                    column_config={
                        "description": st.column_config.TextColumn("Subagent"),
                        "agent_type": st.column_config.TextColumn("Type"),
                        "cost": st.column_config.NumberColumn("Cost", format="$%.2f"),
                    },
                )

    st.divider()
    st.subheader("Browse raw usage events")

    min_cost = st.number_input(
        "Min cost per turn ($)", min_value=0.0, value=0.0, step=0.01,
        help="Narrows the table below only -- the leaderboard and session detail are unaffected.",
        width=220,
    )
    filtered = df[df["total_cost"] >= min_cost] if min_cost > 0 else df

    # Paginate by week so the table only ever shows a single week at a time.
    week_starts = (
        filtered["ts_local"].dt.to_period("W").dt.start_time
        if not filtered.empty else pd.Series([], dtype="datetime64[ns]")
    )
    unique_weeks = sorted(week_starts.dropna().unique(), reverse=True)
    if unique_weeks:
        week_labels = {
            w: f"{pd.Timestamp(w):%b %d, %Y} – {pd.Timestamp(w) + pd.Timedelta(days=6):%b %d, %Y}"
            for w in unique_weeks
        }
        selected_week = st.selectbox(
            "Week",
            unique_weeks,
            format_func=lambda w: week_labels[w],
            help="The raw events table shows one week at a time.",
        )
        week_mask = week_starts == selected_week
        filtered = filtered[week_mask.values]

    st.caption(f"{len(filtered):,} rows match")
    st.dataframe(
        filtered.sort_values("timestamp", ascending=False)[
            ["timestamp", "model", "repo", "branch", "pr_url",
             "input_tokens", "output_tokens", "thinking_tokens", "total_cost"]
        ],
        width='stretch',
        height=320,
        column_config={
            "pr_url": st.column_config.LinkColumn("PR", display_text=r".*/pull/(\d+)$"),
        },
    )


def compute_metrics(df):
    """Everything the insights tab and the suggestion engine both need."""
    m = {}
    m["total_cost"] = df["total_cost"].sum()

    cache_read = df["cache_read_input_tokens"].sum()
    cache_write = df["cache_creation_5m_tokens"].sum() + df["cache_creation_1h_tokens"].sum()
    uncached_input = df["input_tokens"].sum()
    effective_input = cache_read + cache_write + uncached_input
    m["cache_hit_rate"] = (cache_read / effective_input) if effective_input else None

    m["cache_read_cost"] = df["cache_read_cost"].sum()
    m["cache_write_cost"] = df["cache_write_cost"].sum()
    m["input_cost"] = df["input_cost"].sum()
    m["output_cost"] = df["output_cost"].sum()

    thinking = df["thinking_tokens"].sum()
    output = df["output_tokens"].sum()
    m["thinking_ratio"] = (thinking / output) if output else None

    per_session = df.groupby("session_id")["total_cost"].sum().sort_values(ascending=False)
    m["per_session"] = per_session
    n_sessions = len(per_session)
    if n_sessions >= 10:
        top_decile_n = max(1, n_sessions // 10)
        m["top_decile_share"] = per_session.head(top_decile_n).sum() / per_session.sum()
    else:
        m["top_decile_share"] = None
    m["max_session_cost"] = per_session.iloc[0] if n_sessions else 0
    m["max_session_share"] = (m["max_session_cost"] / m["total_cost"]) if m["total_cost"] else None

    per_model = df.groupby("model", as_index=False).agg(
        turns=("model", "count"), cost=("total_cost", "sum")
    )
    per_model["cost_per_turn"] = per_model["cost"] / per_model["turns"]
    m["per_model"] = per_model.sort_values("cost", ascending=False)

    per_repo = df.groupby("repo", as_index=False).agg(
        cost=("total_cost", "sum"), sessions=("session_id", "nunique")
    )
    per_repo["avg_per_session"] = per_repo["cost"] / per_repo["sessions"]
    m["per_repo"] = per_repo.sort_values("cost", ascending=False)

    return m


def generate_suggestions(m):
    """Rule-based findings, mirroring the manual cost-profile analysis."""
    suggestions = []

    if m["cache_hit_rate"] is not None:
        if m["cache_hit_rate"] < 0.70:
            suggestions.append(dict(
                severity="serious",
                title="Low prompt-cache hit rate",
                detail=(
                    f"Only {m['cache_hit_rate']*100:.1f}% of effective input tokens are served "
                    "from cache. Look for a cache-breaking pattern: dynamic content above the "
                    "cache breakpoint, frequent model/effort switches, or gaps between requests "
                    "longer than the cache TTL."
                ),
            ))
        elif m["cache_hit_rate"] < 0.85:
            suggestions.append(dict(
                severity="warning",
                title="Cache hit rate has room to improve",
                detail=(
                    f"{m['cache_hit_rate']*100:.1f}% hit rate is decent but below the healthy-loop "
                    "range (usually >90%). Worth a look, not urgent."
                ),
            ))
        else:
            suggestions.append(dict(
                severity="good",
                title="Prompt caching is healthy",
                detail=f"{m['cache_hit_rate']*100:.1f}% cache hit rate -- this is not a cost lever right now.",
            ))

    if m["top_decile_share"] is not None and m["top_decile_share"] > 0.40:
        suggestions.append(dict(
            severity="serious",
            title="Cost is concentrated in a few very long sessions",
            detail=(
                f"The top 10% of sessions account for {m['top_decile_share']*100:.1f}% of total "
                f"spend. The single most expensive session cost ${m['max_session_cost']:.2f} "
                f"({(m['max_session_share'] or 0)*100:.1f}% of the whole bill). Conversation cost "
                "compounds with turn count because every turn re-reads the growing history via "
                "cache -- breaking up marathon sessions (compact, /clear, or delegate exploration "
                "to a subagent with its own smaller context) is usually the biggest available lever."
            ),
        ))

    if m["thinking_ratio"] is not None and m["thinking_ratio"] > 0.40:
        suggestions.append(dict(
            severity="warning",
            title="Thinking tokens are a large share of output spend",
            detail=(
                f"Thinking tokens are {m['thinking_ratio']*100:.1f}% of all output tokens. "
                "Sweeping effort down for routine tasks may cut this meaningfully on workloads "
                "that don't need deep reasoning -- validate on a few real tasks before trusting it, "
                "since lowering effort is a real capability tradeoff, not a free win."
            ),
        ))

    per_repo = m["per_repo"]
    if len(per_repo) > 1:
        richest = per_repo.iloc[0]
        if richest["sessions"] <= 5 and richest["avg_per_session"] > 2 * per_repo["avg_per_session"].median():
            suggestions.append(dict(
                severity="warning",
                title=f"'{richest['repo']}' has a high average cost per session",
                detail=(
                    f"${richest['avg_per_session']:.2f}/session across only {richest['sessions']} "
                    "session(s) -- check whether this is a systemic pattern or a single outlier "
                    "session before treating it as a repo-level issue."
                ),
            ))

    priced_models = m["per_model"][m["per_model"]["cost_per_turn"] > 0].sort_values(
        "cost_per_turn", ascending=False
    )
    if len(priced_models) > 1:
        priciest = priced_models.iloc[0]
        cheapest = priced_models.iloc[-1]
        if priciest["cost_per_turn"] > 3 * cheapest["cost_per_turn"] and priciest["turns"] > 20:
            suggestions.append(dict(
                severity="warning",
                title=f"'{priciest['model']}' costs {priciest['cost_per_turn']/cheapest['cost_per_turn']:.1f}x per turn vs your cheapest model",
                detail=(
                    "Worth checking whether the higher-cost model is concentrated in the same "
                    "long sessions flagged above (fix those first) or is a separate, independent "
                    "pattern of model overuse on tasks a cheaper model/lower effort would also solve."
                ),
            ))

    if not suggestions:
        suggestions.append(dict(
            severity="good",
            title="Nothing stands out",
            detail="No concentration, caching, or model-mix pattern crossed the thresholds checked here.",
        ))

    return suggestions


SEVERITY_COLOR = {
    "good": STATUS_GOOD,
    "warning": STATUS_WARNING,
    "serious": STATUS_SERIOUS,
    "critical": STATUS_CRITICAL,
}
SEVERITY_ICON = {"good": "✅", "warning": "⚠️", "serious": "🔴", "critical": "🔴"}


def tab_insights(df):
    m = compute_metrics(df)

    st.subheader("Optimization suggestions")
    for s in generate_suggestions(m):
        icon = SEVERITY_ICON[s["severity"]]
        st.markdown(f"**{icon} {s['title']}**")
        st.caption(s["detail"])

    st.divider()
    st.subheader("Cost composition")
    composition = pd.DataFrame({
        "component": ["Cache read", "Cache write", "Uncached input", "Output"],
        "cost": [m["cache_read_cost"], m["cache_write_cost"], m["input_cost"], m["output_cost"]],
    })
    fig = px.bar(
        composition, x="cost", y=["Cost"] * len(composition), color="component",
        orientation="h", color_discrete_sequence=CATEGORICAL,
        text=composition["cost"].map(lambda v: f"${v:,.2f}"),
        labels={"cost": "Cost ($)", "y": ""},
    )
    # Force dark text on every segment -- Plotly's auto-contrast text picks
    # one color per trace, not per segment, and gets some hues wrong here.
    fig.update_traces(textposition="inside", textfont=dict(color=INK_PRIMARY, size=13))
    style_axes(fig)
    fig.update_layout(height=180, yaxis=dict(showticklabels=False))
    st.plotly_chart(fig, width='stretch')
    if m["cache_hit_rate"] is not None:
        st.caption(f"Cache hit rate: {m['cache_hit_rate']*100:.1f}% of effective input tokens")

    left, right = st.columns(2)
    with left:
        st.subheader("Session cost concentration")
        per_session = m["per_session"].reset_index()
        per_session.columns = ["session_id", "cost"]
        per_session = per_session.sort_values("cost", ascending=False).reset_index(drop=True)
        per_session["cumulative_share"] = per_session["cost"].cumsum() / per_session["cost"].sum()
        per_session["session_rank"] = range(1, len(per_session) + 1)
        fig = px.area(
            per_session, x="session_rank", y="cumulative_share",
            color_discrete_sequence=[ACCENT],
            labels={"session_rank": "Sessions, most expensive first", "cumulative_share": "Cumulative share of total cost"},
        )
        fig.update_yaxes(tickformat=".0%")
        style_axes(fig)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with right:
        st.subheader("Cost per turn by model")
        fig = px.bar(
            m["per_model"], x="model", y="cost_per_turn",
            color_discrete_sequence=[ACCENT],
            labels={"cost_per_turn": "Avg cost / turn ($)", "model": ""},
        )
        style_axes(fig)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')


def tab_prs(df):
    pr_df = df[df["pr_number"].notna()].copy()
    coverage = len(pr_df) / len(df) if len(df) else 0
    st.caption(
        f"{len(pr_df):,} of {df.shape[0]:,} events ({coverage*100:.0f}%) are attributed to a PR. "
        "The rest are on branches with no open/merged PR yet, on `main`/`HEAD`, or not yet "
        "resolved by the background enrichment pass -- PR attribution is best-effort, not complete."
    )

    if pr_df.empty:
        st.info("No events with a resolved PR number in the current filter.")
        return

    pr_df["pr_label"] = pr_df["repo"] + " #" + pr_df["pr_number"].astype(int).astype(str)

    st.subheader("Cost by PR")
    leaderboard = (
        pr_df.groupby(["pr_label", "repo", "pr_number"], as_index=False)
        .agg(
            pr_url=("pr_url", "first"),
            cost=("total_cost", "sum"),
            turns=("total_cost", "count"),
            sessions=("session_id", "nunique"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .sort_values("cost", ascending=False)
    )
    st.dataframe(
        leaderboard[["pr_label", "pr_url", "cost", "turns", "sessions", "first_seen", "last_seen"]],
        width='stretch', height=320,
        column_config={
            "pr_url": st.column_config.LinkColumn("Open on GitHub", display_text="Open ↗"),
        },
    )

    st.divider()
    st.subheader("Look up a PR")
    chosen = st.selectbox(
        "Search or pick a PR",
        options=leaderboard["pr_label"].tolist(),
        index=None,
        placeholder="Type a PR number or repo name...",
    )
    if not chosen:
        return

    pr_events = pr_df[pr_df["pr_label"] == chosen]
    total_cost = pr_events["total_cost"].sum()
    sessions = pr_events["session_id"].nunique()
    span = f"{pr_events['date'].min().date()} to {pr_events['date'].max().date()}"
    subagent_cost = pr_events.loc[pr_events["is_subagent"], "total_cost"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cost", f"${total_cost:,.2f}")
    c2.metric("Sessions", sessions)
    c3.metric("Worked on", span)
    c4.metric("Subagent cost", f"${subagent_cost:,.2f}")

    left, right = st.columns(2)
    with left:
        by_model = pr_events.groupby("model", as_index=False)["total_cost"].sum().sort_values(
            "total_cost", ascending=True
        )
        fig = px.bar(
            by_model, x="total_cost", y="model", orientation="h",
            color_discrete_sequence=[ACCENT],
            labels={"total_cost": "Cost ($)", "model": ""},
        )
        style_axes(fig)
        fig.update_layout(showlegend=False, height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

    with right:
        by_day = pr_events.groupby("date", as_index=False)["total_cost"].sum()
        fig = px.bar(
            by_day, x="date", y="total_cost",
            color_discrete_sequence=[ACCENT],
            labels={"total_cost": "Cost ($)", "date": ""},
        )
        style_axes(fig)
        fig.update_layout(showlegend=False, height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')


def render_time_heatmap(df):
    st.subheader("When you spend")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grid = (
        df.groupby(["dow", "hour"])["total_cost"].sum()
        .unstack("hour")
        .reindex(index=dow_order, columns=range(24))
        .fillna(0.0)
    )
    fig = px.imshow(
        grid, color_continuous_scale=SEQUENTIAL_RAMP, aspect="auto",
        labels=dict(x="Hour of day (local)", y="", color="Cost ($)"),
    )
    style_axes(fig, spikes=False)
    fig.update_layout(height=280, coloraxis_colorbar=dict(title=""))
    st.plotly_chart(fig, width='stretch')


def render_cost_per_turn_trend(df):
    st.subheader("Cost per turn over time")
    weekly = df.copy()
    weekly["week"] = weekly["date"].dt.to_period("W").dt.start_time
    weekly = weekly.groupby("week", as_index=False).agg(
        cost=("total_cost", "sum"), turns=("total_cost", "count")
    )
    weekly["cost_per_turn"] = weekly["cost"] / weekly["turns"]
    fig = px.line(
        weekly, x="week", y="cost_per_turn", markers=True,
        color_discrete_sequence=[ACCENT],
        labels={"week": "Week", "cost_per_turn": "Avg cost / turn ($)"},
    )
    style_axes(fig)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Weekly average, all models blended -- a rising line means turns are "
        "getting more expensive over time (context growth, effort creep), "
        "independent of how many turns you're running."
    )


def render_cache_by_repo(df):
    st.subheader("Cache economics by repo")
    g = df.groupby("repo", as_index=False).agg(
        turns=("model", "count"),
        cache_read=("cache_read_input_tokens", "sum"),
        cache_write=("cache_write_tokens", "sum"),
        uncached=("input_tokens", "sum"),
        cost=("total_cost", "sum"),
    )
    effective = g["cache_read"] + g["cache_write"] + g["uncached"]
    g["cache_hit_rate_%"] = (g["cache_read"] / effective.where(effective > 0) * 100).round(1)
    g = g.sort_values("cost", ascending=False)
    st.dataframe(
        g[["repo", "turns", "cache_hit_rate_%", "cost"]],
        width='stretch', height=250,
    )
    st.caption(
        "The aggregate cache hit rate (Insights tab) can hide this: a repo "
        "worked on in short, spaced-out bursts misses the 5-minute cache far "
        "more often than one worked on continuously, even if the overall "
        "number looks healthy."
    )


def render_worst_cache_sessions(df):
    st.subheader("Sessions with the weakest cache economics")
    g = df.groupby(["session_id", "repo"], as_index=False).agg(
        turns=("model", "count"),
        cache_read=("cache_read_input_tokens", "sum"),
        cache_write=("cache_write_tokens", "sum"),
        uncached=("input_tokens", "sum"),
        cost=("total_cost", "sum"),
    )
    g["effective_tokens"] = g["cache_read"] + g["cache_write"] + g["uncached"]
    g["cache_hit_rate_%"] = (g["cache_read"] / g["effective_tokens"].where(g["effective_tokens"] > 0) * 100)

    # Only rank sessions with enough token volume for the rate to mean anything.
    MIN_EFFECTIVE_TOKENS = 100_000
    meaningful = g[g["effective_tokens"] > MIN_EFFECTIVE_TOKENS].copy()
    if meaningful.empty:
        st.info(
            f"No sessions clear the {MIN_EFFECTIVE_TOKENS:,}-effective-token volume "
            "threshold in the current filter."
        )
        return
    worst = meaningful.sort_values("cache_hit_rate_%", ascending=True).head(10)
    worst = worst.assign(**{"cache_hit_rate_%": worst["cache_hit_rate_%"].round(1)})
    st.dataframe(
        worst[["session_id", "repo", "turns", "cache_hit_rate_%", "cost"]],
        width='stretch', height=280,
    )
    st.caption(
        f"Ranked among the {len(meaningful):,} sessions (of {len(g):,} total) with "
        f"over {MIN_EFFECTIVE_TOKENS:,} effective input tokens -- smaller sessions "
        "don't have enough volume for a hit rate to be meaningful."
    )


def render_session_pacing(df):
    st.subheader("Session pacing -- continuous work vs. long idle gaps")
    st.caption(
        "The top-cost sessions, broken down by wall-clock duration and the "
        "single longest gap between turns. Two 300-turn sessions can have "
        "very different problems: one that's continuous for 3 hours is "
        "likely a genuinely large task; one with a 90-minute gap in the "
        "middle paid to re-warm the cache when it resumed, for no reason "
        "the task itself required."
    )
    top_session_ids = df.groupby("session_id")["total_cost"].sum().sort_values(ascending=False).head(15).index

    rows = []
    for sid in top_session_ids:
        sess = df[df["session_id"] == sid].sort_values("ts_local")
        gaps = sess["ts_local"].diff().dropna()
        duration = sess["ts_local"].iloc[-1] - sess["ts_local"].iloc[0]
        max_gap = gaps.max() if len(gaps) else pd.Timedelta(0)
        rows.append(dict(
            session_id=sid,
            repo=sess["repo"].iloc[0],
            turns=len(sess),
            cost=sess["total_cost"].sum(),
            duration_hours=round(duration.total_seconds() / 3600, 1),
            max_gap_minutes=round(max_gap.total_seconds() / 60, 1),
        ))
    pacing = pd.DataFrame(rows).sort_values("cost", ascending=False)
    st.dataframe(pacing, width='stretch', height=380)


ASK_MODEL = "claude-opus-5"
ASK_TIMEOUT_SECONDS = 180
ASK_MCP_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ask_mcp_server.py")
# The MCP tool as Claude Code names it: mcp__<server name>__<tool name>.
ASK_CLI_TOOL = "mcp__usage__run_sql"


class AskError(Exception):
    """A user-presentable Ask failure: what went wrong, what to do about it,
    and (optionally) raw output for the "Technical details" expander."""

    def __init__(self, title, message, hint=None, details=None):
        super().__init__(f"{title}: {message}")
        self.title, self.message, self.hint, self.details = title, message, hint, details

    def as_dict(self):
        return {"title": self.title, "message": self.message,
                "hint": self.hint, "details": self.details}


def _render_ask_error(err):
    body = f"**{err['title']}**\n\n{err['message']}"
    if err.get("hint"):
        body += f"\n\n{err['hint']}"
    st.error(body, icon=":material/error:")
    if err.get("details"):
        with st.expander("Technical details"):
            st.code(err["details"], language=None)


def _ask_system_prompt():
    return (
        "You are a data analyst answering questions about the user's personal Claude Code "
        f"token-usage and cost history, stored in a local SQLite database. Today's date is "
        f"{dt.date.today().isoformat()}.\n{USAGE_EVENTS_SCHEMA}\n"
        "Always use the run_sql tool to answer -- never guess or estimate numbers. "
        "Keep answers to a few sentences with concrete figures (dollars, tokens, counts)."
    )


# --- Backend 1: Anthropic API (preferred, needs ANTHROPIC_API_KEY) ----------

@beta_tool
def run_sql(sql: str) -> str:
    """Run a read-only SQL query against the usage_events table and return the results as JSON.

    Args:
        sql: A single SELECT (or WITH ... SELECT) statement against usage_events.
            No INSERT/UPDATE/DELETE/DDL and no multiple statements.
    """
    return run_sql_query(DB_PATH, sql)


def ask_claude_api(question):
    """One-shot Q&A over usage_events via the Tool Runner. Returns (answer_text, sql_queries_run)."""
    client = Anthropic()
    runner = client.beta.messages.tool_runner(
        model=ASK_MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=_ask_system_prompt(),
        tools=[run_sql],
        messages=[{"role": "user", "content": question}],
    )

    queries_run = []
    last = None
    try:
        for message in runner:
            last = message
            for block in message.content:
                if block.type == "tool_use" and block.name == "run_sql":
                    queries_run.append(block.input.get("sql", ""))
    except Exception as e:
        raise AskError("The Claude API request failed", str(e)) from e

    answer = next((b.text for b in last.content if b.type == "text"), "") if last else ""
    return answer, queries_run


# --- Backend 2: Claude Code CLI (fallback, uses its own login) --------------

def _find_claude_cli():
    """Absolute path to the `claude` CLI, checking PATH and the usual install
    spots (the dashboard may be started from a shell with a trimmed PATH)."""
    import shutil
    found = shutil.which("claude")
    if found:
        return found
    for candidate in (
        os.path.expanduser("~/.claude/local/claude"),
        os.path.expanduser("~/.local/bin/claude"),
        "/opt/homebrew/bin/claude",
        "/usr/local/bin/claude",
    ):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _claude_cli_env():
    """Environment for spawning the CLI. Strips API credentials so it
    authenticates with its own login (otherwise a stray key would silently be
    billed instead), and the nested-session marker so it doesn't refuse to
    start when the dashboard itself was launched from inside Claude Code."""
    env = dict(os.environ)
    for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDECODE"):
        env.pop(key, None)
    return env


_CLI_AUTH_HINT = "Run `claude auth login` in a terminal, then click **Re-check**."


@st.cache_data(ttl=60, show_spinner=False)
def _claude_cli_status():
    """Preflight: is the CLI installed and logged in? Returns
    ("ok", status_dict) or ("error", AskError.as_dict()). Cached briefly so it
    isn't re-run on every rerun; only reads local credentials (no API call)."""
    cli = _find_claude_cli()
    if cli is None:
        return "error", AskError(
            "Claude Code CLI not found",
            "Without an API key, Ask answers through the Claude Code CLI, but "
            "no `claude` executable was found on PATH.",
            "Install Claude Code (https://docs.claude.com/en/docs/claude-code), "
            "log in, then click **Re-check** -- or set `ANTHROPIC_API_KEY`.",
        ).as_dict()
    try:
        proc = subprocess.run(
            [cli, "auth", "status", "--json"], env=_claude_cli_env(),
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return "error", AskError(
            "Couldn't run the Claude Code CLI",
            f"Checking the login status with `{cli}` failed.",
            None, str(e),
        ).as_dict()
    try:
        status = json.loads(proc.stdout)
    except ValueError:
        status = None
    if not isinstance(status, dict):
        return "error", AskError(
            "Couldn't read the Claude Code login status",
            "`claude auth status` didn't return the expected JSON. The installed "
            "CLI may be too old for this dashboard.",
            "Update Claude Code (`claude update`), then click **Re-check**.",
            (proc.stdout + proc.stderr).strip() or f"exit code {proc.returncode}",
        ).as_dict()
    if not status.get("loggedIn"):
        return "error", AskError(
            "Claude Code isn't logged in",
            "Without an API key, Ask answers through your Claude Code login, "
            "but the CLI reports no active login.",
            _CLI_AUTH_HINT,
        ).as_dict()
    status["cli_path"] = cli
    return "ok", status


def _classify_cli_error(text, api_status):
    """Map a failed CLI result onto a friendly AskError."""
    lowered = (text or "").lower()
    if api_status in (401, 403) or any(s in lowered for s in (
        "not logged in", "/login", "oauth", "authentication", "invalid api key",
        "credentials",
    )):
        return AskError(
            "Claude Code couldn't authenticate",
            "Your Claude Code login was rejected -- it may have expired or been revoked.",
            _CLI_AUTH_HINT, text,
        )
    if api_status == 404 or "selected model" in lowered:
        return AskError(
            "Model not available",
            f"Claude Code couldn't use `{ASK_MODEL}` -- it may not be enabled "
            "for your account or organization.",
            "Ask your Claude admin about access, or set `ANTHROPIC_API_KEY` to "
            "use the API instead.",
            text,
        )
    if api_status == 429 or any(s in lowered for s in ("rate limit", "usage limit", "limit reached")):
        return AskError(
            "Usage limit reached",
            "Claude Code reports that your plan's usage or rate limit has been hit.",
            "Wait a bit and try again, or set `ANTHROPIC_API_KEY` to use the API instead.",
            text,
        )
    return AskError("Claude Code returned an error", "The request didn't complete.", None, text)


def ask_claude_cli(question):
    """Answer via `claude -p`, sandboxed to a single read-only SQL tool served
    by ask_mcp_server.py. Returns (answer_text, sql_queries_run); raises
    AskError with a presentable message on any failure."""
    ok, status = _claude_cli_status()
    if ok != "ok":
        raise AskError(**status)

    mcp_config = {"mcpServers": {"usage": {
        "type": "stdio", "command": sys.executable, "args": [ASK_MCP_SERVER, DB_PATH],
    }}}
    cmd = [
        status["cli_path"], "-p",
        "--output-format", "stream-json", "--verbose",
        "--model", ASK_MODEL,
        "--system-prompt", _ask_system_prompt(),
        # No built-in tools (Bash, file access, web...), only our MCP server,
        # pre-approved so print mode never stalls on a permission prompt.
        "--tools", "",
        "--mcp-config", json.dumps(mcp_config), "--strict-mcp-config",
        "--allowedTools", ASK_CLI_TOOL,
        # Keeps Ask's own turns out of ~/.claude/projects, and so out of usage.db.
        "--no-session-persistence",
        "--setting-sources", "", "--disable-slash-commands",
    ]
    try:
        proc = subprocess.Popen(
            cmd, env=_claude_cli_env(), cwd=os.path.dirname(DB_PATH),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as e:
        raise AskError("Couldn't start the Claude Code CLI", str(e)) from e

    import threading
    timed_out = threading.Event()

    def _on_timeout():
        timed_out.set()
        proc.kill()

    timer = threading.Timer(ASK_TIMEOUT_SECONDS, _on_timeout)
    timer.start()
    # Drain stderr concurrently so a chatty CLI can't block on a full pipe.
    stderr_chunks = []
    stderr_reader = threading.Thread(
        target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True,
    )
    stderr_reader.start()

    queries_run, result, mcp_failure = [], None, None
    try:
        # The prompt goes over stdin, not argv, so a question starting with
        # "-" can't be parsed as a flag.
        proc.stdin.write(question)
        proc.stdin.close()
        for line in proc.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "system" and event.get("subtype") == "init":
                servers = {s.get("name"): s.get("status") for s in event.get("mcp_servers") or []}
                # With the tool server down, the CLI carries on tool-less and
                # the model improvises -- never let that reach the user.
                if servers.get("usage") != "connected":
                    mcp_failure = servers.get("usage") or "missing"
                    proc.kill()
                    break
            elif kind == "assistant":
                for block in (event.get("message") or {}).get("content") or []:
                    if block.get("type") == "tool_use" and block.get("name") == ASK_CLI_TOOL:
                        queries_run.append((block.get("input") or {}).get("sql", ""))
            elif kind == "result":
                result = event
        proc.wait()
    finally:
        timer.cancel()
    stderr_reader.join(timeout=2)
    stderr = "".join(stderr_chunks).strip()

    if timed_out.is_set():
        raise AskError(
            "Claude Code timed out",
            f"No answer after {ASK_TIMEOUT_SECONDS} seconds, so the request was stopped.",
            "Try again, or try a narrower question.",
            stderr or None,
        )
    if mcp_failure:
        raise AskError(
            "The usage database tool didn't start",
            f"Claude Code couldn't launch the SQL tool server (status: {mcp_failure}), "
            "so the question wasn't sent.",
            "Check that the dashboard's install is complete (re-run `claude-usage --update`).",
            f"{sys.executable} {ASK_MCP_SERVER} {DB_PATH}\n\n{stderr}".strip(),
        )
    if result is None:
        raise AskError(
            "Claude Code exited unexpectedly",
            f"The CLI stopped without returning an answer (exit code {proc.returncode}).",
            "If this keeps happening, try `claude update`.",
            stderr or None,
        )
    if result.get("is_error"):
        raise _classify_cli_error(result.get("result") or stderr, result.get("api_error_status"))
    return result.get("result") or "", queries_run


# --- Tab ----------------------------------------------------------------------

def tab_ask():
    st.subheader("Ask about your usage")

    use_api = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_api:
        st.caption(
            "Queries your full history directly via SQL, independent of the filter bar above -- "
            "and calls the real Claude API, which incurs its own (small) cost not logged in this DB."
        )
    else:
        st.info(
            "No `ANTHROPIC_API_KEY` is set, so Ask answers through your **Claude Code "
            "login** instead. That works, but an API key is the better setup: answers "
            "come back faster (no CLI process to start per question) and don't depend "
            "on the installed Claude Code version or its login staying valid. Set "
            "`ANTHROPIC_API_KEY` and restart the dashboard to switch.",
            icon=":material/info:",
        )
        ok, status = _claude_cli_status()
        if ok != "ok":
            _render_ask_error(status)
            if st.button("Re-check", key="ask-recheck"):
                _claude_cli_status.clear()
                st.rerun()
            return
        who = status.get("email") or "your account"
        org = status.get("orgName")
        plan = status.get("subscriptionType")
        extra = ", ".join(p for p in (org, plan and plan.capitalize()) if p)
        st.caption(
            f"Queries your full history directly via SQL, independent of the filter bar above. "
            f"Answering via Claude Code as **{who}**" + (f" ({extra})" if extra else "") + "."
        )

    # Spinner-in-place-of-button across reruns via session_state: each pass
    # renders either the form or the spinner, never both.
    pending = st.session_state.get("ask_pending")

    if pending:
        with st.spinner("Thinking..."):
            try:
                answer, queries = (ask_claude_api if use_api else ask_claude_cli)(pending)
                st.session_state.ask_result = (answer, queries)
                st.session_state.ask_error = None
            except AskError as e:
                st.session_state.ask_result = None
                st.session_state.ask_error = e.as_dict()
            except Exception as e:
                st.session_state.ask_result = None
                st.session_state.ask_error = AskError(
                    "Something went wrong", str(e) or type(e).__name__,
                ).as_dict()
        st.session_state.ask_pending = None
        st.rerun()
    else:
        with st.form("ask_form", clear_on_submit=False):
            question = st.text_input(
                "Ask a question",
                placeholder="e.g. Which repo had the highest cache-write cost last month?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Ask", type="primary")
        if submitted and question:
            st.session_state.ask_pending = question
            st.rerun()

    if st.session_state.get("ask_error"):
        _render_ask_error(st.session_state.ask_error)
        return

    result = st.session_state.get("ask_result")
    if not result:
        return
    answer, queries = result

    # Escape "$" -- st.markdown treats pairs of it as inline LaTeX delimiters,
    # which mangles multiple dollar-cost figures in the same answer.
    st.markdown((answer or "_No answer returned._").replace("$", "\\$"))
    if queries:
        with st.expander("SQL run"):
            for q in queries:
                st.code(q, language="sql")


def tab_trends(df):
    render_time_heatmap(df)
    st.divider()
    render_cost_per_turn_trend(df)
    st.divider()

    left, right = st.columns(2)
    with left:
        render_cache_by_repo(df)
    with right:
        render_worst_cache_sessions(df)
    st.divider()

    render_session_pacing(df)


# --- First-run ingest / rebuild orchestration -------------------------------
#
# Ingest is normally run on a schedule by launchd, but on a fresh install the
# database is empty (or absent) until that first run happens. Rather than show a
# bare error, the dashboard kicks off ingest itself and plays a build animation
# until it completes. The same path backs the sidebar "Purge & rebuild" action.

INGEST_MESSAGES = [
    "Scanning Claude Code transcripts\u2026",
    "Parsing token usage\u2026",
    "Pricing sessions against the rate card\u2026",
    "Resolving pull requests\u2026",
    "Assembling charts and trends\u2026",
]


def _db_has_rows():
    """True only if usage_events exists AND holds at least one row. A file can
    exist with the schema but no data mid-first-ingest."""
    if not os.path.exists(DB_PATH):
        return False
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        row = conn.execute("SELECT 1 FROM usage_events LIMIT 1").fetchone()
        conn.close()
        return row is not None
    except sqlite3.Error:
        return False


def _ingest_in_progress():
    """True if some other process holds the ingest lock (e.g. a scheduled run,
    or an ingest we launched that survived a browser refresh). Probing the
    advisory flock is how we detect it without a PID handle."""
    if not os.path.exists(LOCK_PATH):
        return False
    try:
        fd = open(LOCK_PATH, "r+")
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        fd.close()


def _read_progress():
    """(fraction, label) from ingest's progress file, or None if unavailable."""
    try:
        with open(PROGRESS_PATH) as fh:
            data = json.load(fh)
        return float(data.get("fraction", 0.0)), str(data.get("label", ""))
    except (OSError, ValueError, TypeError):
        return None


def _start_ingest():
    """Launch ingest.py as a detached subprocess pointed at our data dir."""
    # Clear any stale progress so we don't briefly show the last run's 100%.
    try:
        os.remove(PROGRESS_PATH)
    except OSError:
        pass
    env = dict(os.environ)
    env["CLAUDE_USAGE_DATA_DIR"] = os.path.dirname(DB_PATH)
    proc = subprocess.Popen(
        [sys.executable, INGEST_SCRIPT],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    st.session_state["ingest_started"] = time.time()
    return proc


def _build_animation_html():
    """An abstract SVG of a dashboard being constructed: data blocks feed in
    along a track, bars rise from the baseline, a trend line draws across their
    tops, and a scan beam sweeps the frame. Pure CSS/SVG, loops forever."""
    bars = "".join(
        f'<rect class="ib-bar" x="{x}" y="{190 - h}" width="34" height="{h}" rx="5" '
        f'fill="url(#ibBar)" style="animation-delay:{d}s"/>'
        for x, h, d in [
            (60, 70, 0.0), (120, 110, 0.15), (180, 90, 0.30),
            (240, 140, 0.45), (300, 120, 0.60), (360, 160, 0.75),
        ]
    )
    nodes = "".join(
        f'<circle class="ib-node" cx="{cx}" cy="{cy}" r="4" fill="{ACCENT_MAGENTA}" '
        f'style="animation-delay:{d}s"/>'
        for cx, cy, d in [
            (77, 120, 0.3), (137, 80, 0.6), (197, 100, 0.9),
            (257, 50, 1.2), (317, 70, 1.5), (377, 30, 1.8),
        ]
    )
    feed = "".join(
        f'<rect class="ib-feed" x="0" y="{y}" width="22" height="22" rx="5" '
        f'fill="{ACCENT}" style="animation-delay:{d}s"/>'
        for y, d in [(30, 0.0), (30, 0.95), (30, 1.9)]
    )
    return f"""
    <style>
    .ingest-stage {{ display:flex; justify-content:center; padding:1.4rem 0 0.4rem; }}
    .ingest-stage svg {{ width:min(540px,92%); height:auto; }}
    .ib-grid {{ stroke:{GRIDLINE}; stroke-width:1.2; stroke-dasharray:480;
                stroke-dashoffset:480; animation:ibGrid 1.6s ease forwards; }}
    @keyframes ibGrid {{ to {{ stroke-dashoffset:0; }} }}
    .ib-bar {{ transform-box:fill-box; transform-origin:50% 100%;
               animation:ibGrow 2.2s cubic-bezier(.5,0,.2,1) infinite alternate; }}
    @keyframes ibGrow {{ 0% {{ transform:scaleY(.12); opacity:.5; }}
                         100% {{ transform:scaleY(1); opacity:1; }} }}
    .ib-line {{ fill:none; stroke:{ACCENT_MAGENTA}; stroke-width:2.5;
                stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:720;
                stroke-dashoffset:720; animation:ibDraw 3.6s ease-in-out infinite; }}
    @keyframes ibDraw {{ 0% {{ stroke-dashoffset:720; }} 55% {{ stroke-dashoffset:0; }}
                         85% {{ stroke-dashoffset:0; }} 100% {{ stroke-dashoffset:-720; }} }}
    .ib-node {{ opacity:0; animation:ibPulse 3.6s ease-in-out infinite; }}
    @keyframes ibPulse {{ 0%,42% {{ opacity:0; }} 60% {{ opacity:1; }} 100% {{ opacity:.85; }} }}
    .ib-feed {{ animation:ibFeed 2.85s linear infinite; }}
    @keyframes ibFeed {{ 0% {{ transform:translateX(-30px); opacity:0; }}
                         12% {{ opacity:1; }} 80% {{ opacity:1; }}
                         100% {{ transform:translateX(430px); opacity:0; }} }}
    .ib-scan {{ animation:ibScan 3s ease-in-out infinite; }}
    @keyframes ibScan {{ 0% {{ transform:translateX(0); opacity:0; }}
                         20% {{ opacity:.45; }} 80% {{ opacity:.45; }}
                         100% {{ transform:translateX(430px); opacity:0; }} }}
    @media (prefers-reduced-motion: reduce) {{
      .ib-bar,.ib-line,.ib-node,.ib-feed,.ib-scan,.ib-grid {{ animation:none; }}
      .ib-bar {{ transform:none; }} .ib-line,.ib-node {{ stroke-dashoffset:0; opacity:1; }}
    }}
    </style>
    <div class="ingest-stage">
      <svg viewBox="0 0 440 210" role="img" aria-label="Building your dashboard">
        <defs>
          <linearGradient id="ibBar" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="{ACCENT}"/>
            <stop offset="1" stop-color="{ACCENT_DEEP}"/>
          </linearGradient>
          <linearGradient id="ibBeam" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0" stop-color="{ACCENT_MAGENTA}" stop-opacity="0"/>
            <stop offset=".5" stop-color="{ACCENT_MAGENTA}" stop-opacity=".7"/>
            <stop offset="1" stop-color="{ACCENT_MAGENTA}" stop-opacity="0"/>
          </linearGradient>
        </defs>
        <line class="ib-grid" x1="40" y1="190" x2="410" y2="190"/>
        <line class="ib-grid" x1="40" y1="140" x2="410" y2="140" style="animation-delay:.2s"/>
        <line class="ib-grid" x1="40" y1="90" x2="410" y2="90" style="animation-delay:.4s"/>
        {feed}
        {bars}
        <polyline class="ib-line" points="77,120 137,80 197,100 257,50 317,70 377,30"/>
        {nodes}
        <rect class="ib-scan" x="0" y="20" width="46" height="175" fill="url(#ibBeam)"/>
      </svg>
    </div>
    """


def _run_ingest_with_animation(proc):
    """Render the build animation once, then hold this Streamlit run open while
    ingest runs, refreshing only the progress bar + elapsed line so the SVG
    keeps animating smoothly. `proc` is our subprocess, or None when we're just
    waiting on an ingest another process already started."""
    st.markdown('<div class="eyebrow">Setting things up</div>', unsafe_allow_html=True)
    st.title("Building your usage dashboard")
    st.markdown(_build_animation_html(), unsafe_allow_html=True)
    bar_slot = st.empty()
    caption_slot = st.empty()
    start = st.session_state.get("ingest_started", time.time())

    def render(fraction, label, elapsed):
        bar_slot.progress(fraction, text=label)
        caption_slot.markdown(
            f'<div style="text-align:center; font-family:\'IBM Plex Mono\',monospace; '
            f'font-size:.85rem; color:{MUTED}; margin-top:.35rem;">elapsed {elapsed}s</div>',
            unsafe_allow_html=True,
        )

    def still_running():
        if proc is not None:
            return proc.poll() is None
        return _ingest_in_progress()

    while still_running():
        elapsed = int(time.time() - start)
        prog = _read_progress()
        if prog is not None:
            render(prog[0], prog[1], elapsed)
        else:
            # Ingest hasn't written progress yet -- show an indeterminate hint.
            message = INGEST_MESSAGES[(elapsed // 3) % len(INGEST_MESSAGES)]
            render(0.0, message, elapsed)
        time.sleep(0.5)

    elapsed = int(time.time() - start)
    render(1.0, "Done \u2014 loading your dashboard\u2026", elapsed)
    st.session_state["ingest_completed"] = True
    st.session_state.pop("purge_rebuild_running", None)
    # Already reloading below; stop the header timer triggering a second one.
    st.session_state["ingest_seen_run"] = _read_schedule()[0]
    load_data.clear()
    time.sleep(0.6)
    st.rerun()


@st.dialog("Settings")
def _settings_dialog():
    st.markdown("#### Data")
    st.caption("Rebuild the database from your Claude Code transcripts if "
               "it looks corrupted or out of date. This re-scans every "
               "transcript from scratch and can take a while.")
    if st.button("Purge & rebuild data", use_container_width=True, type="primary"):
        try:
            os.remove(DB_PATH)
        except OSError:
            pass
        load_data.clear()
        st.session_state["ingest_completed"] = False
        st.session_state["force_ingest"] = True
        st.rerun()


def _render_settings_button():
    """A gear icon pinned to the top-right corner that opens the settings
    modal -- keeps the chrome minimal and avoids a whole sidebar for one
    rarely-used action."""
    st.markdown(
        f"""
        <style>
        /* Hide only Streamlit's Deploy button; keep the hamburger menu. */
        [data-testid="stAppDeployButton"] {{ display: none !important; }}

        /* Sit above Streamlit's header (very high z-index) so we're not hidden.
           Placed just below the hamburger menu so the two don't collide. */
        .st-key-settings-gear {{
            position: fixed; top: 3.25rem; right: 0.75rem; z-index: 999999;
            width: auto;
        }}
        .st-key-settings-gear button {{
            border-radius: 50%; width: 2.6rem; height: 2.6rem; padding: 0;
            min-height: 0; border: 1px solid {BORDER};
            background: {SURFACE}; color: {INK_SECONDARY} !important;
            box-shadow: 0 1px 4px rgba(11,6,32,0.08);
            display: inline-flex; align-items: center; justify-content: center;
            transition: color .15s ease, border-color .15s ease, transform .15s ease;
        }}
        .st-key-settings-gear button:hover {{
            color: {ACCENT} !important; border-color: {ACCENT};
            transform: rotate(30deg);
        }}
        .st-key-settings-gear button p {{ color: inherit !important; }}
        .st-key-settings-gear button [data-testid="stIconMaterial"] {{
            font-size: 1.35rem; color: inherit !important;
        }}

        /* Ingest countdown pill, just left of the gear (2.6rem + gap). */
        .st-key-ingest-timer {{
            position: fixed; top: 3.25rem; right: 3.85rem; z-index: 999999;
            width: auto; height: 2.6rem; padding: 0 0.3rem 0 0.9rem;
            border: 1px solid {BORDER}; border-radius: 1.3rem;
            background: {SURFACE}; box-shadow: 0 1px 4px rgba(11,6,32,0.08);
        }}
        /* !important: beats the global start-alignment for horizontal blocks
           and the global filled-purple button style. */
        div.st-key-ingest-timer {{ align-items: center !important; }}
        .st-key-ingest-timer [data-testid="stMarkdownContainer"],
        .st-key-ingest-timer [data-testid="stMarkdownContainer"] p {{ margin: 0 !important; }}
        .ingest-timer__label {{
            font-family: 'IBM Plex Mono', monospace; font-size: .8rem;
            color: {INK_SECONDARY}; white-space: nowrap;
        }}
        .st-key-ingest-timer div[data-testid="stButton"] button {{
            border-radius: 50% !important; width: 2rem; height: 2rem;
            padding: 0 !important; min-height: 0; border: none !important;
            background: transparent !important; box-shadow: none !important;
            color: {ACCENT} !important;
            transition: background-color .15s ease;
        }}
        .st-key-ingest-timer div[data-testid="stButton"] button:hover:not(:disabled) {{
            background: {LAVENDER_PALE} !important;
        }}
        .st-key-ingest-timer div[data-testid="stButton"] button:disabled {{
            color: {MUTED} !important; opacity: .6;
        }}
        .st-key-ingest-timer button p {{ color: inherit !important; }}
        .st-key-ingest-timer button [data-testid="stIconMaterial"] {{
            font-size: 1.2rem; color: inherit !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="settings-gear"):
        if st.button("", icon=":material/settings:", key="settings-gear-btn",
                     help="Settings"):
            _settings_dialog()


def _read_schedule():
    """(last_run, interval_minutes) from ingest's schedule file; either may be
    None (no run recorded yet / no scheduled job configured)."""
    try:
        with open(SCHEDULE_PATH) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None, None
    if not isinstance(data, dict):
        return None, None
    last_run, interval = data.get("last_run"), data.get("interval_minutes")
    return (
        last_run if isinstance(last_run, (int, float)) else None,
        interval if isinstance(interval, int) and interval > 0 else None,
    )


def _manual_ingest_running():
    proc = st.session_state.get("manual_ingest_proc")
    return proc is not None and proc.poll() is None


def _ingest_timer_label(last_run, interval):
    if _manual_ingest_running() or _ingest_in_progress():
        return "Ingesting\u2026"
    if interval is None:
        return None
    if last_run is None:
        return "Next ingest <1m"
    remaining = last_run + interval * 60 - time.time()
    # Ceil so the label only reads "0" territory once it's actually due;
    # launchd polls once a minute, so "<1m" can linger up to a minute.
    minutes = -(-int(remaining) // 60)
    return "Next ingest <1m" if minutes <= 0 else f"Next ingest {minutes}m"


@st.fragment(run_every=5)
def _render_ingest_timer():
    """Countdown to the next scheduled ingest plus an "ingest now" button,
    pinned beside the settings gear. Reruns on its own every few seconds; when
    any ingest (ours or launchd's) finishes, reruns the whole app so charts
    pick up the new rows."""
    last_run, interval = _read_schedule()
    running = _manual_ingest_running() or _ingest_in_progress()

    seen = st.session_state.setdefault("ingest_seen_run", last_run)
    if not running and last_run != seen:
        st.session_state["ingest_seen_run"] = last_run
        st.session_state.pop("manual_ingest_proc", None)
        st.rerun(scope="app")

    label = _ingest_timer_label(last_run, interval)
    tip = "Ingest now and reset the timer"
    if last_run is not None:
        tip += f" (last run {dt.datetime.fromtimestamp(last_run):%H:%M})"

    with st.container(key="ingest-timer", horizontal=True, wrap=False,
                      width="content", vertical_alignment="center", gap="small"):
        if label:
            st.markdown(f'<span class="ingest-timer__label">{label}</span>',
                        unsafe_allow_html=True)
        if st.button("", icon=":material/refresh:", key="ingest-now-btn",
                     help=tip, disabled=running):
            st.session_state["manual_ingest_proc"] = _start_ingest()
            st.rerun(scope="fragment")


def main():
    apply_theme()
    inject_css()

    # The full-screen build animation is reserved for first-time setup (no
    # data yet) and an explicit "Purge & rebuild". Routine ingests (scheduled
    # or the header refresh button) run behind the live dashboard instead; the
    # header timer shows "Ingesting\u2026" and reloads the page when done.
    if st.session_state.pop("force_ingest", False):
        st.session_state["purge_rebuild_running"] = True
        _run_ingest_with_animation(_start_ingest())
        return
    first_setup = not _db_has_rows()
    if _ingest_in_progress() and (
        first_setup or st.session_state.get("purge_rebuild_running")
    ):
        # Re-attach after a rerun mid-build, e.g. the user clicked something.
        _run_ingest_with_animation(None)
        return
    if first_setup and not st.session_state.get("ingest_completed"):
        _run_ingest_with_animation(_start_ingest())
        return

    st.markdown('<div class="eyebrow">Personal cost intelligence</div>', unsafe_allow_html=True)
    st.title("Claude Code Usage")
    _render_settings_button()
    _render_ingest_timer()

    if not _db_has_rows():
        st.info(
            "No usage data found yet. Once you've used Claude Code, its "
            "transcripts will be ingested here \u2014 or use **Purge & rebuild "
            "data** from the \u2699 settings menu to try again."
        )
        return

    df_all = load_data(get_db_mtime())
    if df_all.empty:
        st.warning("Database has no rows yet.")
        return

    df, fmeta = filter_bar(df_all)
    if df.empty:
        st.info("No events match the current filters.")
        return

    prior_df = previous_period(df_all, fmeta)
    render_cost_hero(df, prior_df)
    render_kpi_band(df, prior_df)

    # Ranked from the FULL history, not the filtered slice -- see
    # build_model_color_map's docstring for why that matters.
    top_models, model_color_map = build_model_color_map(df_all)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Overview", "Explore", "Insights", "PRs", "Trends", "Ask"]
    )
    with tab1:
        tab_overview(df, model_color_map, top_models)
    with tab2:
        tab_explore(df)
    with tab3:
        tab_insights(df)
    with tab4:
        tab_prs(df)
    with tab5:
        tab_trends(df)
    with tab6:
        tab_ask()


if __name__ == "__main__":
    main()
