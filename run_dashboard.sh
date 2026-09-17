#!/bin/bash
cd "$(dirname "$0")"
exec ./venv/bin/streamlit run dashboard.py --server.headless true --server.port 8501
