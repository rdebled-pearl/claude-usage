#!/bin/bash
cd "$(dirname "$0")"
exec ./venv/bin/streamlit run dashboard.py --server.headless true --server.address 127.0.0.1 --server.port 8501
