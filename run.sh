#!/usr/bin/env bash
# Start the simulation suite.
set -e
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
