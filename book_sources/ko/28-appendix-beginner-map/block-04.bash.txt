#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements-full.txt
python verify.py
