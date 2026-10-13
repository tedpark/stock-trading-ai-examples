#!/usr/bin/env bash
set -euo pipefail
python -m uvicorn ch18_lab:app --host 127.0.0.1 --port 8766
